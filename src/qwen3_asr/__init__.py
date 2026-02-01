"""
Qwen3-ASR Demo with support for .wav, .mp3, and .m4a audio files.

Usage:
    qwen3-asr <audio_file>           # Single file
    qwen3-asr <file1> <file2> ...    # Multiple files
    qwen3-asr --dir <directory>      # All audio files in directory
    
Options:
    --language LANG     Force language (e.g., "Chinese", "English")
    --timestamps        Return timestamps for each word/segment
    --model PATH        Model path (default: ./Qwen3-ASR-1.7B or Qwen/Qwen3-ASR-1.7B)
    --chunk-duration    Chunk duration in seconds for long audio (default: 300)
    --output FILE       Output transcription to file instead of stdout
    --traditional       Convert output to Traditional Chinese (zh_TW)
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
import opencc

from qwen_asr import Qwen3ASRModel


# OpenCC converter for Simplified to Traditional Chinese
_opencc_converter = None


def get_opencc_converter():
    """Get or create OpenCC converter (lazy initialization)."""
    global _opencc_converter
    if _opencc_converter is None:
        _opencc_converter = opencc.OpenCC('s2twp')  # Simplified to Traditional (Taiwan phrases)
    return _opencc_converter


def convert_to_traditional(text: str) -> str:
    """Convert Simplified Chinese text to Traditional Chinese (zh_TW)."""
    if not text:
        return text
    converter = get_opencc_converter()
    return converter.convert(text)


# Supported audio extensions
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

# Default chunk duration in seconds (5 minutes)
DEFAULT_CHUNK_DURATION = 300

# Overlap between chunks in seconds
CHUNK_OVERLAP = 1

# Default pause threshold for merging timestamps (seconds)
DEFAULT_PAUSE_THRESHOLD = 0.5

# Sentence-ending punctuation for timestamp merging
SENTENCE_END_PUNCTUATION = set('。！？.!?')
CLAUSE_PUNCTUATION = set('，,；;：:')


def merge_timestamps(
    timestamps: List[dict],
    pause_threshold: float = DEFAULT_PAUSE_THRESHOLD,
) -> List[dict]:
    """
    Merge character-level timestamps into sentence-level timestamps.
    
    Merges consecutive timestamps based on:
    1. Pause detection: Gap > pause_threshold starts new sentence
    2. Punctuation: Sentence-ending punctuation (。！？.!?) starts new sentence
    
    Args:
        timestamps: List of dicts with 'text', 'start', 'end' keys
        pause_threshold: Gap in seconds to trigger new sentence (default: 0.5)
        
    Returns:
        List of merged timestamp dicts
    """
    if not timestamps:
        return []
    
    merged = []
    current_texts = []
    current_start = timestamps[0]['start']
    current_end = timestamps[0]['end']
    
    for i, ts in enumerate(timestamps):
        text = ts['text']
        start = ts['start']
        end = ts['end']
        
        # Check if we should start a new sentence
        should_split = False
        
        if i > 0:
            # Check for pause (gap between previous end and current start)
            gap = start - current_end
            if gap > pause_threshold:
                should_split = True
        
        if should_split and current_texts:
            # Save current sentence
            merged.append({
                'text': ''.join(current_texts),
                'start': current_start,
                'end': current_end,
            })
            # Start new sentence
            current_texts = []
            current_start = start
        
        # Add current character/word
        current_texts.append(text)
        current_end = end
        
        # Check for sentence-ending punctuation
        if text and text[-1] in SENTENCE_END_PUNCTUATION:
            merged.append({
                'text': ''.join(current_texts),
                'start': current_start,
                'end': current_end,
            })
            current_texts = []
            if i + 1 < len(timestamps):
                current_start = timestamps[i + 1]['start']
    
    # Don't forget the last sentence
    if current_texts:
        merged.append({
            'text': ''.join(current_texts),
            'start': current_start,
            'end': current_end,
        })
    
    return merged


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return (numpy array, sample rate).
    Supports .wav, .mp3, .m4a, .flac, .ogg, .aac formats.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Tuple of (audio_data as np.ndarray, sample_rate as int)
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {ext}. Supported: {SUPPORTED_EXTENSIONS}")
    
    # For .wav and some other formats, try soundfile first (faster)
    if ext in {".wav", ".flac", ".ogg"}:
        try:
            audio_data, sample_rate = sf.read(str(file_path), dtype="float32")
            return np.asarray(audio_data, dtype=np.float32), sample_rate
        except Exception:
            pass  # Fall back to pydub
    
    # Use pydub for .mp3, .m4a, .aac and as fallback for others
    # pydub uses ffmpeg under the hood
    try:
        if ext == ".mp3":
            audio = AudioSegment.from_mp3(str(file_path))
        elif ext == ".m4a":
            audio = AudioSegment.from_file(str(file_path), format="m4a")
        elif ext == ".aac":
            audio = AudioSegment.from_file(str(file_path), format="aac")
        else:
            audio = AudioSegment.from_file(str(file_path))
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Get sample rate and audio data
        sample_rate = audio.frame_rate
        
        # Convert to numpy array (float32, normalized to [-1, 1])
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize based on sample width
        if audio.sample_width == 1:  # 8-bit
            samples = (samples - 128) / 128.0
        elif audio.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio.sample_width == 4:  # 32-bit
            samples = samples / 2147483648.0
        
        return samples, sample_rate
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file '{file_path}': {e}")


def find_audio_files(directory: str) -> List[Path]:
    """Find all supported audio files in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    audio_files = []
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(dir_path.glob(f"*{ext}"))
        audio_files.extend(dir_path.glob(f"*{ext.upper()}"))
    
    return sorted(audio_files)


def split_audio_into_chunks(
    audio_data: np.ndarray, 
    sample_rate: int, 
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    overlap: float = CHUNK_OVERLAP,
) -> List[Tuple[np.ndarray, float]]:
    """
    Split audio into chunks for memory-efficient processing.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds
        
    Returns:
        List of (chunk_data, start_time) tuples
    """
    total_samples = len(audio_data)
    total_duration = total_samples / sample_rate
    
    # If audio is short enough, return as single chunk
    if total_duration <= chunk_duration:
        return [(audio_data, 0.0)]
    
    chunks = []
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step_samples = chunk_samples - overlap_samples
    
    start_sample = 0
    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk = audio_data[start_sample:end_sample]
        start_time = start_sample / sample_rate
        chunks.append((chunk, start_time))
        
        start_sample += step_samples
        
        # Avoid tiny last chunk
        if total_samples - start_sample < chunk_samples * 0.1:
            break
    
    return chunks


def create_asr_model(
    model_path: Optional[str] = None,
    use_timestamps: bool = False,
    model_size: str = "1.7B",
) -> Qwen3ASRModel:
    """Create and return the ASR model."""
    
    # Determine model path
    if model_path is None:
        model_name = f"Qwen3-ASR-{model_size}"
        # Check for local model first
        local_model = Path(f"./{model_name}")
        if local_model.exists():
            model_path = str(local_model)
        else:
            model_path = f"Qwen/{model_name}"
    
    print(f"Loading ASR model from: {model_path}")
    
    model_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="cuda:1",
        max_inference_batch_size=8,  # Reduced for memory efficiency
        max_new_tokens=512,  # Increased for longer chunks
    )
    
    # Add forced aligner for timestamp support
    if use_timestamps:
        local_aligner = Path("./Qwen3-ForcedAligner-0.6B")
        aligner_path = str(local_aligner) if local_aligner.exists() else "Qwen/Qwen3-ForcedAligner-0.6B"
        
        model_kwargs["forced_aligner"] = aligner_path
        model_kwargs["forced_aligner_kwargs"] = dict(
            dtype=torch.bfloat16,
            device_map="cuda:1",
        )
    
    model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)
    return model


def transcribe_audio_chunked(
    model: Qwen3ASRModel,
    audio_data: np.ndarray,
    sample_rate: int,
    language: Optional[str] = None,
    return_timestamps: bool = False,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    merge_ts: bool = False,
) -> str:
    """
    Transcribe audio with chunking for long files.
    
    Args:
        model: ASR model
        audio_data: Audio samples
        sample_rate: Sample rate
        language: Force language
        return_timestamps: Whether to return timestamps
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        Full transcription text
    """
    total_duration = len(audio_data) / sample_rate
    
    # Split into chunks
    chunks = split_audio_into_chunks(audio_data, sample_rate, chunk_duration)
    
    print(f"Processing {len(chunks)} chunk(s) of ~{chunk_duration}s each...")
    
    all_text = []
    all_timestamps = []
    
    for i, (chunk_data, start_time) in enumerate(chunks):
        chunk_duration_actual = len(chunk_data) / sample_rate
        print(f"  Chunk {i+1}/{len(chunks)}: {start_time:.1f}s - {start_time + chunk_duration_actual:.1f}s", end="")
        
        try:
            chunk_start_time = time.time()
            results = model.transcribe(
                audio=(chunk_data, sample_rate),
                language=language,
                return_time_stamps=return_timestamps,
            )
            chunk_elapsed = time.time() - chunk_start_time
            rtf = chunk_elapsed / chunk_duration_actual if chunk_duration_actual > 0 else 0
            print(f" → {chunk_elapsed:.1f}s (RTF: {rtf:.2f}x)")
            
            for result in results:
                if result.text:
                    all_text.append(result.text)
                
                if return_timestamps and result.time_stamps:
                    for ts in result.time_stamps:
                        # Adjust timestamps to absolute time
                        all_timestamps.append({
                            'text': ts.text,
                            'start': ts.start_time + start_time,
                            'end': ts.end_time + start_time,
                        })
            
            # Clear GPU cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"    Error in chunk {i+1}: {e}")
            continue
    
    # Merge timestamps into sentences (based on pauses and punctuation)
    if all_timestamps and merge_ts:
        all_timestamps = merge_timestamps(all_timestamps)
    
    return " ".join(all_text), all_timestamps


def transcribe_files(
    model: Qwen3ASRModel,
    audio_files: List[Union[str, Path]],
    language: Optional[str] = None,
    return_timestamps: bool = False,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    output_file: Optional[str] = None,
    convert_traditional: bool = False,
    auto_save: bool = False,
    merge_ts: bool = False,
    quiet: bool = False,
    suffix: str = "_qwen3",
) -> None:
    """Transcribe multiple audio files and print/save results.
    
    If auto_save is True, save transcription to .txt file with same name as audio file.
    """
    
    output_lines = []
    
    for audio_file in audio_files:
        file_path = Path(audio_file)
        header = f"\n{'='*60}\nFile: {file_path.name}\n{'='*60}"
        print(header)
        output_lines.append(header)
        
        try:
            # Load audio as numpy array
            audio_data, sample_rate = load_audio(str(file_path))
            duration = len(audio_data) / sample_rate
            info = f"Duration: {duration:.2f}s ({duration/60:.1f} min), Sample Rate: {sample_rate}Hz"
            print(info)
            output_lines.append(info)
            
            # Start timing
            file_start_time = time.time()
            
            # Transcribe with chunking
            text, timestamps = transcribe_audio_chunked(
                model=model,
                audio_data=audio_data,
                sample_rate=sample_rate,
                language=language,
                return_timestamps=return_timestamps,
                chunk_duration=chunk_duration,
                merge_ts=merge_ts,
            )
            
            # Convert to Traditional Chinese if requested
            if convert_traditional:
                text = convert_to_traditional(text)
                if timestamps:
                    for ts in timestamps:
                        ts['text'] = convert_to_traditional(ts['text'])
            
            # Print results (unless quiet mode)
            if not quiet:
                print(f"\nTranscription:")
                print(text)
            output_lines.append(f"\nTranscription:\n{text}")
            
            if return_timestamps and timestamps:
                if not quiet:
                    print("\nTimestamps:")
                output_lines.append("\nTimestamps:")
                for ts in timestamps:
                    ts_line = f"  [{ts['start']:.2f}s - {ts['end']:.2f}s] {ts['text']}"
                    if not quiet:
                        print(ts_line)
                    output_lines.append(ts_line)
            
            # Auto-save to .txt file with same name as audio
            if auto_save:
                txt_path = file_path.parent / f"{file_path.stem}{suffix}.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                    if return_timestamps and timestamps:
                        f.write('\n\nTimestamps:\n')
                        for ts in timestamps:
                            f.write(f"[{ts['start']:.2f}s - {ts['end']:.2f}s] {ts['text']}\n")
                print(f"\nSaved to: {txt_path}")
            
            # Print total time for this file
            file_elapsed = time.time() - file_start_time
            file_rtf = file_elapsed / duration if duration > 0 else 0
            time_info = f"\nTotal processing time: {file_elapsed:.1f}s (Audio: {duration:.1f}s, RTF: {file_rtf:.2f}x)"
            print(time_info)
            output_lines.append(time_info)
                        
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            print(error_msg)
            output_lines.append(error_msg)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\nOutput saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR: Transcribe audio files (wav, mp3, m4a, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qwen3-asr audio.mp3
  qwen3-asr audio1.wav audio2.m4a --language Chinese
  qwen3-asr --dir ./audio_folder --timestamps
  qwen3-asr long_audio.m4a --chunk-duration 180 --output result.txt
        """
    )
    
    parser.add_argument(
        "files",
        nargs="*",
        help="Audio file(s) to transcribe"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Directory containing audio files to transcribe"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Force language (e.g., 'Chinese', 'English'). Auto-detect if not specified."
    )
    parser.add_argument(
        "--timestamps", "-t",
        action="store_true",
        help="Return character/word-level timestamps (raw, very detailed)"
    )
    parser.add_argument(
        "--merge-timestamps", "-mt",
        action="store_true",
        help="Merge timestamps into sentences (requires --timestamps)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to ASR model (overrides --model-size)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["1.7B", "0.6B"],
        default="1.7B",
        help="Model size: 1.7B (default) or 0.6B"
    )
    parser.add_argument(
        "--chunk-duration", "-c",
        type=int,
        default=DEFAULT_CHUNK_DURATION,
        help=f"Chunk duration in seconds for long audio (default: {DEFAULT_CHUNK_DURATION})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output transcription to file"
    )
    parser.add_argument(
        "--traditional", "-tw",
        action="store_true",
        help="Convert output to Traditional Chinese (zh_TW)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Auto-save transcription to .txt file with same name as audio"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress transcription output (still saves to file with -s)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_qwen3",
        help="Output filename suffix (default: _qwen3)"
    )
    
    args = parser.parse_args()
    
    # Collect audio files
    audio_files = []
    
    if args.dir:
        audio_files.extend(find_audio_files(args.dir))
    
    for f in args.files:
        path = Path(f)
        if path.is_file():
            audio_files.append(path)
        elif path.is_dir():
            audio_files.extend(find_audio_files(str(path)))
        else:
            print(f"Warning: '{f}' not found, skipping.")
    
    if not audio_files:
        print("No audio files specified. Use --help for usage information.")
        print(f"\nSupported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio file(s) to process.")
    print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    print(f"Chunk duration: {args.chunk_duration}s")
    if args.traditional:
        print("Output will be converted to Traditional Chinese (zh_TW)")
    
    # Create model
    model = create_asr_model(
        model_path=args.model,
        use_timestamps=args.timestamps,
        model_size=args.model_size,
    )
    
    # Transcribe
    transcribe_files(
        model=model,
        audio_files=audio_files,
        language=args.language,
        return_timestamps=args.timestamps,
        chunk_duration=args.chunk_duration,
        output_file=args.output,
        convert_traditional=args.traditional,
        auto_save=args.save,
        merge_ts=args.merge_timestamps,
        quiet=args.quiet,
        suffix=args.suffix,
    )
    
    print("\n" + "="*60)
    print("Transcription complete!")


if __name__ == "__main__":
    main()
