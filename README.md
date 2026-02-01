# Qwen3-ASR CLI

基於 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的命令列語音辨識工具，支援多種音檔格式並可自動轉換為繁體中文。

## 功能特色

- 🎵 **多格式支援**: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`
- 🌐 **多語言辨識**: 支援 52 種語言和方言的自動識別
- 🔄 **繁體中文轉換**: 使用 OpenCC 將輸出轉換為臺灣繁體中文 (zh_TW)
- 📦 **長音檔分段處理**: 自動切分長音檔以避免記憶體不足
- ⏱️ **時間戳記**: 支援逐字或句子級時間戳記輸出
- 📊 **處理時間統計**: 顯示即時處理速度 (RTF)

## 安裝

```bash
# 使用 uv (推薦)
git clone https://github.com/allencyhsu/qwen3-asr.git
cd qwen3-asr
uv sync

# 或使用 pip
pip install -e .
```

### 前置需求

- Python >= 3.12
- CUDA 支援的 GPU (建議 24GB+ VRAM)
- FFmpeg (處理 MP3/M4A 格式需要)

```bash
# Ubuntu/Debian
sudo apt install ffmpeg
```

## 使用方式

### 基本使用

```bash
# 轉錄單一檔案
uv run qwen3-asr audio.mp3

# 轉錄多個檔案
uv run qwen3-asr audio1.wav audio2.m4a

# 轉錄整個目錄
uv run qwen3-asr --dir ./audio_folder

# 自動儲存到檔案 (audio_qwen3.txt)
uv run qwen3-asr audio.m4a --save
```

### 繁體中文輸出

```bash
# 使用 --traditional 或 -tw 轉換為繁體中文
uv run qwen3-asr audio.m4a --traditional
uv run qwen3-asr audio.m4a -tw -s  # 轉換並儲存
```

### 時間戳記

```bash
# 逐字/逐詞時間戳記 (詳細)
uv run qwen3-asr audio.m4a -t

# 句子級時間戳記 (合併後更易讀)
uv run qwen3-asr audio.m4a -t -mt
```

### 進階選項

```bash
# 指定語言 (跳過自動偵測)
uv run qwen3-asr audio.m4a --language Chinese

# 使用 0.6B 較小模型 (更快、佔用更少 VRAM)
uv run qwen3-asr audio.m4a --model-size 0.6B

# 儲存結果到指定檔案
uv run qwen3-asr audio.m4a --output result.txt

# 自訂輸出檔名後綴 (預設 _qwen3)
uv run qwen3-asr audio.m4a -s --suffix _whisper

# 調整分段長度 (預設 300 秒)
uv run qwen3-asr long_audio.m4a --chunk-duration 180

# 安靜模式 (不輸出文字稿，搭配 -s 儲存)
uv run qwen3-asr audio.m4a -q -s

# 組合使用
uv run qwen3-asr audio.m4a -l Chinese -tw -t -mt -s
```

### 完整參數列表

| 參數 | 簡寫 | 說明 |
|------|------|------|
| `--dir` | `-d` | 指定包含音檔的目錄 |
| `--language` | `-l` | 強制指定語言 (如 `Chinese`, `English`) |
| `--timestamps` | `-t` | 輸出逐字/逐詞時間戳記 |
| `--merge-timestamps` | `-mt` | 合併時間戳記為句子 (需搭配 `-t`) |
| `--model` | `-m` | 指定模型路徑 (覆蓋 `--model-size`) |
| `--model-size` | | 模型大小：`1.7B` (預設) 或 `0.6B` |
| `--chunk-duration` | `-c` | 分段長度 (秒)，預設 300 |
| `--output` | `-o` | 輸出到指定檔案 |
| `--traditional` | `-tw` | 轉換為繁體中文 (zh_TW) |
| `--save` | `-s` | 自動儲存到 `音檔名{suffix}.txt` |
| `--suffix` | | 輸出檔名後綴 (預設 `_qwen3`) |
| `--quiet` | `-q` | 安靜模式，不輸出文字稿 |

## 模型

支援兩種模型大小：
- **1.7B** (預設) - 較大、較精準
- **0.6B** - 較小、較快、佔用較少 VRAM

如果當前目錄有對應的模型資料夾 (如 `./Qwen3-ASR-1.7B`)，會優先使用本地模型。

### 下載本地模型

```bash
# ASR 模型 (必要)
hf download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B

# ForcedAligner 模型 (時間戳記功能需要)
hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./Qwen3-ForcedAligner-0.6B
```

使用 ModelScope (中國大陸推薦):

```bash
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B
```

## 記憶體優化

處理長音檔時，程式會自動將音檔切分為較短的片段進行處理。如果遇到 CUDA OOM 錯誤，可以：

1. 減少 `--chunk-duration` (例如 `-c 120`)
2. 確保 GPU 有足夠的可用記憶體

## 致謝

- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - 阿里雲通義千問團隊
- [OpenCC](https://github.com/BYVoid/OpenCC) - 開放中文轉換

## 授權

MIT License
