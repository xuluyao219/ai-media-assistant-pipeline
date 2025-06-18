# Multimodal AI Media Analysis System

An easy-to-use AI media analysis system integrating CLIP, YOLOv8, and Whisper V3, with web interface support for upload and analysis.

## Features

- **Scene Understanding**: Scene segmentation and understanding using CLIP
- **Object Detection**: Object detection and tracking using YOLOv8  
- **Speech Recognition**: Multi-language transcription using Whisper Large V3
- **Natural Language Query**: Cross-modal retrieval based on CLIP
- **Web Interface**: Modern drag-and-drop upload interface
- **Multi-format Support**: MP4, AVI, MOV, MKV, MP3, WAV, FLAC, etc.

## System Requirements

- Python 3.8+
- CUDA 11.8+ (recommended)
- RTX 3080 or higher GPU (recommended)
- 16GB+ RAM
- 10GB+ VRAM (for Whisper Large V3)

## Installation

### 1. Install Dependencies
```bash
# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Run the System
```bash
python run.py
```

### 3. Access Web Interface
Open browser and visit: http://localhost:5000

## Usage

### Web Interface
1. Open browser and visit http://localhost:5000
2. Drag and drop or select media files to upload
3. Click "Start Analysis"
4. View analysis results
5. Use natural language query to search content
