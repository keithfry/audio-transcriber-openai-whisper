# Audio Transcription System

A real-time audio transcription application using OpenAI Whisper for accurate speech-to-text conversion. Features voice activity detection for automatic recording control, transcription verification with audio playback, and configurable model selection.

## Features

- 🎙️ **Real-time microphone audio capture** with voice activity detection
- 🤖 **Speech-to-text using OpenAI Whisper** (multiple model sizes available)
- ⚡ **Automatic recording control** - stops when you finish speaking
- 🔍 **Transcription verification** - confirm accuracy with audio playback
- 📝 **Handles long audio recordings** automatically (no 30-second limit)
- 🔧 **Configurable recording parameters** via environment variables
- 🎯 **Optimized for real-time use** with minimal delay
- 💬 **Interactive session management** with continue/stop prompts

## Quick Start

1. **Activate virtual environment:**
   ```bash
   source .env/bin/activate
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

3. **First run setup:**
   - OpenAI Whisper model will be downloaded automatically
   - Speak clearly when prompted
   - Verify transcriptions and use audio playback if needed
   - Answer 'y' to continue or 'n' to exit

## Installation & Setup

The project uses `uv` package manager with a virtual environment in `.env/`:

```bash
# Virtual environment is already set up in .env/
# Just activate and run:
source .env/bin/activate
python main.py
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Transcriber type (recommended: openai)
TRANSCRIBER_TYPE=openai

# OpenAI Whisper model size
OPENAI_MODEL_SIZE=base

# Audio settings
SAMPLE_RATE=16000
CHANNELS=1
MAX_RECORDING_DURATION=300.0

# Voice Activity Detection
VAD_AGGRESSIVENESS=2
SILENCE_THRESHOLD=1.0

# Show audio devices on startup
SHOW_AUDIO_DEVICES=false
```

### Available OpenAI Whisper Models

- `tiny` - Fastest, basic accuracy (~39 MB)
- `base` - Good balance of speed/accuracy (~74 MB) **[Default]**
- `small` - Better accuracy (~244 MB)
- `medium` - High accuracy (~769 MB)
- `large` - Best accuracy (~1550 MB)

## Usage Examples

### Basic Usage
```bash
source .env/bin/activate
python main.py
```

### Show Available Audio Devices
```bash
python main.py --show-devices
```

### Using Different Model Size
```bash
OPENAI_MODEL_SIZE=small python main.py
```

## Application Workflow

1. **Start Recording** - Automatic voice activity detection begins
2. **Speak Naturally** - Recording stops automatically when you finish
3. **View Transcription** - Text appears with session information
4. **Verify Accuracy** - Confirm if transcription is correct (y/n)
5. **Audio Playback** - If incorrect, hear the original recording
6. **Continue/Exit** - Choose to record again or exit

## Key Features Explained

### Voice Activity Detection (VAD)
- Uses WebRTC VAD for speech detection
- Automatically stops recording when speech ends
- No manual timing or button pressing required
- Configurable sensitivity levels

### Transcription Verification
- Interactive prompts to verify transcription accuracy
- Audio playback feature to hear original recording
- Helps identify audio quality or clarity issues
- Provides feedback for improving recording conditions

### OpenAI Whisper Integration
- Uses the original OpenAI Whisper implementation
- Handles audio of any length automatically
- More reliable than HuggingFace alternatives
- Multiple model sizes for different accuracy/speed needs

## Project Structure

```
.
├── main.py                    # Main application entry point
├── audio_capture.py           # Audio recording with VAD
├── audio_playback.py          # Audio verification playback
├── transcriber_openai.py      # OpenAI Whisper implementation
├── config.py                  # Configuration management
├── pyproject.toml            # Project dependencies (uv)
├── requirements.txt          # Alternative dependency list
├── test/                     # Test scripts
│   ├── test_integrated_app.py    # Full integration test
│   └── test_openai_whisper.py    # Whisper-only test
├── backup/                   # Old/unused files
└── .env/                     # Virtual environment (uv)
```

## Testing

### Run Integration Test
```bash
source .env/bin/activate
python test/test_integrated_app.py
```

### Test OpenAI Whisper Only
```bash
source .env/bin/activate
python test/test_openai_whisper.py
```

## Troubleshooting

### Common Issues

1. **No audio recorded:**
   - Check microphone permissions in system settings
   - Try `python main.py --show-devices` to verify audio devices
   - Ensure microphone is not muted or disabled
   - Test with different VAD sensitivity: `VAD_AGGRESSIVENESS=1`

2. **Recording doesn't stop:**
   - Reduce background noise
   - Speak closer to microphone
   - Adjust silence threshold: `SILENCE_THRESHOLD=0.5`
   - Try higher VAD aggressiveness: `VAD_AGGRESSIVENESS=3`

3. **Poor transcription quality:**
   - Use larger model: `OPENAI_MODEL_SIZE=small` or `medium`
   - Improve recording environment (reduce noise)
   - Speak more clearly and at normal pace
   - Check audio playback to verify capture quality

4. **Model loading errors:**
   - Ensure stable internet connection for initial download
   - Check available disk space (up to ~1.5GB for large models)
   - Try smaller model if having memory issues

### Audio Device Configuration

List available audio devices:
```bash
python main.py --show-devices
```

If using non-default audio device, check system audio settings.

## Requirements

- **Python 3.9+**
- **Working microphone**
- **Internet connection** (for initial model download)
- **Disk space:** 74MB (base) to 1.5GB (large model)
- **Optional:** GPU for faster processing (automatically detected)

## Dependencies

Key packages (managed via `pyproject.toml`):
- `openai-whisper` - Speech recognition
- `sounddevice` - Audio capture
- `webrtcvad` - Voice activity detection
- `numpy` - Audio processing
- `soundfile` - Audio file operations

## License

This project uses OpenAI Whisper and other open-source components. Please check individual package licenses for commercial use.