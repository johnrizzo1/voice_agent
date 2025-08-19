# Speech-to-Text Configuration

## Overview

The voice agent supports multiple STT backends with intelligent fallback mechanisms:

1. **Whisper** (Primary) - High accuracy, supports multiple model sizes
2. **Vosk** (Fallback) - Lightweight, good for streaming
3. **Dummy** (Final fallback) - Placeholder mode when no models available

## Configuration

### Basic STT Settings

```yaml
stt:
  model: "whisper-base" # Primary model (whisper-base, whisper-small, etc.)
  language: "auto" # Language code or "auto" for detection
  streaming: true # Enable streaming recognition
  fallback_vosk_model: "vosk-model-en-us-0.22" # Vosk model for fallback
  allow_dummy_fallback: true # Allow dummy mode if all backends fail
```

### Model Options

#### Whisper Models

- `whisper-tiny`: Fastest, least accurate (~39 MB)
- `whisper-base`: Good balance (~74 MB)
- `whisper-small`: Better accuracy (~244 MB)
- `whisper-medium`: High accuracy (~769 MB)
- `whisper-large`: Best accuracy (~1550 MB)

#### Vosk Models

- `vosk-model-en-us-0.22`: Standard English model (~50 MB)
- `vosk-model-small-en-us-0.15`: Smaller English model (~40 MB)
- `vosk-model-en-us-0.22-lgraph`: Large graph model (~128 MB)

## Initialization Flow

1. **Primary Backend**: Attempts to load the configured `model`
   - If model name contains "whisper": tries Whisper
   - If model name contains "vosk": tries Vosk
   - Otherwise: tries Whisper first

2. **Fallback Backend**: If primary fails, tries alternate
   - Whisper failure → Vosk using `fallback_vosk_model`
   - Vosk failure → Whisper (if available)

3. **Dummy Backend**: If all backends fail and `allow_dummy_fallback: true`
   - Returns placeholder text: "[Dummy STT: Voice input detected but STT backend unavailable]"
   - Keeps UI in ready state instead of error
   - Logs warning with installation instructions

## Auto-Download

The system automatically downloads Vosk models if missing:

- Downloads from official Vosk model repository
- Extracts to `~/.cache/voice_agent/vosk_models/`
- Supports common English models out of the box
- Falls back to dummy mode if download fails

## Troubleshooting

### Common Issues

**"[Dummy STT: Voice input detected but STT backend unavailable]"**

- Install Whisper: `pip install faster-whisper`
- Or install Vosk: `pip install vosk` (model auto-downloads)
- Check logs for specific initialization errors

**Whisper fails on macOS**

- Uses standard whisper instead of faster-whisper
- Applies process isolation to avoid CoreAudio conflicts
- Falls back to CPU-only mode

**Vosk model not found**

- Check internet connection for auto-download
- Manually download to `~/.cache/voice_agent/vosk_models/`
- Verify model name in `fallback_vosk_model` config

### Logging

Enable debug logging to see detailed initialization:

```yaml
logging:
  level: "DEBUG"
```

Look for these log messages:

- `"WhisperModel available: True/False"`
- `"Whisper initialization failed: <error>"`
- `"Vosk model path: <path> (exists: True/False)"`
- `"Downloading Vosk model <name>..."`
- `"STT using dummy backend. Install faster-whisper..."`

## Performance Tuning

### For Speed

```yaml
stt:
  model: "whisper-tiny"
  fallback_vosk_model: "vosk-model-small-en-us-0.15"
```

### For Accuracy

```yaml
stt:
  model: "whisper-small"
  fallback_vosk_model: "vosk-model-en-us-0.22-lgraph"
```

### Disable Fallback

```yaml
stt:
  allow_dummy_fallback: false # Fail with error instead of dummy mode
```
