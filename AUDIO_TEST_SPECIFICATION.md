# SoundDevice Audio Test Specification

## Overview

This specification defines a standalone audio test to verify the SoundDevice implementation works correctly and establish proper integration patterns for the voice agent application.

## Test Objectives

1. **Verify SoundDevice Installation**: Confirm sounddevice library is properly installed and accessible
2. **Test Audio Device Detection**: Enumerate and validate available input/output devices
3. **Test Basic Audio Recording**: Capture audio from microphone using SoundDevice
4. **Test Audio Playback**: Play audio through speakers using SoundDevice
5. **Test Real-time Processing**: Verify low-latency audio streaming capabilities
6. **Cross-platform Compatibility**: Ensure functionality across Windows, macOS, and Linux

## Test Implementation Structure

### 1. Basic SoundDevice Functionality Test

```python
#!/usr/bin/env python3
"""
Standalone SoundDevice Audio Test
Tests core audio functionality without modifying existing voice agent code.
"""

import sounddevice as sd
import numpy as np
import time
import logging
from typing import Optional, List, Tuple

class SoundDeviceAudioTest:
    """Test class for verifying SoundDevice functionality."""

    def __init__(self):
        self.sample_rate = 16000  # Standard for speech processing
        self.channels = 1         # Mono audio
        self.dtype = np.float32   # Audio data type
        self.block_size = 1024    # Audio buffer size

    def test_device_enumeration(self) -> bool:
        """Test 1: Enumerate available audio devices."""

    def test_basic_recording(self, duration: float = 3.0) -> bool:
        """Test 2: Record audio for specified duration."""

    def test_basic_playback(self, audio_data: np.ndarray) -> bool:
        """Test 3: Play back recorded audio."""

    def test_realtime_streaming(self, duration: float = 10.0) -> bool:
        """Test 4: Real-time audio streaming (record + immediate playback)."""

    def test_voice_activity_simulation(self) -> bool:
        """Test 5: Simulate voice activity detection on audio stream."""
```

### 2. Device Detection and Validation

The test should:

- List all available input devices
- List all available output devices
- Identify default devices
- Validate device capabilities (sample rates, channels)
- Handle device selection gracefully

**Expected Output Format:**

```
Available Input Devices:
  0: Built-in Microphone (2 channels, 44100 Hz max)
  1: USB Headset Microphone (1 channel, 48000 Hz max)

Available Output Devices:
  0: Built-in Speakers (2 channels, 44100 Hz max)
  1: USB Headset Speakers (2 channels, 48000 Hz max)

Default Input Device: 0
Default Output Device: 0
```

### 3. Basic Recording Test

**Test Parameters:**

- Duration: 3 seconds
- Sample Rate: 16000 Hz
- Channels: 1 (mono)
- Data Type: float32

**Success Criteria:**

- Successfully capture audio without errors
- Audio data contains non-zero values (detecting actual input)
- No buffer overruns or underruns
- Proper sample count (duration × sample_rate)

### 4. Basic Playback Test

**Test Parameters:**

- Use previously recorded audio
- Same audio format as recording
- Clear, audible playback

**Success Criteria:**

- Audio plays without distortion
- No buffer underruns
- Playback completes successfully

### 5. Real-time Streaming Test

**Test Implementation:**

```python
def test_realtime_streaming(self, duration: float = 10.0) -> bool:
    """
    Test real-time audio processing:
    1. Start input stream
    2. Start output stream
    3. Route input directly to output with minimal latency
    4. Monitor for dropouts or errors
    """

    def audio_callback(indata, outdata, frames, time, status):
        """Real-time audio processing callback."""
        if status:
            print(f"Audio status: {status}")
        # Route input to output (echo test)
        outdata[:] = indata

    # Implementation details...
```

**Success Criteria:**

- Low latency audio routing (< 50ms)
- No audio dropouts
- Clear audio quality
- Stable operation for full test duration

### 6. Error Handling and Recovery

The test should verify:

- **Device Disconnection**: Handle USB device removal gracefully
- **Sample Rate Mismatch**: Proper error reporting for unsupported rates
- **Buffer Size Issues**: Adaptive buffer sizing
- **Permission Errors**: Clear error messages for microphone access

### 7. Platform-Specific Tests

#### macOS Specific

- Test Core Audio backend
- Verify microphone permissions
- Handle audio unit initialization

#### Windows Specific

- Test WASAPI backend
- Verify DirectSound compatibility
- Handle Windows audio service interactions

#### Linux Specific

- Test ALSA backend (through PortAudio)
- Test PulseAudio compatibility
- Handle audio server connections

## Integration Specification

### AudioManager Integration Pattern

Based on test results, the AudioManager should follow this pattern:

```python
class AudioManager:
    def __init__(self, config: AudioConfig):
        # 1. Device validation
        self._validate_devices()

        # 2. Stream initialization with error handling
        self._init_streams_with_fallbacks()

        # 3. Callback setup with proper error reporting
        self._setup_callbacks()

    def _validate_devices(self):
        """Validate selected devices are available and compatible."""

    def _init_streams_with_fallbacks(self):
        """Initialize streams with fallback device selection."""

    def _setup_callbacks(self):
        """Setup audio callbacks with proper error handling."""
```

### Error Recovery Strategies

1. **Device Selection Fallbacks**:
   - Primary device → Default device → Any compatible device
2. **Sample Rate Adaptation**:
   - Preferred rate → Device native rate → Resampling

3. **Buffer Size Optimization**:
   - Target latency → Stable operation balance

## Expected Test Results

### Successful Test Output

```
[INFO] SoundDevice Audio Test Starting...
[INFO] ✅ Device enumeration: PASSED
[INFO] ✅ Basic recording (3.0s): PASSED - 48000 samples captured
[INFO] ✅ Basic playback: PASSED - Audio played successfully
[INFO] ✅ Real-time streaming (10.0s): PASSED - Avg latency: 23ms
[INFO] ✅ Voice activity simulation: PASSED
[INFO] 🎉 All tests PASSED - SoundDevice implementation ready
```

### Error Scenarios to Handle

```
[ERROR] No input devices available
[ERROR] Selected device not found (ID: 5)
[ERROR] Sample rate 16000 Hz not supported, using 44100 Hz
[WARNING] Buffer underrun detected, increasing buffer size
[INFO] Device disconnected, switching to default device
```

## Usage Instructions

1. **Run the test**:

   ```bash
   python test_sounddevice_audio_specification.py
   ```

2. **Interpret results**:
   - All tests should pass for successful integration
   - Note any warnings about device compatibility
   - Verify latency is acceptable for voice applications

3. **Integration validation**:
   - Use test results to validate AudioManager configuration
   - Apply any device-specific optimizations discovered
   - Implement error handling patterns proven in testing

## Success Criteria for Voice Agent Integration

- [ ] All basic tests pass without errors
- [ ] Audio latency < 100ms for real-time processing
- [ ] Robust device enumeration and selection
- [ ] Graceful error handling and recovery
- [ ] Cross-platform compatibility confirmed
- [ ] No audio dropouts during extended operation
- [ ] Clear audio quality suitable for speech recognition

This specification provides a comprehensive framework for testing SoundDevice integration without modifying existing voice agent code, ensuring robust audio functionality across all supported platforms.
