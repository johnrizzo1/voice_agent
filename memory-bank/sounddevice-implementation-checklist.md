# SoundDevice Implementation Checklist

## Phase 1: Foundation Setup ✅ Ready to Execute

### 1.1 Dependency Management

- [ ] Update `setup.py` - remove pyaudio, add sounddevice>=0.4.6
- [ ] Update `devenv.nix` - add sounddevice to Python packages
- [ ] Update `devenv.yaml` - add sounddevice dependency
- [ ] Test installation on development environment
- [ ] Verify PortAudio system dependency still works

### 1.2 Configuration Updates

- [ ] Update `src/voice_agent/config/default.yaml` - add SoundDevice audio settings
- [ ] Add new audio configuration fields (dtype, latency, buffer_size)
- [ ] Maintain backward compatibility with existing configs
- [ ] Update configuration validation in `config.py`

### 1.3 Project Setup

- [ ] Create feature branch: `feature/sounddevice-migration`
- [ ] Backup current `audio_manager.py` as `audio_manager_pyaudio_backup.py`
- [ ] Create implementation tracking document

## Phase 2: Core AudioManager Migration ✅ Ready to Execute

### 2.1 Remove ALSA-Specific Code

- [ ] Delete ALSA error suppression code (lines 16-87 in audio_manager.py)
  - [ ] Remove `_py_alsa_error_handler()` function
  - [ ] Remove `_suppress_alsa_jack_errors()` context manager
  - [ ] Remove ctypes ALSA library loading
  - [ ] Remove ERROR_HANDLER_FUNC and related ctypes setup
- [ ] Clean up related imports (ctypes, threading for ALSA)

### 2.2 Update Imports and Dependencies

- [ ] Replace `import pyaudio` with `import sounddevice as sd`
- [ ] Add `import asyncio` for future async support
- [ ] Update numpy import usage for audio processing
- [ ] Remove pyaudio-specific threading imports

### 2.3 Refactor Audio Input (Recording)

- [ ] Replace PyAudio InputStream creation in `listen()` method
- [ ] Update `_input_callback()` signature for SoundDevice
- [ ] Modify `_wait_for_speech()` for SoundDevice streams
- [ ] Update `_is_speech()` VAD integration
- [ ] Test microphone input with new implementation

### 2.4 Refactor Audio Output (Playback)

- [ ] Replace PyAudio OutputStream in `play_audio()` method
- [ ] Update audio data handling for SoundDevice format
- [ ] Implement non-blocking playback with SoundDevice
- [ ] Add proper sample rate conversion
- [ ] Test TTS audio playback

### 2.5 Device Management Overhaul

- [ ] Replace `get_device_info()` with SoundDevice device query
- [ ] Update `_has_input_device()` and `_has_output_device()` methods
- [ ] Implement SoundDevice device enumeration
- [ ] Add default device selection logic
- [ ] Test device discovery on multiple platforms

### 2.6 Error Handling Modernization

- [ ] Replace ALSA error handling with unified cross-platform approach
- [ ] Add `_handle_audio_error()` method for SoundDevice errors
- [ ] Implement fallback device selection
- [ ] Add proper logging for audio errors
- [ ] Test error scenarios (device unplugged, etc.)

## Phase 3: Integration & Validation ✅ Ready to Execute

### 3.1 Service Integration Testing

- [ ] Verify STT service integration - ensure numpy array compatibility
- [ ] Verify TTS service integration - test audio playback chain
- [ ] Test voice activity detection with new audio input
- [ ] Validate barge-in functionality during TTS playback
- [ ] Test conversation flow end-to-end

### 3.2 Configuration Testing

- [ ] Test with default configuration settings
- [ ] Test with custom device selection
- [ ] Test with different sample rates and chunk sizes
- [ ] Validate audio quality settings
- [ ] Test configuration edge cases

### 3.3 Cross-Platform Validation

- [ ] Linux testing - verify ALSA backend works without direct calls
- [ ] macOS testing - verify Core Audio backend
- [ ] Windows testing - verify WASAPI backend
- [ ] Test device enumeration on all platforms
- [ ] Validate audio latency improvements

### 3.4 Performance & Quality Testing

- [ ] Measure audio latency improvement (target: <10ms vs >50ms)
- [ ] Test CPU usage reduction
- [ ] Validate audio quality (no degradation)
- [ ] Test memory usage patterns
- [ ] Benchmark against previous implementation

## Phase 4: Cleanup & Documentation ✅ Ready to Execute

### 4.1 Code Cleanup

- [ ] Remove unused PyAudio imports across project
- [ ] Update example files that reference PyAudio
- [ ] Clean up any remaining ALSA references in comments
- [ ] Update type hints for SoundDevice integration
- [ ] Code review and refactoring

### 4.2 Documentation Updates

- [ ] Update README.md installation instructions
- [ ] Update getting started guides
- [ ] Document new audio configuration options
- [ ] Update troubleshooting guides
- [ ] Create migration notes for users

### 4.3 Testing & Validation

- [ ] Run full test suite
- [ ] Execute integration tests
- [ ] Test example applications
- [ ] Validate multi-agent functionality
- [ ] Performance regression testing

### 4.4 Deployment Preparation

- [ ] Update CI/CD pipeline dependencies
- [ ] Test package building and distribution
- [ ] Prepare release notes
- [ ] Update version numbers
- [ ] Final code review

## Success Criteria ✅

- [ ] All PyAudio code successfully replaced with SoundDevice
- [ ] ALSA error suppression code completely removed
- [ ] Cross-platform compatibility verified (Windows/macOS/Linux)
- [ ] Audio latency improved to <10ms
- [ ] All existing functionality preserved
- [ ] Performance improvements validated
- [ ] Documentation updated
- [ ] Test suite passes completely

## Risk Mitigation ✅

- [ ] Backup implementation available for rollback
- [ ] Feature branch isolated from main development
- [ ] Incremental testing at each phase
- [ ] Cross-platform validation before deployment
- [ ] Performance benchmarking to prevent regressions

---

**Implementation Ready:** All phases planned with detailed checklists
**Estimated Timeline:** 2-3 days for complete migration
**Risk Level:** LOW - SoundDevice is a mature PyAudio alternative
