# Decision Log

This file records architectural and implementation decisions using a list format.

## Decision

2025-08-14 18:20:18 - **PyTorch 2.6 Compatibility Fix: Monkey Patch Approach**

## Rationale

The Bark TTS library uses `torch.load()` internally, and PyTorch 2.6 changed the default `weights_only` parameter from `False` to `True`. This broke Bark model loading because the models contain numpy objects not on the default allowlist.

Alternative approaches considered:
1. Using `torch.serialization.safe_globals()` context manager - Failed because it only affects the current thread context, not the library's internal calls
2. Patching the bark library directly - Too invasive and would require maintaining a fork
3. Downgrading PyTorch - Would conflict with other dependencies and reduce security benefits
4. Monkey patching torch.load temporarily - **CHOSEN** as it's minimally invasive and preserves security for other uses

## Implementation Details

- Modified `_initialize_bark()` method in `src/voice_agent/core/tts_service.py`
- Created `safe_preload()` function that temporarily replaces `torch.load` with a version that forces `weights_only=False`
- Used try/finally block to ensure original `torch.load` is always restored
- Added proper error handling and type checking for cases where torch might be None
- Patch is applied only during Bark model loading, maintaining security for other torch.load calls

---

## Decision

2025-08-14 18:20:18 - **Development Environment: devenv/nix over requirements.txt**

## Rationale

The project uses devenv/nix for reproducible development environments instead of traditional Python requirements.txt. This provides:
- Fully reproducible builds across different systems
- Integrated system dependencies (portaudio, espeak-ng, etc.)
- CUDA support configuration
- Development tools integration (linters, formatters)

## Implementation Details

- Configured `devenv.nix` with all runtime and development dependencies
- Used both nixpkgs Python packages and venv requirements for packages not available in nixpkgs
- Set up proper PYTHONPATH configuration to use local source directory
- Added convenient process definitions for common development tasks (run, test, lint)

---

## Decision

2025-08-14 22:24:06 - **Multi-Backend TTS Architecture with Automatic Fallback**

## Rationale

Rather than relying on a single TTS engine, implemented a multi-backend architecture to ensure reliability and quality options:

1. **Bark TTS** (Primary): Neural network voices, most realistic quality
2. **Coqui TTS** (Secondary): High-quality neural voices, good fallback
3. **eSpeak-NG** (Tertiary): Lightweight, fast, synthetic but clear
4. **pyttsx3** (Fallback): System TTS, always available, cross-platform

This provides graceful degradation and allows users to choose quality vs. speed trade-offs.

## Implementation Details

- TTS service automatically detects available engines at initialization
- Engine priority: Bark → Coqui → eSpeak-NG → pyttsx3
- Configuration allows explicit engine selection via `engine: "auto"` or specific name
- All engines work completely offline after initial model downloads
- Error handling ensures fallback to next available engine if primary fails

---

## Decision

2025-08-14 22:24:06 - **STT Engine Selection: faster-whisper as Primary**

## Rationale

Selected faster-whisper over alternatives for primary STT:

**faster-whisper** (Primary):
- Optimized Whisper implementation with significant speed improvements
- Same accuracy as original Whisper but with better performance
- Good model size options (base, small, medium, large-v2)
- Excellent offline capabilities

**Vosk** (Alternative):
- Lightweight, good for streaming applications
- Lower accuracy than Whisper but faster startup
- Better for resource-constrained environments

## Implementation Details

- faster-whisper configured with automatic model downloads
- Model size configurable: base (39MB), small (244MB), medium (769MB), large-v2 (1550MB)
- CPU/GPU acceleration support
- Streaming capability for real-time transcription

---

## Decision

2025-08-14 22:24:06 - **LLM Integration: Ollama as Primary Provider**

## Rationale

Chose Ollama over direct model integration for several reasons:

**Advantages of Ollama**:
- Simplified model management (download, switching, updates)
- Optimized inference with automatic hardware acceleration
- RESTful API for clean separation of concerns
- Wide model support (Mistral, Llama, CodeLlama, etc.)
- Better resource management and concurrent request handling

**Alternative Considered**: llama-cpp-python for direct integration
- More complex model management
- Requires more manual optimization
- Better for embedded scenarios but more complex for development

## Implementation Details

- Ollama service runs independently on localhost:11434
- LLM service communicates via HTTP API
- Model selection configurable (mistral:7b default for balance of speed/quality)
- Function calling support for tool integration
- Context management for conversation memory

---

## Decision

2025-08-14 22:24:06 - **Tool Framework: Plugin-Based Architecture with Automatic Discovery**

## Rationale

Designed extensible tool framework to support easy addition of new capabilities:

**Key Design Principles**:
- Plugin-based: Tools can be added without modifying core code
- Automatic Discovery: Tools register themselves via decorators or class inheritance
- Schema Generation: Tool parameters automatically documented for LLM function calling
- Sandboxed Execution: Tools run with proper error handling and isolation
- Type Safety: Pydantic models for parameter validation

## Implementation Details

- Base `Tool` class with standard interface (`execute` method, parameter schemas)
- `@tool` decorator for simple function-based tools
- Tool registry for automatic discovery and registration
- Built-in tools: calculator, weather, file operations, system info
- JSON schema generation for LLM function calling integration
- Error handling prevents tool failures from crashing agent

---

## Decision

2025-08-14 22:24:06 - **Configuration Management: YAML-Based with Environment Override**

## Rationale

Implemented hierarchical configuration system for flexibility:

**Configuration Sources** (priority order):
1. Environment variables (highest priority)
2. Command-line arguments
3. Configuration files (YAML)
4. Default values (lowest priority)

**Benefits**:
- Easy deployment configuration changes
- Development vs. production environment support
- User customization without code changes
- Validation and type checking via Pydantic

## Implementation Details

- Primary config: `voice_agent/config/default.yaml`
- Pydantic models for configuration validation
- Environment variable support (e.g., `VOICE_AGENT_DEBUG=1`)
- Runtime configuration updates for testing/debugging
- Audio device configuration with automatic detection fallback

---

## Decision

2025-08-14 22:24:06 - **Testing Strategy: Multi-Level Testing Approach**

## Rationale

Implemented comprehensive testing strategy addressing different aspects:

**Testing Levels**:
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction testing
3. **System Tests**: End-to-end conversation flows
4. **Performance Tests**: Latency and throughput benchmarks
5. **Manual Testing**: Interactive testing tools

**Benefits**:
- Early detection of component failures
- Regression prevention during development
- Performance monitoring and optimization guidance
- User acceptance testing capabilities

## Implementation Details

- Integration testing guide with step-by-step component verification
- Automated test scripts for common scenarios
- Performance benchmarking tools for STT/TTS latency measurement
- Interactive testing modes (`--debug`, `--no-audio`)
- Component isolation testing for troubleshooting
2025-08-14 19:18:50 - **Audio Format Fix: Bark Float32 → PCM16 Conversion Before Playback**

Rationale:
Bark generated IEEE float32 WAV data (format code 3). Python's wave module and downstream playback/load path expected PCM (format code 1), producing log: "Error loading audio file: unknown format: 3". Converting to int16 ensures broad compatibility and eliminates runtime error.

Implementation Details:
- Updated [`_speak_bark()`](src/voice_agent/core/tts_service.py:344) to:
  - Clip float32 array to [-1.0, 1.0]
  - Scale to int16 (value * 32767) and cast
  - Write via scipy wavfile as PCM16
- Added explanatory comments and guarded conversion block
- Verified removal of the error after rerun

Implications:
- Playback reliability improved across environments lacking float WAV support
- Enables future post-processing (e.g., normalization) on consistent PCM16 data
- No change to synthesis latency (conversion negligible versus model generation)
