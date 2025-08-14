# System Patterns

This file documents recurring patterns and standards used in the project.
It is optional, but recommended to be updated as the project evolves.

## Coding Patterns

2025-08-14 18:20:37 - **Error Handling Pattern for External Dependencies**
- All external library imports wrapped in try/except blocks with fallback None values
- Runtime availability checks before using optional dependencies
- Graceful degradation when dependencies are unavailable
- Example: BARK_AVAILABLE, COQUI_AVAILABLE, ESPEAK_AVAILABLE flags

2025-08-14 18:20:37 - **Async Service Initialization Pattern**
- Services use separate initialization phase with `async def initialize()`
- Heavy operations (model loading) performed in thread executors to avoid blocking
- Initialization state tracking with boolean flags
- Clean separation between constructor and resource-intensive setup

2025-08-14 18:20:37 - **Compatibility Monkey Patching Pattern**
- Temporary function replacement for third-party library compatibility
- Always use try/finally to ensure original function is restored
- Minimal scope - only patch during specific operations
- Document the reason and alternative approaches considered

## Architectural Patterns

2025-08-14 18:20:37 - **Multi-Backend Service Pattern**
- Services support multiple implementations (TTS: bark/coqui/espeak/pyttsx3)
- Backend selection based on availability and configuration preferences
- Consistent interface regardless of underlying implementation
- Fallback chains for graceful degradation

2025-08-14 18:20:37 - **Configuration-Driven Component Selection**
- Services determine behavior based on configuration objects
- Support for "auto" mode that selects best available option
- Clear separation between configuration and implementation logic

## Testing Patterns

2025-08-14 18:20:37 - **Integration Testing with devenv**
- Use `devenv shell -- command` for testing in reproducible environment
- Test with --debug and --no-audio flags for development/CI scenarios
- Verify full pipeline functionality rather than just unit tests
- Monitor logs to confirm each component initializes and functions correctly

2025-08-14 22:25:43 - **Service Architecture Pattern**
- Each major component (STT, TTS, LLM, Audio) implemented as independent service classes
- Base service interface with `initialize()`, `cleanup()`, and primary function methods
- Example: `STTService`, `TTSService`, `LLMService`, `AudioManager`
- Benefits: Modularity, testability, easy component replacement

2025-08-14 22:25:43 - **Tool Registration Pattern**
- Decorator-based tool registration with automatic schema generation
- Example: `@tool(name="calculator", description="Perform math operations")`
- Benefits: Easy extensibility, automatic documentation, type safety
- Plugin architecture with automatic discovery and registration

2025-08-14 22:25:43 - **Configuration Hierarchical Override Pattern**
- Pydantic models for configuration with hierarchical override system
- Priority: Environment vars → CLI args → YAML files → defaults
- Benefits: Type safety, validation, flexible deployment configuration
- Example: `Config`, `AudioConfig`, `STTConfig`, `TTSConfig`, `LLMConfig`

## Performance Optimization Patterns

2025-08-14 22:25:43 - **Model Preloading Strategy**
- Models loaded at startup when possible, on-demand as backup
- Thread executor usage for non-blocking model initialization
- Benefits: Fast response times, graceful handling of resource constraints

2025-08-14 22:25:43 - **Pipeline Architecture Pattern**
- Linear processing: Audio Input → VAD → STT → LLM → Tool Execution → TTS → Audio Output
- Each stage as independent service with clear interfaces
- Async coordination between pipeline stages
- Benefits: Clear data flow, easy debugging, component isolation

## Configuration Management Patterns

2025-08-14 22:25:43 - **Environment-Specific Configuration**
- Development vs. Production configuration support
- Auto-detection with manual override capability
- Device configuration: `null` values trigger auto-detection, specific indices for manual

2025-08-14 22:25:43 - **Performance Tuning Configuration Patterns**
- Speed-optimized: base STT models, smaller LLM models, eSpeak TTS
- Quality-optimized: large STT models, bigger LLM models, Bark TTS
- Balanced default: small STT, mistral:7b LLM, auto TTS selection

## Deployment Patterns

2025-08-14 22:25:43 - **Environment Reproducibility Pattern**
- devenv/nix for identical development environments across systems
- `devenv.nix` with all system and Python dependencies declared
- Commands: `devenv shell`, `devenv test`, `devenv ci`
- Benefits: Eliminates "works on my machine" problems

2025-08-14 22:25:43 - **Standalone Deployment Pattern**
- Direct Python execution with local dependencies
- Setup: `devenv shell` or virtual environment activation
- Usage: `python main.py` with various flags (--debug, --no-audio)
- Benefits: Simple development, no containerization overhead

2025-08-14 22:25:43 - **Resource Management Pattern**
- Automatic hardware detection (CPU/GPU for STT, audio devices)
- Graceful degradation when resources unavailable
- Memory management with model size configuration options

## Advanced Testing Patterns

2025-08-14 22:25:43 - **Component Isolation Testing**
- Test each service independently before integration
- Mock dependencies, test service interfaces in isolation
- Example: Test STT with generated audio, test TTS with sample text
- Progressive integration: components → services → full pipeline

2025-08-14 22:25:43 - **Performance Benchmarking Pattern**
- Timing-based performance measurement for optimization
- Metrics: STT latency, TTS generation time, LLM response time, end-to-end latency
- Command-line benchmarking tools with `time` measurements
- Resource monitoring during testing

2025-08-14 22:25:43 - **Error Simulation Testing**
- Deliberately trigger error conditions to test recovery mechanisms
- Scenarios: Missing models, audio device unavailable, network issues
- Configuration overrides to force error conditions
- Validates error handling and graceful degradation

2025-08-14 22:25:43 - **Manual Testing Pattern**
- Interactive testing modes for user acceptance
- Modes: `--debug` (verbose logging), `--no-audio` (text-only), demos
- Real-world validation with `tool_demo.py`, `basic_chat.py`
- User experience testing and acceptance validation

## Security and Monitoring Patterns

2025-08-14 22:25:43 - **Sandboxed Tool Execution Pattern**
- Pydantic validation for tool parameters
- Error catching and resource limits for tool execution
- Input validation and system protection
- Benefits: Prevent malicious input, system stability

2025-08-14 22:25:43 - **Structured Logging Pattern**
- Configurable verbosity: ERROR (production), INFO (normal), DEBUG (development)
- Contextual information in log messages
- Environment variable control: `VOICE_AGENT_DEBUG=1`
- Python logging module with proper formatting

2025-08-14 22:25:43 - **Systematic Troubleshooting Pattern**
- Component isolation for problem diagnosis
- Process: Audio devices → Ollama → STT models → TTS engines → Tools
- Benefits: Systematic problem identification, efficient debugging
- Documented troubleshooting procedures in integration testing guide

2025-08-14 22:25:43 - Enhanced with comprehensive patterns from existing documentation and implementation experience.