# Multi-Agent Configuration Reference

## Overview

This comprehensive reference covers all configuration options for the Voice Agent multi-agent system. The configuration system uses YAML files located in [`src/voice_agent/config/`](../src/voice_agent/config/) and supports hierarchical configuration with environment-specific overrides.

## 📁 Configuration Files

### Primary Configuration Files

- **[`default.yaml`](../src/voice_agent/config/default.yaml)**: Main configuration file with all system settings
- **[`models.yaml`](../src/voice_agent/config/models.yaml)**: Model specifications and download URLs
- **[`templates.py`](../src/voice_agent/config/templates.py)**: Pre-configured templates for different use cases

### Configuration Loading

```python
from voice_agent.core.config import Config

# Load default configuration
config = Config.load("src/voice_agent/config/default.yaml")

# Load with custom overrides
config = Config.load("my_custom_config.yaml")

# Programmatic configuration
config.multi_agent.enabled = True
config.llm.model = "llama3.2:1b"
```

## 🎛️ Core System Configuration

### Audio Settings

```yaml
audio:
  # Device configuration
  input_device: null # Auto-detect microphone (or device ID)
  output_device: null # Auto-detect speakers (or device ID)

  # Audio processing
  sample_rate: 16000 # Audio sample rate (Hz)
  chunk_size: 1024 # Audio buffer size

  # Voice Activity Detection (VAD)
  vad_aggressiveness: 3 # 0-3, higher = more noise filtering
  min_speech_frames: 8 # Minimum frames to detect speech start
  max_silence_frames: 15 # Maximum frames of silence before stopping
  speech_detection_cooldown: 2.0 # Seconds after TTS before listening
```

**Audio Device Configuration:**

```bash
# List available audio devices
python -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f'Device {i}: {info[\"name\"]} - {info[\"maxInputChannels\"]} in, {info[\"maxOutputChannels\"]} out')
p.terminate()
"

# Use specific devices
audio:
  input_device: 1    # Use device ID 1 for microphone
  output_device: 2   # Use device ID 2 for speakers
```

### Speech-to-Text (STT) Configuration

```yaml
stt:
  # Model selection
  model: "whisper-base" # whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large
  language: "auto" # Language code or "auto" for detection
  streaming: true # Enable streaming recognition

  # Performance settings
  beam_size: 5 # Beam search size (higher = more accurate, slower)
  best_of: 5 # Number of candidates (higher = more accurate, slower)
  temperature: 0.0 # Sampling temperature (0.0 = deterministic)

  # Whisper-specific settings
  compute_type: "float16" # float16, float32, int8 (affects speed/memory)
  device: "auto" # auto, cpu, cuda
```

**Model Size Guide:**

- `whisper-tiny`: Fastest, least accurate (~39 MB)
- `whisper-base`: Good balance (~74 MB)
- `whisper-small`: Better accuracy (~244 MB)
- `whisper-medium`: High accuracy (~769 MB)
- `whisper-large`: Best accuracy (~1550 MB)

### Text-to-Speech (TTS) Configuration

```yaml
tts:
  # Engine selection
  engine: "bark" # bark, coqui, espeak, pyttsx3, auto
  voice: "default" # Voice selection (engine-specific)
  speed: 1.0 # Speech speed multiplier

  # Bark-specific settings
  bark_voice_preset: "v2/en_speaker_1" # Consistent voice personality

  # Performance tuning
  post_tts_cooldown: 0.3 # Pause after speech synthesis
  tts_cooldown_margin: 0.25 # Additional margin for timing
  enable_tts_buffer_double_clear: false # Extra buffer clearing
```

**TTS Engine Comparison:**

- **Bark**: High-quality neural TTS, slower but most natural
- **Coqui**: Good quality, moderate speed
- **eSpeak**: Fast, robotic voice, minimal resources
- **pyttsx3**: System TTS, varies by platform

### Language Model (LLM) Configuration

```yaml
llm:
  # Provider and model
  provider: "ollama" # ollama, transformers
  model: "llama3.2:3b" # Model identifier

  # Generation parameters
  temperature: 0.7 # Creativity level (0.0 = deterministic, 1.0 = creative)
  max_tokens: 2048 # Maximum response length
  top_p: 0.9 # Nucleus sampling parameter
  top_k: 40 # Top-k sampling parameter

  # Performance settings
  num_ctx: 4096 # Context window size
  num_predict: 512 # Prediction limit per request
  repeat_penalty: 1.1 # Repetition penalty

  # Ollama-specific settings
  keep_alive: "5m" # Keep model loaded in memory
  num_thread: null # Number of threads (null = auto)
```

**Model Selection Guide:**

- **llama3.2:1b**: Fastest, basic responses (~1.3 GB)
- **llama3.2:3b**: Good balance of speed/quality (~2.0 GB)
- **mistral:7b**: High quality, slower (~4.1 GB)
- **codellama:7b**: Code-focused (~3.8 GB)

### Tool Configuration

```yaml
tools:
  # Enabled tools
  enabled:
    - calculator # Mathematical calculations
    - weather # Weather information
    - file_ops # File operations
    - web_search # Internet search
    - news # News retrieval
    - calendar # Calendar operations

  # Disabled tools
  disabled: []

  # Tool-specific settings
  calculator:
    max_expression_length: 1000
    allowed_functions: ["abs", "round", "min", "max", "sum"]

  file_ops:
    allowed_extensions: [".txt", ".md", ".json", ".yaml", ".py"]
    max_file_size: "1MB"
    safe_directories: [".", "./data", "./output"]

  weather:
    default_units: "celsius"
    cache_duration: 300 # Cache weather data for 5 minutes

  web_search:
    max_results: 10
    timeout: 30
```

## 🤖 Multi-Agent System Configuration

### Core Multi-Agent Settings

```yaml
multi_agent:
  # Enable/disable multi-agent functionality
  enabled: true # Master switch for multi-agent features

  # Agent routing
  default_agent: "general_agent" # Fallback agent
  routing_strategy: "hybrid" # hybrid, rules_only, embeddings_only, llm_fallback
  confidence_threshold: 0.7 # Minimum confidence for routing
  max_concurrent_agents: 5 # Maximum simultaneous agents

  # Context management
  context_sharing_enabled: true # Enable context sharing between agents
  context_window_size: 4000 # Context window per agent
  embedding_model: "nomic-embed-text" # Model for semantic routing
```

### Workflow Orchestration

```yaml
workflow_orchestration:
  enabled: true # Enable workflow coordination
  max_concurrent_workflows: 3 # Maximum simultaneous workflows
  default_timeout: 300 # Default workflow timeout (seconds)
  enable_parallel_execution: true # Allow parallel task execution
  enable_pipeline_execution: true # Allow pipeline workflows
  task_dependency_timeout: 60 # Task dependency resolution timeout

  # Workflow types
  supported_types:
    - "sequential" # One after another
    - "parallel" # Simultaneous execution
    - "pipeline" # Output chaining
    - "conditional" # Condition-based execution
```

### Inter-Agent Communication

```yaml
inter_agent_communication:
  enabled: true # Enable agent messaging
  message_queue_size: 1000 # Message buffer size
  collaboration_timeout: 120 # Collaboration session timeout
  broadcast_enabled: true # Allow broadcast messages
  priority_messaging: true # Support message priorities

  # Communication patterns
  patterns:
    - "direct_messaging" # One-to-one communication
    - "broadcast" # One-to-many messaging
    - "request_response" # Structured queries
    - "collaboration_sessions" # Multi-agent sessions
    - "status_updates" # Progress notifications
```

### Enhanced Delegation

```yaml
enhanced_delegation:
  enabled: true # Enable advanced delegation
  consensus_threshold: 0.6 # Agreement threshold for consensus
  collaboration_min_agents: 2 # Minimum agents for collaboration

  # Available delegation patterns
  patterns:
    - "capability_based" # Route by required capabilities
    - "load_balanced" # Distribute load evenly
    - "expertise_weighted" # Prioritize by expertise
    - "collaborative" # Multi-agent collaboration
    - "hierarchical" # Priority-based routing
    - "consensus" # Agreement-based decisions

  # Expertise weights (0.0-1.0)
  expertise_weights:
    weather_info: 0.9
    calculations: 0.95
    file_operations: 0.85
    web_search: 0.8
    news_info: 0.85
    calendar_management: 0.8
    general_chat: 0.7
```

### Context Preservation

```yaml
context_preservation:
  enhanced_handoff: true # Comprehensive context transfer
  context_compression: true # Compress context for efficiency
  handoff_metadata: true # Include handoff metadata
  preserve_tool_states: true # Maintain tool configurations
  context_validation: true # Validate context integrity

  # Context management
  max_context_age: 3600 # Maximum context age (seconds)
  context_cleanup_interval: 300 # Cleanup interval (seconds)
  preserve_conversation_history: true # Keep conversation context
```

### Advanced Features

```yaml
advanced_features:
  status_reporting: true # Real-time progress updates
  result_aggregation: true # Combine multi-agent results
  collaborative_reasoning: true # Multi-agent problem solving
  workflow_monitoring: true # Monitor workflow execution
  performance_metrics: true # Collect performance data

  # Monitoring settings
  metrics_collection_interval: 60 # Metrics collection frequency
  performance_logging: true # Log performance data
  workflow_analytics: true # Analyze workflow patterns
```

## 🏗️ Agent Definitions

### Agent Configuration Structure

```yaml
multi_agent:
  agents:
    agent_name:
      type: "AgentClassName" # Agent implementation class
      capabilities: [] # List of agent capabilities
      tools: [] # Available tools
      system_prompt: "..." # Agent personality/instructions
      max_concurrent_tasks: 3 # Task concurrency limit
      timeout_seconds: 30.0 # Task timeout

      # Optional advanced settings
      expertise_level: 0.9 # Expertise rating (0.0-1.0)
      priority: 1 # Agent priority (1=highest)
      resource_weight: 1.0 # Resource usage weight
```

### GeneralAgent Configuration

```yaml
general_agent:
  type: "GeneralAgent"
  capabilities:
    - "general_chat" # Natural conversation
    - "tool_execution" # Basic tool usage
    - "conversation_memory" # Context retention
  tools:
    - "calculator"
    - "weather"
    - "file_ops"
  system_prompt: "You are a helpful general-purpose AI assistant with access to various tools."
  max_concurrent_tasks: 5
  timeout_seconds: 30.0

  # General agent specific settings
  conversation_style: "friendly" # friendly, professional, concise
  help_mode: true # Provide usage tips
  context_awareness: true # Understand conversation flow
```

### InformationAgent Configuration

```yaml
information_agent:
  type: "InformationAgent"
  capabilities:
    - "weather_info" # Weather data retrieval
    - "web_search" # Internet search
    - "news_info" # News and headlines
    - "tool_execution" # Tool usage
    - "conversation_memory" # Context retention
  tools:
    - "weather"
    - "web_search"
    - "news"
  system_prompt: "You are an information specialist focused on retrieving and presenting accurate, up-to-date information from various sources."
  max_concurrent_tasks: 4
  timeout_seconds: 45.0

  # Information agent specific settings
  research_depth: "comprehensive" # basic, standard, comprehensive
  source_attribution: true # Include information sources
  fact_checking: true # Verify information accuracy
  update_frequency: "realtime" # realtime, hourly, daily
```

### UtilityAgent Configuration

```yaml
utility_agent:
  type: "UtilityAgent"
  capabilities:
    - "calculations" # Mathematical operations
    - "tool_execution" # Tool usage
    - "conversation_memory" # Context retention
    - "system_info" # System information
  tools:
    - "calculator"
  system_prompt: "You are a UtilityAgent specialized in mathematical calculations and utility functions. Focus on precision, accuracy, and clear explanations."
  max_concurrent_tasks: 4
  timeout_seconds: 30.0

  # Utility agent specific settings
  calculation_precision: 10 # Decimal places
  show_work: true # Show calculation steps
  verify_results: true # Double-check calculations
  unit_conversion: true # Support unit conversions
```

### ProductivityAgent Configuration

```yaml
productivity_agent:
  type: "ProductivityAgent"
  capabilities:
    - "file_operations" # File system operations
    - "calendar_management" # Calendar operations
    - "task_organization" # Task management
    - "workflow_coordination" # Workflow management
    - "tool_execution" # Tool usage
    - "conversation_memory" # Context retention
  tools:
    - "file_ops"
    - "calendar"
  system_prompt: "You are a ProductivityAgent specialized in file management, calendar operations, and task organization."
  max_concurrent_tasks: 3
  timeout_seconds: 45.0

  # Productivity agent specific settings
  auto_backup: true # Automatic file backups
  organize_by_date: true # Date-based organization
  task_prioritization: true # Priority-based task handling
  deadline_tracking: true # Track deadlines and reminders
```

### ToolSpecialistAgent Configuration

```yaml
tool_specialist:
  type: "ToolSpecialistAgent"
  capabilities:
    - "tool_execution" # Advanced tool usage
    - "file_operations" # File operations
    - "calculations" # Mathematical operations
    - "system_info" # System information
  tools:
    - "calculator"
    - "file_ops"
    - "weather"
    - "web_search"
  system_prompt: "You are a specialist in tool execution and file operations. Be precise and thorough."
  max_concurrent_tasks: 3
  timeout_seconds: 60.0

  # Tool specialist specific settings
  tool_chaining: true # Chain multiple tools
  error_recovery: true # Automatic error recovery
  tool_optimization: true # Optimize tool usage
  advanced_workflows: true # Support complex workflows
```

## 🎯 Routing Rules Configuration

### Routing Rule Structure

```yaml
multi_agent:
  routing_rules:
    - name: "rule_name" # Unique rule identifier
      target_agent: "agent_name" # Target agent for matching queries
      patterns: [] # Text patterns to match
      capabilities: [] # Required capabilities
      priority: 1 # Rule priority (1=highest)
      confidence: 0.9 # Confidence threshold

      # Optional conditions
      conditions:
        min_words: 3 # Minimum word count
        max_words: 100 # Maximum word count
        requires_tools: [] # Required tools
        excludes_patterns: [] # Patterns to exclude
```

### Example Routing Rules

```yaml
routing_rules:
  # Weather-related queries
  - name: "weather_requests"
    target_agent: "information_agent"
    patterns:
      - "weather"
      - "forecast"
      - "temperature"
      - "rain"
      - "snow"
      - "storm"
      - "sunny"
      - "cloudy"
      - "climate"
    capabilities:
      - "weather_info"
    priority: 1
    confidence: 0.9

  # Mathematical calculations
  - name: "calculation_requests"
    target_agent: "utility_agent"
    patterns:
      - "calculate"
      - "compute"
      - "math"
      - "equation"
      - "sum"
      - "multiply"
      - "divide"
      - "percentage"
    capabilities:
      - "calculations"
    priority: 1
    confidence: 0.9

  # File operations
  - name: "file_operations"
    target_agent: "productivity_agent"
    patterns:
      - "file"
      - "save"
      - "load"
      - "directory"
      - "folder"
      - "create"
      - "delete"
      - "copy"
    capabilities:
      - "file_operations"
    priority: 1
    confidence: 0.85

  # Web search queries
  - name: "search_requests"
    target_agent: "information_agent"
    patterns:
      - "search"
      - "find"
      - "look up"
      - "google"
      - "research"
      - "information about"
    capabilities:
      - "web_search"
    priority: 1
    confidence: 0.8
```

## 🔧 Environment-Specific Configuration

### Development Configuration

```yaml
# development.yaml
environment: "development"

# Debug settings
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/voice_agent_dev.log"

# Faster models for development
llm:
  model: "llama3.2:1b" # Smaller, faster model

tts:
  engine: "espeak" # Fast TTS for development

# Reduced limits for testing
multi_agent:
  max_concurrent_agents: 2
  workflow_orchestration:
    max_concurrent_workflows: 1
    default_timeout: 60 # Shorter timeouts for dev
```

### Production Configuration

```yaml
# production.yaml
environment: "production"

# Production logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
  file: "logs/voice_agent_prod.log"
  rotation: "daily"
  max_files: 30

# High-quality models
llm:
  model: "llama3.2:3b" # Balanced model for production

tts:
  engine: "bark" # High-quality TTS

# Production limits
multi_agent:
  max_concurrent_agents: 10
  workflow_orchestration:
    max_concurrent_workflows: 5
    default_timeout: 300

# Performance monitoring
advanced_features:
  performance_metrics: true
  workflow_analytics: true
  metrics_export: "prometheus" # Export metrics
```

### Testing Configuration

```yaml
# testing.yaml
environment: "testing"

# Test-specific settings
ui:
  force_text_only: true # No audio during tests

# Mock implementations
tools:
  enabled:
    - "calculator" # Only essential tools for testing

# Minimal agent set
multi_agent:
  agents:
    general_agent:
      # Minimal configuration
      timeout_seconds: 10.0
    utility_agent:
      timeout_seconds: 5.0
```

## 📊 Performance Tuning

### Memory Optimization

```yaml
# Memory-conscious settings
llm:
  model: "llama3.2:1b" # Smaller model
  num_ctx: 2048 # Smaller context window
  keep_alive: "1m" # Shorter model retention

multi_agent:
  max_concurrent_agents: 3 # Fewer simultaneous agents
  context_window_size: 2000 # Smaller context windows

# Context management
context_preservation:
  context_compression: true # Compress contexts
  max_context_age: 1800 # Shorter context lifetime
```

### Speed Optimization

```yaml
# Speed-focused settings
audio:
  chunk_size: 512 # Smaller audio chunks
  vad_aggressiveness: 2 # Less aggressive VAD

stt:
  model: "whisper-tiny" # Fastest STT model
  compute_type: "int8" # Faster inference

tts:
  engine: "espeak" # Fastest TTS

llm:
  temperature: 0.0 # Deterministic generation
  max_tokens: 512 # Shorter responses
```

### Quality Optimization

```yaml
# Quality-focused settings
stt:
  model: "whisper-large" # Most accurate STT
  beam_size: 10 # Better accuracy
  compute_type: "float32" # Higher precision

tts:
  engine: "bark" # Highest quality TTS
  bark_voice_preset: "v2/en_speaker_6" # Consistent voice

llm:
  model: "mistral:7b" # High-quality model
  temperature: 0.7 # Balanced creativity
  top_p: 0.9 # Good diversity
```

## 🚨 Troubleshooting Configuration

### Common Configuration Issues

**Issue: Multi-agent not working**

```yaml
# Ensure multi-agent is enabled
multi_agent:
  enabled: true             # Must be true

# Check agent definitions exist
multi_agent:
  agents:
    general_agent: {...}    # Must have agent definitions
```

**Issue: Audio problems**

```yaml
# Test different audio settings
audio:
  sample_rate: 44100 # Try different rates
  input_device: 1 # Specify device ID
  vad_aggressiveness: 1 # Reduce noise filtering
```

**Issue: Slow performance**

```yaml
# Optimize for speed
llm:
  model: "llama3.2:1b" # Use smaller model

multi_agent:
  max_concurrent_agents: 2 # Reduce concurrency
```

### Configuration Validation

```python
# Validate configuration
from voice_agent.core.config import Config

try:
    config = Config.load("path/to/config.yaml")
    print("✅ Configuration loaded successfully")

    # Check critical settings
    if not config.multi_agent.enabled:
        print("⚠️ Multi-agent is disabled")

    if len(config.multi_agent.agents) == 0:
        print("❌ No agents configured")

except Exception as e:
    print(f"❌ Configuration error: {e}")
```

### Environment Variables

```bash
# Override configuration with environment variables
export VOICE_AGENT_DEBUG=true
export VOICE_AGENT_LLM_MODEL="llama3.2:1b"
export VOICE_AGENT_MULTI_AGENT_ENABLED=true
export VOICE_AGENT_TTS_ENGINE="espeak"

# Run with environment overrides
python -m voice_agent.main
```

## 📝 Configuration Templates

### Minimal Configuration

```yaml
# Minimal working configuration
multi_agent:
  enabled: true
  agents:
    general_agent:
      type: "GeneralAgent"
      capabilities: ["general_chat", "tool_execution"]
      tools: ["calculator"]
      system_prompt: "You are a helpful assistant."

tools:
  enabled: ["calculator"]

llm:
  provider: "ollama"
  model: "llama3.2:1b"
```

### Complete Feature Configuration

```yaml
# Full-featured configuration with all options
multi_agent:
  enabled: true
  default_agent: "general_agent"
  routing_strategy: "hybrid"
  confidence_threshold: 0.7
  max_concurrent_agents: 5

  workflow_orchestration:
    enabled: true
    max_concurrent_workflows: 3
    enable_parallel_execution: true

  inter_agent_communication:
    enabled: true
    collaboration_timeout: 120

  enhanced_delegation:
    enabled: true
    patterns: ["capability_based", "collaborative", "expertise_weighted"]

  context_preservation:
    enhanced_handoff: true
    context_compression: true

  advanced_features:
    status_reporting: true
    result_aggregation: true
    collaborative_reasoning: true
# Complete agent definitions...
# Complete routing rules...
# Complete tool configuration...
```

This configuration reference provides comprehensive coverage of all multi-agent system options, from basic setup to advanced optimization. Use it as a guide to customize your Voice Agent system for your specific needs and requirements.
