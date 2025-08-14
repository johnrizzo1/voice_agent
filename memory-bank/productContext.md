# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.

## Project Goal

A fully local realtime voice agent in Python with extensible tooling capabilities. Core requirements include:
- **100% Local Processing**: No cloud dependencies
- **Real-time Voice Interaction**: Bidirectional speech communication with interrupt handling
- **Extensible Tool Framework**: Easy integration of new capabilities with plugin architecture
- **Python-based**: Leveraging Python ecosystem for maximum compatibility

## Key Features

Based on the comprehensive implementation plan, the voice agent provides:

### Core Audio Pipeline
- **Audio Input Manager**: Microphone capture using PyAudio with Voice Activity Detection (webrtcvad)
- **Audio Output Manager**: Speaker output with real-time buffering and streaming support
- **Audio Preprocessing**: Real-time audio chunking, buffering, and noise handling

### Speech Processing Services
- **Speech-to-Text Service**:
  - Primary: faster-whisper (optimized Whisper) with streaming support
  - Alternative: Vosk (lightweight, good for streaming)
  - Model management with download, cache, and switching capabilities
- **Text-to-Speech Service**: Multi-backend architecture with automatic fallback
  - Primary: Bark TTS (neural, high quality voices) - **NOW WORKING**
  - Secondary: Coqui TTS (neural, high quality)
  - Tertiary: eSpeak-NG (lightweight, fast)
  - Fallback: pyttsx3 (cross-platform, reliable)

### LLM Integration
- **LLM Service**:
  - Primary: Ollama integration with local model support
  - Alternative: llama-cpp-python for direct model access
  - Model options: Mistral, Llama, CodeLlama
  - Function calling support for tool integration
  - Context management and conversation memory

### Conversation Management
- **Conversation Manager**:
  - Turn-taking logic with interrupt handling
  - Context preservation across conversations
  - Conversation history and memory management
  - Real-time processing coordination

### Tool Framework
- **Tool Executor**: Plugin-based architecture with sandboxed execution
- **Built-in Tools**: Calculator, weather, file operations, system info
- **Tool Registry**: Automatic discovery and registration system
- **Schema Generation**: Automatic tool documentation and validation

## Overall Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Audio Input    │───▶│  Speech-to-Text  │───▶│  Conversation   │
│  Manager        │    │  Service         │    │  Manager        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Audio Output   │◀───│  Text-to-Speech  │◀───│  LLM Service    │
│  Manager        │    │  Service         │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Tool Executor   │◀───│                 │
                       │                  │    │                 │
                       └──────────────────┘    └─────────────────┘
```

### System Requirements
- **Hardware**: Multi-core CPU (4+ recommended), 8GB RAM minimum (16GB recommended), 10GB storage
- **Audio**: Working microphone and speakers/headphones
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+ with WSL2

### Technology Stack
- **Audio**: PyAudio, webrtcvad, numpy
- **STT**: faster-whisper, vosk
- **TTS**: Bark TTS, Coqui TTS, eSpeak-NG, pyttsx3
- **LLM**: Ollama, transformers, torch
- **Framework**: asyncio, pydantic, click, pyyaml
- **Environment**: devenv/nix for reproducible builds

### Project Structure
```
voice_agent/
├── core/                    # Core services and managers
├── tools/                   # Tool framework and built-in tools
├── models/                  # Model management
├── config/                  # Configuration files
├── examples/                # Usage examples and demos
├── tests/                   # Test suites
└── docs/                    # Comprehensive documentation
```

## Development Environment

The project uses **devenv/nix** for reproducible development environments, providing:
- Fully reproducible builds across different systems
- Integrated system dependencies (portaudio, espeak-ng, CUDA)
- Development tools integration (linters, formatters, testing)
- Automated dependency management

## Current Status

**FULLY OPERATIONAL** - All major components implemented and working:
- ✅ Complete voice interaction pipeline verified
- ✅ Multi-backend TTS system with Bark TTS working
- ✅ PyTorch 2.6 compatibility issues resolved
- ✅ Real-time audio processing functional
- ✅ Tool framework with extensible plugin system
- ✅ Configuration management system
- ✅ Comprehensive testing and setup documentation

2025-08-14 18:19:27 - Initial Memory Bank creation documenting voice agent project structure and goals
2025-08-14 22:22:29 - Enhanced with comprehensive project overview from existing documentation