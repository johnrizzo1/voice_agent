# Local Realtime Voice Agent

A fully local realtime voice agent in Python with extensible tooling capabilities.

## Overview

This voice agent provides:
- **100% Local Processing**: No cloud dependencies
- **Real-time Voice Interaction**: Bidirectional speech communication
- **Extensible Tool Framework**: Easy integration of new capabilities
- **Python-based**: Leveraging Python ecosystem

## Architecture

The voice agent consists of several core components:

- **Audio Manager**: Handles microphone input and speaker output with voice activity detection
- **Speech-to-Text Service**: Converts speech to text using Whisper or Vosk
- **Text-to-Speech Service**: Converts text to speech using Coqui TTS or pyttsx3
- **LLM Service**: Processes conversation using Ollama or Transformers
- **Tool Executor**: Manages and executes extensible tools
- **Conversation Manager**: Orchestrates the entire interaction flow

## Installation

### Prerequisites

- Python 3.8 or higher
- Audio input/output devices (microphone and speakers)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install the Package

```bash
pip install -e .
```

## Configuration

The voice agent uses YAML configuration files located in `voice_agent/config/`:

- `default.yaml`: Main configuration file
- `models.yaml`: Model specifications and download locations

### Key Configuration Options

```yaml
audio:
  sample_rate: 16000
  chunk_size: 1024

stt:
  model: "whisper-base"
  language: "auto"
  streaming: true

tts:
  engine: "coqui"
  voice: "default"
  speed: 1.0

llm:
  provider: "ollama"
  model: "mistral:7b"
  temperature: 0.7

tools:
  enabled:
    - calculator
    - weather
    - file_ops
```

## Quick Start

### Basic Usage

```python
from voice_agent import VoiceAgent

# Create and start the voice agent
agent = VoiceAgent()
agent.start()  # Starts listening for voice input
```

### Command Line

```bash
# Start with default configuration
python main.py

# Start with custom configuration
python main.py --config custom_config.yaml

# Enable debug mode
python main.py --debug

# Text-only mode (no audio)
python main.py --no-audio
```

## Built-in Tools

The voice agent comes with several built-in tools:

### Calculator
Performs mathematical calculations safely.
```
"What is 25 times 7 plus 12?"
"Calculate the square root of 144"
```

### Weather
Gets weather information (mock implementation).
```
"What's the weather like in London?"
"Get weather for New York in fahrenheit"
```

### File Operations
Basic file system operations with safety restrictions.
```
"List files in the current directory"
"Read the contents of README.md"
```

### Web Search
Internet search capabilities (mock implementation).
```
"Search for Python programming tutorials"
"Find news about artificial intelligence"
```

## Creating Custom Tools

### Method 1: Tool Class

```python
from voice_agent.tools.base import Tool
from pydantic import BaseModel, Field

class MyToolParameters(BaseModel):
    query: str = Field(description="Query parameter")

class MyTool(Tool):
    name = "my_tool"
    description = "My custom tool"
    Parameters = MyToolParameters
    
    def execute(self, query: str) -> dict:
        return {
            "success": True,
            "result": f"Processed: {query}",
            "error": None
        }
```

### Method 2: Function Decorator

```python
from voice_agent.tools.registry import tool

@tool(name="timestamp", description="Get current timestamp")
def get_timestamp(format_type: str = "iso") -> dict:
    from datetime import datetime
    return {"timestamp": datetime.now().isoformat()}
```

## Examples

The `examples/` directory contains several demonstration scripts:

- `basic_chat.py`: Simple voice chat session
- `tool_demo.py`: Demonstrates built-in tools
- `custom_tool.py`: Shows how to create custom tools

Run examples:
```bash
python voice_agent/examples/basic_chat.py
python voice_agent/examples/tool_demo.py
python voice_agent/examples/custom_tool.py
```

## Development

### Project Structure

```
voice_agent/
├── core/                 # Core components
│   ├── audio_manager.py
│   ├── stt_service.py
│   ├── tts_service.py
│   ├── llm_service.py
│   ├── conversation.py
│   ├── tool_executor.py
│   └── config.py
├── tools/                # Tool framework
│   ├── base.py
│   ├── registry.py
│   └── builtin/          # Built-in tools
├── models/               # Model management
│   └── model_manager.py
├── config/               # Configuration files
│   ├── default.yaml
│   └── models.yaml
├── examples/             # Example scripts
└── tests/                # Test suite
```

### Running Tests

```bash
python -m pytest voice_agent/tests/
```

### Code Quality

```bash
# Run linting
flake8 voice_agent/

# Run type checking
mypy voice_agent/

# Format code
black voice_agent/
```

## Dependencies

### Core Dependencies
- `pyaudio>=0.2.11` - Audio I/O
- `webrtcvad>=2.0.10` - Voice activity detection
- `numpy>=1.21.0` - Numerical computing
- `faster-whisper>=0.10.0` - Speech recognition
- `TTS>=0.22.0` - Text-to-speech
- `ollama>=0.1.0` - LLM integration
- `pydantic>=2.0.0` - Data validation
- `click>=8.0.0` - CLI interface
- `pyyaml>=6.0` - Configuration parsing

### Optional Dependencies
- `vosk>=0.3.45` - Alternative STT engine
- `pyttsx3>=2.90` - Fallback TTS engine
- `transformers>=4.30.0` - Alternative LLM backend
- `torch>=2.0.0` - Deep learning framework

## Troubleshooting

### Audio Issues
- Ensure microphone and speakers are properly connected
- Check audio device permissions
- Try different sample rates in configuration

### Model Loading
- Models are automatically downloaded on first use
- Check internet connection for model downloads
- Verify sufficient disk space for models

### Performance
- Use GPU acceleration when available
- Adjust model sizes based on hardware capabilities
- Consider streaming options for real-time processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Coqui TTS for text-to-speech
- Ollama for local LLM serving
- All the open-source contributors who made this possible

## Roadmap

- [ ] Enhanced tool sandboxing
- [ ] Plugin marketplace
- [ ] Multi-language support
- [ ] Voice cloning capabilities
- [ ] Mobile app integration
- [ ] Docker deployment
- [ ] Cloud-optional hybrid mode