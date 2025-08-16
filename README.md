# Local Realtime Voice Agent

A fully local realtime voice agent in Python with sophisticated multi-agent capabilities and extensible tooling framework.

## Overview

This voice agent provides a comprehensive AI assistant experience with:

- **100% Local Processing**: No cloud dependencies, complete privacy
- **Real-time Voice Interaction**: Bidirectional speech communication with natural conversation flow
- **Multi-Agent Coordination**: Intelligent task delegation across specialized agents
- **Extensible Tool Framework**: Easy integration of new capabilities and services
- **Python-based Architecture**: Leveraging the rich Python ecosystem

## 🚀 Key Features

### Advanced Multi-Agent System

- **Intelligent Routing**: Automatically routes tasks to the most appropriate specialized agents
- **Workflow Orchestration**: Execute complex multi-step tasks with dependencies
- **Agent Collaboration**: Multiple agents work together on complex problems
- **Context Preservation**: Seamless context sharing across agent handoffs
- **Performance Optimization**: Parallel execution and load balancing

### Voice Interaction

- **Natural Conversations**: Real-time speech-to-text and text-to-speech
- **Voice Activity Detection**: Smart detection of speech vs. silence
- **Multiple TTS Engines**: Bark, Coqui TTS, eSpeak, and pyttsx3 support
- **Streaming Processing**: Low-latency audio processing

### Extensible Tools

- **Built-in Tools**: Calculator, weather, file operations, web search, news, calendar
- **Custom Tools**: Easy framework for adding new capabilities
- **Tool Chaining**: Combine multiple tools in complex workflows
- **Safe Execution**: Sandboxed tool execution environment

## Architecture

The voice agent consists of several core components working together:

### Core Services

- **Audio Manager**: Handles microphone input and speaker output with voice activity detection
- **Speech-to-Text Service**: Converts speech to text using Whisper or Vosk
- **Text-to-Speech Service**: Converts text to speech using multiple engine options
- **LLM Service**: Processes conversation using Ollama or Transformers
- **Tool Executor**: Manages and executes extensible tools safely
- **Conversation Manager**: Orchestrates the entire interaction flow

### Multi-Agent System

- **VoiceAgentOrchestrator**: Main coordination hub with fallback support
- **MultiAgentService**: Central service for agent coordination and routing
- **Specialized Agents**: Task-specific agents for optimal performance
- **Communication Hub**: Inter-agent messaging and collaboration
- **Workflow Engine**: Complex multi-step task coordination

### Specialized Agents

#### 🔍 InformationAgent

- **Purpose**: Research and information retrieval
- **Capabilities**: Weather queries, web search, news retrieval
- **Tools**: Weather API, web search, news feeds
- **Use Cases**: "What's the weather in Tokyo?", "Search for AI news"

#### 🧮 UtilityAgent

- **Purpose**: Mathematical calculations and utility functions
- **Capabilities**: Complex calculations, data analysis, formula solving
- **Tools**: Advanced calculator with mathematical functions
- **Use Cases**: "Calculate compound interest", "Solve this equation"

#### 📁 ProductivityAgent

- **Purpose**: File management and task organization
- **Capabilities**: File operations, calendar management, task coordination
- **Tools**: File system operations, calendar integration
- **Use Cases**: "Save this data to a file", "Schedule a meeting"

#### 💬 GeneralAgent

- **Purpose**: General conversation and coordination
- **Capabilities**: Natural conversation, general assistance, task routing
- **Tools**: Basic tool access for general tasks
- **Use Cases**: General questions, conversation, help requests

#### 🛠️ ToolSpecialistAgent

- **Purpose**: Advanced tool execution and system operations
- **Capabilities**: Complex tool chains, system information, specialized operations
- **Tools**: Full tool suite access with advanced capabilities
- **Use Cases**: Complex multi-tool workflows, system diagnostics

## 📦 Installation

### Prerequisites

- **Python 3.8 or higher**
- **Audio devices**: Microphone and speakers for voice interaction
- **System dependencies**:
  - Linux: `sudo apt-get install portaudio19-dev python3-dev`
  - macOS: `brew install portaudio`
  - Windows: Audio drivers typically included

### Option 1: Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd voice_agent

# Install with pip
pip install -e .

# Install system dependencies (if needed)
pip install -r requirements.txt
```

### Option 2: Development Setup

```bash
# Clone and setup development environment
git clone <repository-url>
cd voice_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks (optional)
pre-commit install
```

### Option 3: Using devenv (Nix)

```bash
# If you have Nix with flakes enabled
direnv allow  # or nix develop
```

## ⚙️ Configuration

### Quick Configuration

The voice agent uses YAML configuration files in [`src/voice_agent/config/`](src/voice_agent/config/):

- [`default.yaml`](src/voice_agent/config/default.yaml): Main configuration file
- [`models.yaml`](src/voice_agent/config/models.yaml): Model specifications and download locations

### Multi-Agent Configuration

Enable multi-agent features by updating [`default.yaml`](src/voice_agent/config/default.yaml):

```yaml
# Enable multi-agent system
multi_agent:
  enabled: true
  routing_strategy: "hybrid"
  max_concurrent_agents: 5

  # Enable advanced features
  workflow_orchestration:
    enabled: true
    max_concurrent_workflows: 3
    enable_parallel_execution: true

  inter_agent_communication:
    enabled: true
    collaboration_timeout: 120
    broadcast_enabled: true

  enhanced_delegation:
    enabled: true
    patterns:
      - "capability_based"
      - "collaborative"
      - "expertise_weighted"
      - "consensus"
```

### Core System Configuration

```yaml
# Audio settings
audio:
  sample_rate: 16000
  chunk_size: 1024
  vad_aggressiveness: 3

# Speech recognition
stt:
  model: "whisper-base"
  language: "auto"
  streaming: true

# Text-to-speech
tts:
  engine: "bark" # bark | coqui | espeak | pyttsx3
  voice: "default"
  speed: 1.0

# Language model
llm:
  provider: "ollama"
  model: "llama3.2:3b"
  temperature: 0.7

# Available tools
tools:
  enabled:
    - calculator
    - weather
    - file_ops
    - web_search
    - news
    - calendar
```

## 🚀 Quick Start

### 1. Basic Voice Interaction

```python
from voice_agent import VoiceAgent

# Create and start the voice agent
agent = VoiceAgent()
agent.start()  # Starts listening for voice input
```

### 2. Multi-Agent Mode

```python
from voice_agent import VoiceAgent
from voice_agent.core.config import Config

# Load configuration with multi-agent enabled
config = Config.load("src/voice_agent/config/default.yaml")
config.multi_agent.enabled = True

# Create agent with multi-agent capabilities
agent = VoiceAgent(config=config)
agent.start()
```

### 3. Command Line Usage

```bash
# Basic usage with voice
python -m voice_agent.main

# Enable multi-agent mode
python -m voice_agent.main --multi-agent

# Text-only mode (no audio)
python -m voice_agent.main --no-audio

# Debug mode with verbose logging
python -m voice_agent.main --debug

# Custom configuration
python -m voice_agent.main --config my_config.yaml

# TUI mode (text user interface)
python -m voice_agent.main --ui tui
```

### 4. Example Interactions

Once started, try these voice commands:

```
# Single-agent tasks
"What's the weather in London?"
"Calculate 25 times 7 plus 12"
"List files in the current directory"

# Multi-agent workflows (when enabled)
"Get weather for Tokyo and save it to a file"
"Research AI trends and create a summary report"
"Calculate mortgage payments and save the analysis"
```

## 🤖 Multi-Agent Capabilities

### Workflow Types

The multi-agent system supports sophisticated task coordination:

#### Sequential Workflows

Tasks execute one after another:

```
User: "Get weather for Paris, then calculate heating costs, then save analysis"
→ InformationAgent gets weather
→ UtilityAgent calculates costs
→ ProductivityAgent saves results
```

#### Parallel Workflows

Independent tasks run simultaneously:

```
User: "Get weather for London and New York simultaneously"
→ Two InformationAgents work in parallel
→ Results combined and presented
```

#### Pipeline Workflows

Output of one task feeds into the next:

```
User: "Search for stock prices, calculate returns, generate investment advice"
→ InformationAgent searches prices
→ UtilityAgent calculates returns
→ GeneralAgent provides advice
```

#### Collaborative Workflows

Multiple agents work together:

```
User: "Analyze this business plan from multiple perspectives"
→ InformationAgent researches market data
→ UtilityAgent performs financial analysis
→ ProductivityAgent structures the report
→ Results synthesized collaboratively
```

### Agent Specializations

Each agent type has specific strengths and use cases:

| Agent                   | Strengths                  | Example Tasks                         |
| ----------------------- | -------------------------- | ------------------------------------- |
| **InformationAgent**    | Research, data retrieval   | Weather queries, web search, news     |
| **UtilityAgent**        | Mathematics, calculations  | Complex math, data analysis, formulas |
| **ProductivityAgent**   | File ops, organization     | Save files, manage tasks, schedules   |
| **GeneralAgent**        | Conversation, coordination | General chat, help, task routing      |
| **ToolSpecialistAgent** | Advanced tool usage        | Complex tool chains, system ops       |

### Delegation Patterns

The system uses intelligent delegation patterns:

- **Capability-Based**: Routes to agents with required capabilities
- **Load-Balanced**: Distributes work evenly across agents
- **Expertise-Weighted**: Prioritizes agents with highest expertise
- **Collaborative**: Engages multiple agents for complex tasks
- **Hierarchical**: Respects agent priority levels
- **Consensus**: Requires agreement from multiple agents

## 🛠️ Built-in Tools

The voice agent includes a comprehensive set of built-in tools:

### 🧮 Calculator

Advanced mathematical calculations with safety restrictions:

```
# Basic arithmetic
"What is 25 times 7 plus 12?" → 187
"Calculate the square root of 144" → 12.0

# Complex expressions
"What's (15 + 25) * 2.5 / 4?" → 25.0
"Calculate 2 to the power of 8" → 256

# Mathematical functions
"Find the absolute value of -42" → 42
"What's the maximum of 1, 5, 3, 9, 2?" → 9
```

### 🌤️ Weather

Real-time weather information retrieval:

```
# Current weather
"What's the weather like in London?"
→ "London: 18°C, partly cloudy with light winds from the west"

# Different units
"Get weather for New York in Fahrenheit"
→ "New York: 72°F, sunny with low humidity"

# Multiple locations
"Compare weather in Tokyo and Paris"
→ Weather data for both cities with comparison
```

### 📁 File Operations

Safe file system operations with security restrictions:

```
# Directory operations
"List files in the current directory"
→ Shows files and folders with types and sizes

# File reading (safe files only)
"Read the contents of README.md"
→ Displays file contents with safety checks

# File existence checks
"Does setup.py exist in this directory?"
→ Confirms file existence and provides details
```

### 🔍 Web Search

Internet search capabilities for information retrieval:

```
# General search
"Search for Python programming tutorials"
→ Returns relevant search results with descriptions

# News and current events
"Find recent news about artificial intelligence"
→ Latest AI news articles and developments

# Research queries
"Look up information about renewable energy trends"
→ Comprehensive search results on the topic
```

### 📰 News

Current news and headlines:

```
"Get the latest news headlines"
"Find news about technology"
"What's happening in the world today?"
```

### 📅 Calendar

Calendar and scheduling operations:

```
"What's on my calendar today?"
"Schedule a meeting for tomorrow at 2 PM"
"Check my availability next week"
```

### Tool Integration in Multi-Agent Workflows

Tools work seamlessly across agents in complex workflows:

```
# Multi-step tool usage
"Get weather for San Francisco, calculate energy costs, and save the analysis"

Workflow:
1. InformationAgent uses Weather tool → Gets SF weather data
2. UtilityAgent uses Calculator tool → Calculates energy costs
3. ProductivityAgent uses File Operations → Saves analysis report
4. Results combined and presented to user
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
