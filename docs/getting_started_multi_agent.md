# Getting Started with Multi-Agent Voice System

## Overview

This guide will walk you through setting up and using the multi-agent capabilities of the Voice Agent system. The multi-agent system enables sophisticated task coordination, intelligent delegation, and collaborative workflows that go far beyond single-agent capabilities.

## 🚀 Quick Start (5 minutes)

### Step 1: Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd voice_agent

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### Step 2: Enable Multi-Agent Features

Edit [`src/voice_agent/config/default.yaml`](../src/voice_agent/config/default.yaml):

```yaml
# Enable multi-agent system
multi_agent:
  enabled: true # Change from false to true
```

### Step 3: Test Multi-Agent Setup

```bash
# Start the voice agent with multi-agent enabled
python -m voice_agent.main --multi-agent --debug

# Or test in text mode first
python -m voice_agent.main --multi-agent --no-audio
```

### Step 4: Try Your First Multi-Agent Command

Once running, try this voice command:

```
"Get weather for London and save it to weather_report.txt"
```

**Expected Result**:

- InformationAgent retrieves weather data
- ProductivityAgent saves data to file
- You receive confirmation of both actions

🎉 **Congratulations!** You've successfully set up multi-agent capabilities.

## 📋 Complete Setup Guide

### Prerequisites

#### System Requirements

- **Python 3.8+** (3.9+ recommended)
- **4GB+ RAM** (8GB+ for optimal performance)
- **Audio devices**: Microphone and speakers/headphones
- **Internet connection**: For model downloads and web search

#### System Dependencies

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev build-essential
```

**macOS:**

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install portaudio
```

**Windows:**

```bash
# Audio drivers usually included
# If issues, install Visual Studio Build Tools
```

### Installation Options

#### Option 1: Standard Installation

```bash
git clone <repository-url>
cd voice_agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

#### Option 2: Development Installation

```bash
git clone <repository-url>
cd voice_agent

# Setup development environment
python -m venv venv
source venv/bin/activate

# Install with development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Setup pre-commit hooks (optional)
pre-commit install
```

#### Option 3: Nix/devenv (if available)

```bash
cd voice_agent
direnv allow  # or: nix develop
```

### Configuration Setup

#### 1. Basic Multi-Agent Configuration

Create or edit [`src/voice_agent/config/default.yaml`](../src/voice_agent/config/default.yaml):

```yaml
# Core multi-agent settings
multi_agent:
  enabled: true
  default_agent: "general_agent"
  routing_strategy: "hybrid"
  confidence_threshold: 0.7
  max_concurrent_agents: 5

  # Essential features
  workflow_orchestration:
    enabled: true
    max_concurrent_workflows: 3
    enable_parallel_execution: true

  inter_agent_communication:
    enabled: true
    collaboration_timeout: 120

  enhanced_delegation:
    enabled: true
    patterns:
      - "capability_based"
      - "collaborative"
      - "expertise_weighted"
```

#### 2. Audio Configuration

```yaml
audio:
  sample_rate: 16000
  chunk_size: 1024
  vad_aggressiveness: 3 # 0-3, higher = more noise filtering
  min_speech_frames: 8
  max_silence_frames: 15
```

#### 3. Model Configuration

```yaml
# Language model
llm:
  provider: "ollama"
  model: "llama3.2:3b" # Good balance of speed/quality
  temperature: 0.7

# Text-to-speech
tts:
  engine: "bark" # High quality neural TTS
  voice: "default"
  speed: 1.0

# Speech-to-text
stt:
  model: "whisper-base" # Good accuracy/speed balance
  language: "auto"
  streaming: true
```

#### 4. Tool Configuration

```yaml
tools:
  enabled:
    - calculator # Mathematical operations
    - weather # Weather information
    - file_ops # File operations
    - web_search # Internet search
    - news # News retrieval
    - calendar # Calendar operations
```

### Agent Setup and Verification

#### Verify Agent Configuration

Check that all agents are properly configured:

```bash
# Test configuration loading
python -c "
from voice_agent.core.config import Config
config = Config.load('src/voice_agent/config/default.yaml')
print('Multi-agent enabled:', config.multi_agent.enabled)
print('Available agents:', list(config.multi_agent.agents.keys()))
"
```

Expected output:

```
Multi-agent enabled: True
Available agents: ['general_agent', 'information_agent', 'utility_agent', 'productivity_agent', 'tool_specialist']
```

#### Test Individual Agents

```bash
# Test basic agent functionality
python -c "
import asyncio
from voice_agent.core.multi_agent_service import MultiAgentService
from voice_agent.core.config import Config

async def test_agents():
    config = Config.load('src/voice_agent/config/default.yaml')
    service = MultiAgentService(config)
    await service.initialize()

    # Test routing
    result = await service.route_query('What is 2 + 2?')
    print('Calculation result:', result)

    await service.cleanup()

asyncio.run(test_agents())
"
```

### First Multi-Agent Workflows

#### 1. Simple Delegation Test

**Command**: `"What is the weather in Paris?"`

**Expected Flow**:

- System identifies this as weather query
- Routes to InformationAgent
- Agent uses WeatherTool
- Returns weather information

#### 2. Multi-Step Workflow Test

**Command**: `"Get weather for Tokyo and save it to weather_data.txt"`

**Expected Flow**:

1. System creates pipeline workflow
2. InformationAgent gets weather data
3. ProductivityAgent saves to file
4. Confirmation of both operations

#### 3. Collaborative Analysis Test

**Command**: `"Calculate 15% of 250 and explain the calculation"`

**Expected Flow**:

1. UtilityAgent performs calculation
2. GeneralAgent provides explanation
3. Combined response with calculation and explanation

### Testing Your Setup

#### 1. Text Mode Testing (Recommended First)

```bash
# Start in text-only mode for initial testing
python -m voice_agent.main --multi-agent --no-audio --debug
```

Try these test commands:

```
# Basic agent routing
"What is 5 * 7?"  # Should route to UtilityAgent
"What's the weather in London?"  # Should route to InformationAgent
"List files in current directory"  # Should route to ProductivityAgent

# Multi-step workflows
"Get weather for Paris and save it to weather.txt"
"Calculate 20% of 150 and save the result"
"Search for Python tutorials and summarize findings"
```

#### 2. Voice Mode Testing

```bash
# Start with voice enabled
python -m voice_agent.main --multi-agent --debug
```

**Voice Commands to Try**:

- "Hello, can you help me with some calculations?"
- "What's the weather like in New York?"
- "Get weather for London and save it to a file"
- "Calculate mortgage payment for 300,000 at 5% for 30 years"

#### 3. Verification Checklist

- [ ] Multi-agent system starts without errors
- [ ] Agents are properly loaded and configured
- [ ] Simple queries route to correct agents
- [ ] Multi-step workflows execute successfully
- [ ] Voice recognition works (if using voice mode)
- [ ] Text-to-speech works (if using voice mode)
- [ ] File operations complete successfully
- [ ] Debug logs show agent routing decisions

### Common Setup Issues

#### Issue: "Agent not found" Error

**Solution**:

```yaml
# Ensure all required agents are defined in config
multi_agent:
  agents:
    general_agent:
      type: "GeneralAgent"
      # ... configuration
    information_agent:
      type: "InformationAgent"
      # ... configuration
    # etc.
```

#### Issue: Multi-Agent Features Not Working

**Troubleshooting**:

```bash
# Check if multi-agent is enabled
python -c "
from voice_agent.core.config import Config
config = Config.load('src/voice_agent/config/default.yaml')
print('Multi-agent enabled:', config.multi_agent.enabled)
"

# Check for configuration errors
python -m voice_agent.main --multi-agent --debug 2>&1 | grep -i error
```

#### Issue: Audio Problems

**Solutions**:

```bash
# List audio devices
python -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(f'{i}: {p.get_device_info_by_index(i)}')
p.terminate()
"

# Test with different audio settings
# Edit default.yaml:
audio:
  input_device: null    # or specific device number
  output_device: null   # or specific device number
  sample_rate: 44100    # try different rates: 16000, 22050, 44100
```

#### Issue: Slow Performance

**Optimizations**:

```yaml
# Use smaller model
llm:
  model: "llama3.2:1b" # Faster than 3b

# Reduce concurrent agents
multi_agent:
  max_concurrent_agents: 3

# Optimize audio
audio:
  chunk_size: 512 # Smaller chunks for lower latency
```

### Advanced Configuration

#### Custom Agent Behavior

```yaml
multi_agent:
  agents:
    information_agent:
      system_prompt: "You are an expert researcher. Provide detailed, accurate information with sources when possible."
      max_concurrent_tasks: 4
      timeout_seconds: 45

    utility_agent:
      system_prompt: "You are a mathematics expert. Provide step-by-step explanations for calculations."
      max_concurrent_tasks: 4
      timeout_seconds: 30
```

#### Routing Customization

```yaml
multi_agent:
  routing_rules:
    - name: "complex_math"
      target_agent: "utility_agent"
      patterns:
        - "calculate"
        - "solve"
        - "equation"
        - "formula"
      priority: 1
      confidence: 0.9
```

#### Performance Tuning

```yaml
multi_agent:
  workflow_orchestration:
    default_timeout: 180 # Increase for complex tasks
    enable_parallel_execution: true
    task_dependency_timeout: 60

  inter_agent_communication:
    message_queue_size: 2000 # Increase for high throughput
    collaboration_timeout: 180 # Longer for complex collaboration
```

### Next Steps

Once your multi-agent system is working:

1. **Explore Workflows**: Try more complex multi-step commands
2. **Custom Tools**: Add your own tools following the [custom tool guide](multi_agent_usage_guide.md#creating-custom-tools)
3. **Agent Customization**: Modify agent behavior and capabilities
4. **Integration**: Connect with external APIs and services
5. **Advanced Features**: Explore consensus delegation and collaborative reasoning

### Getting Help

If you encounter issues:

1. **Check Logs**: Use `--debug` flag for detailed logging
2. **Review Configuration**: Ensure YAML syntax is correct
3. **Test Components**: Test individual agents and tools
4. **Documentation**: See [Multi-Agent Usage Guide](multi_agent_usage_guide.md) for detailed examples
5. **API Reference**: Check [API Reference](multi_agent_api_reference.md) for technical details

### Support Resources

- **Examples**: [`src/voice_agent/examples/`](../src/voice_agent/examples/)
- **Tests**: Run existing tests to verify functionality
- **Documentation**: Complete documentation in [`docs/`](../) directory
- **Configuration**: Template configurations in [`src/voice_agent/config/`](../src/voice_agent/config/)

## 🎯 Success Indicators

You'll know your multi-agent setup is working when:

- ✅ Different types of queries route to appropriate agents
- ✅ Multi-step workflows execute without errors
- ✅ Context is preserved across agent handoffs
- ✅ Voice interaction is smooth and responsive
- ✅ File operations and tool usage work seamlessly
- ✅ Debug logs show clear agent routing decisions

Welcome to the world of multi-agent AI assistance! 🤖🎉
