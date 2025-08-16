# LlamaIndex Integration Guide

This guide explains how to use LlamaIndex with the voice agent for multi-agent functionality and advanced AI capabilities.

## Overview

LlamaIndex has been integrated into the voice agent to provide:

- **Multi-agent functionality** with ReAct agents
- **Vector store capabilities** for RAG (Retrieval-Augmented Generation)
- **Enhanced tool integration** through LlamaIndex's agent framework
- **Local model support** via Ollama integration

## Prerequisites

1. **Ollama running** with models available:

   ```bash
   ollama serve
   ollama pull mistral:7b  # or your preferred model
   ```

2. **Development environment** with LlamaIndex dependencies:
   ```bash
   devenv shell  # All dependencies are automatically installed
   ```

## Quick Start

### Basic LlamaIndex Service

```python
from voice_agent.core.config import Config
from voice_agent.core.llamaindex_service import LlamaIndexService

# Initialize with existing voice agent config
config = Config()
service = LlamaIndexService(config.llm)

# Initialize the service (connects to Ollama)
await service.initialize()

# Create a simple agent
await service.create_agent()

# Chat with the agent
response = await service.chat("Hello! What is 2+2?")
print(response)

# Cleanup
await service.cleanup()
```

### Vector Store for RAG

```python
# Create vector store from documents
await service.create_vector_index("path/to/documents/")

# Query the knowledge base
response = await service.query_vector_index("What does the documentation say about setup?")
print(response)
```

## Integration with Voice Agent Tools

The LlamaIndex service can use existing voice agent tools:

```python
from voice_agent.tools.builtin import CalculatorTool, WeatherTool
from voice_agent.tools.registry import get_registry

# Load existing tools
registry = get_registry()
registry.register_tool(CalculatorTool())
registry.register_tool(WeatherTool())

# Get tools in LlamaIndex format
tools_info = registry.get_all_tools_info()

# Create agent with tools
await service.create_agent(tools_info)

# Now the agent can use calculator and weather tools
response = await service.chat("Calculate 15 * 7 and then tell me the weather in London")
```

## Configuration

LlamaIndex uses the same LLM configuration as the main voice agent:

```yaml
# In voice_agent/config/default.yaml
llm:
  provider: "ollama"
  model: "mistral:7b"
  temperature: 0.7
  max_tokens: 512
  base_url: "http://localhost:11434"
```

## Available Models

The integration supports any model available in Ollama:

- **mistral:7b** - Good balance of speed and capability (default)
- **llama2:7b** - Alternative general-purpose model
- **codellama:7b** - Better for code-related tasks
- **llama2:13b** - Higher capability, slower
- **nomic-embed-text** - Used for embeddings (auto-pulled)

## Testing

Run the integration test to verify everything works:

```bash
# Basic LlamaIndex + Ollama test
python test_llamaindex_integration.py

# Test with voice agent tools (if available)
python test_integration_with_voice_agent.py
```

## API Reference

### LlamaIndexService

#### `__init__(config: LLMConfig, state_callback=None)`

Initialize the service with LLM configuration.

#### `async initialize() -> None`

Initialize Ollama connection and LlamaIndex components.

#### `async create_agent(tools: List[Any] = None) -> None`

Create a ReAct agent with optional tools.

#### `async chat(message: str) -> str`

Send a message to the agent and get response.

#### `async create_vector_index(documents_path: str = None) -> None`

Create vector store for RAG capabilities.

#### `async query_vector_index(query: str) -> str`

Query the vector store for relevant information.

#### `async get_service_info() -> Dict[str, Any]`

Get service status and configuration info.

#### `async cleanup() -> None`

Cleanup resources and connections.

## Use Cases

### 1. Enhanced Conversation Agent

Replace or augment the basic LLM service with LlamaIndex for better reasoning:

```python
# Instead of basic LLM service
service = LlamaIndexService(config.llm)
await service.initialize()
await service.create_agent()

response = await service.chat("Plan a three-day trip to Paris")
```

### 2. Document Q&A

Create a knowledge base from documents:

```python
# Index project documentation
await service.create_vector_index("docs/")

# Query the documentation
answer = await service.query_vector_index("How do I configure the TTS service?")
```

### 3. Multi-Tool Agent

Create agents that can use multiple tools in sequence:

```python
# Load all available tools
tools = load_all_voice_agent_tools()
await service.create_agent(tools)

# Complex multi-step request
response = await service.chat(
    "Calculate the square root of 144, then search for weather in that many cities"
)
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Voice Agent    │───▶│  LlamaIndex      │───▶│  Ollama LLM     │
│  Tools          │    │  Service         │    │  Service        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │  ReAct Agent     │
                       │  Vector Store    │
                       │  Tool Execution  │
                       └──────────────────┘
```

## Dependencies Added

The following LlamaIndex packages are included in the development environment:

- `llama-index-core` - Core LlamaIndex functionality
- `llama-index-llms-ollama` - Ollama LLM integration
- `llama-index-embeddings-ollama` - Ollama embeddings
- `llama-index-agent-openai` - Agent functionality
- `llama-index-readers-file` - Document loading
- `llama-index-vector-stores-chroma` - Vector storage

## Performance Considerations

1. **Model Selection**: Larger models (13B+) provide better reasoning but are slower
2. **Vector Indexing**: Large document sets may take time to index initially
3. **Agent Iterations**: Complex queries may require multiple LLM calls
4. **Memory Usage**: Vector stores and embeddings use additional RAM

## Troubleshooting

### LlamaIndex Not Available

```
❌ LlamaIndex is not available!
```

**Solution**: Ensure you're in the devenv shell: `devenv shell`

### Ollama Connection Failed

```
❌ Ollama connection failed
```

**Solution**: Start Ollama service: `ollama serve`

### Model Not Found

```
⚠️ mistral:7b model not found
```

**Solution**: Pull the model: `ollama pull mistral:7b`

### Agent Max Iterations

```
Chat error: Reached max iterations
```

**Solution**: This happens when the agent can't find appropriate tools. Either provide relevant tools or ask simpler questions.

## Future Enhancements

- **Streaming responses** for real-time agent output
- **Custom agent workflows** for specific use cases
- **Multi-modal capabilities** with vision models
- **Agent memory** for long-term conversations
- **Distributed agents** for complex task orchestration

## Examples

See the test files for complete working examples:

- `test_llamaindex_integration.py` - Basic integration
- `test_integration_with_voice_agent.py` - Tool integration

The LlamaIndex integration provides a powerful foundation for building sophisticated AI agents while maintaining compatibility with the existing voice agent system.
