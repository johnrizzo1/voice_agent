# Multi-Agent Usage Guide

## Getting Started

This guide provides practical examples and step-by-step instructions for using the enhanced multi-agent system in the Voice Agent project.

## Quick Start

### Enabling Multi-Agent Features

1. **Update Configuration**

   Edit `src/voice_agent/config/default.yaml`:

   ```yaml
   multi_agent:
     enabled: true # Enable multi-agent functionality
     workflow_orchestration:
       enabled: true
     inter_agent_communication:
       enabled: true
     enhanced_delegation:
       enabled: true
   ```

2. **Start the Voice Agent**

   ```bash
   python -m voice_agent
   ```

3. **Test Multi-Agent Functionality**
   ```
   User: "Get the weather for Paris and save it to a file"
   ```

## Common Use Cases

### 1. Weather and File Operations

**Scenario**: Get weather data and save it to a file

**User Input**: "Get the weather for Tokyo and save it to weather_report.txt"

**What Happens**:

1. System creates a pipeline workflow
2. InformationAgent retrieves weather data
3. ProductivityAgent saves data to file
4. Results are aggregated and returned

**Expected Output**:

```
Weather for Tokyo: 22°C, partly cloudy
Successfully saved weather report to weather_report.txt
```

### 2. Research and Analysis

**Scenario**: Multi-agent research collaboration

**User Input**: "Research renewable energy trends and create a comprehensive analysis report"

**What Happens**:

1. System uses collaborative delegation
2. InformationAgent conducts web research
3. UtilityAgent performs calculations on data
4. ProductivityAgent creates and saves report
5. Results are combined into final analysis

### 3. Complex Calculations with Context

**Scenario**: Mathematical computation with explanation

**User Input**: "Calculate the compound interest for $10,000 at 5% annual rate for 10 years and explain the calculation"

**What Happens**:

1. UtilityAgent performs calculations
2. GeneralAgent provides explanation
3. Results are combined with detailed breakdown

### 4. Multi-Step Information Gathering

**Scenario**: Sequential information requests

**User Input**: "Get the latest news about AI, then search for related research papers"

**What Happens**:

1. Sequential workflow created
2. InformationAgent gets latest AI news
3. Second task searches for related research
4. Results are presented in order

## Workflow Types

### Sequential Workflows

Tasks execute one after another, with each task potentially using the output of the previous task.

**Example**: "Get weather, then calculate heating costs, then save the analysis"

```python
# Internal workflow structure
tasks = [
    {"agent": "information_agent", "task": "get weather"},
    {"agent": "utility_agent", "task": "calculate heating costs", "depends_on": ["get weather"]},
    {"agent": "productivity_agent", "task": "save analysis", "depends_on": ["calculate heating costs"]}
]
```

### Parallel Workflows

Independent tasks execute simultaneously for faster completion.

**Example**: "Get weather for London and New York simultaneously"

```python
# Internal workflow structure
tasks = [
    {"agent": "information_agent", "task": "get weather for London"},
    {"agent": "information_agent", "task": "get weather for New York"}
]
# Both tasks run in parallel
```

### Pipeline Workflows

Output of one task becomes input for the next task.

**Example**: "Get stock price, calculate percentage change, and generate investment advice"

```python
# Internal workflow structure
pipeline = [
    {"agent": "information_agent", "task": "get stock price"} →
    {"agent": "utility_agent", "task": "calculate percentage change"} →
    {"agent": "general_agent", "task": "generate investment advice"}
]
```

### Conditional Workflows

Tasks execute based on conditions or results from previous tasks.

**Example**: "Check the weather, and if it's raining, find indoor activities"

```python
# Internal workflow structure
if weather_result.contains("rain"):
    execute_task({"agent": "information_agent", "task": "find indoor activities"})
else:
    execute_task({"agent": "information_agent", "task": "find outdoor activities"})
```

## Delegation Patterns

### 1. Capability-Based Delegation

**When to Use**: When you need specific capabilities for a task

**Example Queries**:

- "Calculate the square root of 144" → UtilityAgent (calculations)
- "What's the weather like?" → InformationAgent (weather_info)
- "Save this data to a file" → ProductivityAgent (file_operations)

### 2. Load-Balanced Delegation

**When to Use**: When you have multiple agents with similar capabilities

**Example**: Multiple general queries distributed across available GeneralAgents

### 3. Expertise-Weighted Delegation

**When to Use**: When task quality depends on agent expertise

**Example**: Complex financial calculations routed to agents with highest calculation expertise

### 4. Collaborative Delegation

**When to Use**: When tasks benefit from multiple perspectives

**Example Queries**:

- "Analyze this business proposal from multiple angles"
- "Create a comprehensive market research report"
- "Develop a project plan with risk assessment"

### 5. Hierarchical Delegation

**When to Use**: When tasks have priority or authority requirements

**Example**: Escalating complex queries to more specialized agents

### 6. Consensus Delegation

**When to Use**: When accuracy is critical and you need agreement

**Example Queries**:

- "Verify these financial calculations"
- "Confirm this research data accuracy"
- "Validate this technical solution"

## Advanced Features

### Agent Collaboration Sessions

For complex tasks requiring sustained multi-agent interaction:

**Example Query**: "Create a business plan for a renewable energy startup"

**Process**:

1. Collaboration session initiated
2. InformationAgent researches market data
3. UtilityAgent performs financial projections
4. ProductivityAgent structures the plan
5. GeneralAgent provides strategic insights
6. Results aggregated into comprehensive business plan

### Context Preservation

The system maintains context across agent handoffs:

**Example Conversation**:

```
User: "Get weather for Paris"
Agent: "The weather in Paris is 18°C and cloudy"
User: "Save that information to a file"
Agent: "I've saved the Paris weather information (18°C, cloudy) to weather_data.txt"
```

The second request preserves context from the first interaction.

### Status Reporting

For long-running tasks, the system provides progress updates:

**Example**:

```
User: "Research AI trends and create a 50-page report"
System: "Starting research phase..."
System: "Research complete. Beginning analysis..."
System: "Analysis complete. Generating report..."
System: "Report generation complete. Saved to ai_trends_report.pdf"
```

## Configuration Examples

### Basic Multi-Agent Setup

```yaml
multi_agent:
  enabled: true
  default_agent: "general_agent"
  routing_strategy: "hybrid"
  confidence_threshold: 0.7
  max_concurrent_agents: 5
```

### Performance-Optimized Setup

```yaml
multi_agent:
  enabled: true
  max_concurrent_agents: 10
  workflow_orchestration:
    enabled: true
    max_concurrent_workflows: 5
    enable_parallel_execution: true
  inter_agent_communication:
    message_queue_size: 2000
    priority_messaging: true
```

### Collaboration-Focused Setup

```yaml
multi_agent:
  enabled: true
  enhanced_delegation:
    enabled: true
    patterns:
      - "collaborative"
      - "consensus"
    consensus_threshold: 0.7
    collaboration_min_agents: 3
  inter_agent_communication:
    collaboration_timeout: 180
    broadcast_enabled: true
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Tasks Taking Too Long

**Symptoms**: Workflows timeout or take excessive time

**Solutions**:

- Increase timeout values in configuration
- Use parallel execution for independent tasks
- Check agent load and distribute tasks better
- Optimize task granularity

**Configuration Fix**:

```yaml
workflow_orchestration:
  default_timeout: 600 # Increase from 300
  enable_parallel_execution: true
```

#### 2. Context Loss During Handoffs

**Symptoms**: Agents lose track of previous conversation context

**Solutions**:

- Enable enhanced context preservation
- Verify context compression settings
- Check context validation

**Configuration Fix**:

```yaml
context_preservation:
  enhanced_handoff: true
  context_compression: true
  context_validation: true
```

#### 3. Agent Selection Issues

**Symptoms**: Wrong agents selected for tasks

**Solutions**:

- Adjust confidence thresholds
- Review routing rules
- Check agent capability definitions
- Use explicit delegation patterns

**Configuration Fix**:

```yaml
multi_agent:
  confidence_threshold: 0.8 # Increase for more precise matching
  routing_strategy: "hybrid"
```

#### 4. Communication Failures

**Symptoms**: Agents can't communicate or collaborate

**Solutions**:

- Check message queue size
- Verify communication hub configuration
- Monitor network connectivity
- Review agent availability

**Configuration Fix**:

```yaml
inter_agent_communication:
  message_queue_size: 2000 # Increase queue size
  collaboration_timeout: 180 # Increase timeout
```

### Debugging Tips

#### 1. Enable Verbose Logging

```yaml
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

#### 2. Monitor Agent Status

Check agent availability and load:

- Review concurrent task counts
- Monitor agent response times
- Check error rates

#### 3. Analyze Workflow Execution

For workflow issues:

- Review task dependencies
- Check execution order
- Monitor resource usage
- Validate input/output data

#### 4. Test Individual Components

Isolate issues by testing:

- Individual agent responses
- Simple workflows before complex ones
- Communication between specific agents
- Context preservation in handoffs

## Performance Optimization

### Best Practices

#### 1. Task Granularity

- **Too Fine**: Overhead from many small tasks
- **Too Coarse**: Reduced parallelization opportunities
- **Optimal**: Balance between overhead and parallelization

#### 2. Agent Selection

- Use capability-based delegation for efficiency
- Leverage expertise weights for quality
- Consider load balancing for performance

#### 3. Workflow Design

- Use parallel execution for independent tasks
- Minimize context size in handoffs
- Cache frequently used results

#### 4. Resource Management

- Monitor agent capacity
- Distribute load evenly
- Set appropriate timeouts

### Performance Monitoring

Key metrics to track:

- **Task Completion Time**: Average time per task
- **Agent Utilization**: Percentage of time agents are busy
- **Workflow Success Rate**: Percentage of successful workflows
- **Context Transfer Time**: Time spent in agent handoffs
- **Queue Depth**: Number of pending messages

## Security Considerations

### Data Protection

- Sensitive data handling in multi-agent contexts
- Context sanitization during handoffs
- Access control for agent communications

### Agent Authentication

- Verify agent identity in communications
- Prevent agent impersonation
- Secure message channels

### Input Validation

- Sanitize all user inputs
- Validate task parameters
- Check context data integrity

## Integration Examples

### With External APIs

```python
# Example: Weather service integration
async def enhanced_weather_workflow(location):
    workflow = await orchestrator.create_workflow("weather_extended", "pipeline")

    # Get current weather
    await orchestrator.add_task(
        "weather_extended",
        "current_weather",
        "information_agent",
        f"get current weather for {location}"
    )

    # Get forecast
    await orchestrator.add_task(
        "weather_extended",
        "forecast",
        "information_agent",
        f"get 5-day forecast for {location}",
        dependencies=["current_weather"]
    )

    # Analyze data
    await orchestrator.add_task(
        "weather_extended",
        "analysis",
        "utility_agent",
        "analyze weather patterns and trends",
        dependencies=["current_weather", "forecast"]
    )

    return await orchestrator.execute_workflow("weather_extended")
```

### With Database Operations

```python
# Example: Data analysis workflow
async def data_analysis_workflow(dataset_id):
    workflow = await orchestrator.create_workflow("data_analysis", "sequential")

    # Load data
    await orchestrator.add_task(
        "data_analysis",
        "load_data",
        "productivity_agent",
        f"load dataset {dataset_id}"
    )

    # Analyze data
    await orchestrator.add_task(
        "data_analysis",
        "analyze",
        "utility_agent",
        "perform statistical analysis",
        dependencies=["load_data"]
    )

    # Generate report
    await orchestrator.add_task(
        "data_analysis",
        "report",
        "productivity_agent",
        "generate analysis report",
        dependencies=["analyze"]
    )

    return await orchestrator.execute_workflow("data_analysis")
```

## Future Enhancements

### Planned Features

1. **Dynamic Agent Creation**: Create specialized agents on demand
2. **Machine Learning Integration**: Improve agent selection through learning
3. **Advanced Reasoning**: Enhanced collaborative problem-solving
4. **External Service Integration**: Direct API connections
5. **Visual Workflow Designer**: GUI for workflow creation

### Experimental Features

1. **Emergent Behavior**: Study complex multi-agent interactions
2. **Adaptive Delegation**: Self-improving delegation patterns
3. **Predictive Context**: Anticipate context needs
4. **Cross-Session Learning**: Learn from previous interactions

## Getting Help

### Resources

- **Documentation**: `/docs/multi_agent_architecture.md`
- **API Reference**: `/docs/multi_agent_api_reference.md`
- **Example Code**: `/examples/multi_agent_examples.py`
- **Test Cases**: `/test_enhanced_multi_agent_workflows.py`

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Contributions**: Submit improvements and extensions

### Support

For technical support:

1. Check this usage guide
2. Review API documentation
3. Examine test cases for examples
4. Enable debug logging for troubleshooting
5. Report issues with detailed logs
