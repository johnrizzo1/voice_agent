# Multi-Agent API Reference

## Overview

This document provides detailed API reference for the enhanced multi-agent system components. It covers all public methods, classes, and interfaces available for developers working with the multi-agent architecture.

## Core Classes

### MultiAgentService

The central service for multi-agent coordination and management.

#### Methods

##### `async process_workflow(query: str, workflow_type: str = "sequential", timeout: int = 300) -> Dict[str, Any]`

Process a complex multi-step workflow using multiple agents.

**Parameters:**

- `query` (str): The user query describing the workflow
- `workflow_type` (str): Type of workflow execution ("sequential", "parallel", "pipeline", "conditional")
- `timeout` (int): Maximum execution time in seconds

**Returns:**

- `Dict[str, Any]`: Workflow execution results including task outputs and metadata

**Example:**

```python
result = await service.process_workflow(
    "get weather for Tokyo and save to file",
    workflow_type="pipeline",
    timeout=120
)
```

##### `async request_agent_collaboration(query: str, required_capabilities: List[str], collaboration_type: str = "parallel") -> Dict[str, Any]`

Request collaboration between multiple agents for complex tasks.

**Parameters:**

- `query` (str): The task description
- `required_capabilities` (List[str]): List of required agent capabilities
- `collaboration_type` (str): Type of collaboration ("parallel", "sequential", "collaborative")

**Returns:**

- `Dict[str, Any]`: Collaboration results with aggregated outputs

**Example:**

```python
result = await service.request_agent_collaboration(
    "analyze market data and create report",
    required_capabilities=["web_search", "calculations", "file_operations"],
    collaboration_type="collaborative"
)
```

##### `async delegate_complex_task(query: str, delegation_pattern: str, **kwargs) -> Dict[str, Any]`

Delegate tasks using sophisticated delegation patterns.

**Parameters:**

- `query` (str): The task description
- `delegation_pattern` (str): Delegation pattern to use
- `**kwargs`: Additional pattern-specific parameters

**Returns:**

- `Dict[str, Any]`: Task execution results

**Delegation Patterns:**

- `"capability_based"`: Select based on capabilities
- `"load_balanced"`: Distribute load evenly
- `"expertise_weighted"`: Weight by expertise levels
- `"collaborative"`: Multi-agent collaboration
- `"hierarchical"`: Priority-based selection
- `"consensus"`: Require consensus from multiple agents

**Example:**

```python
result = await service.delegate_complex_task(
    "financial analysis with risk assessment",
    delegation_pattern="expertise_weighted",
    expertise_weights={"calculations": 0.9, "web_search": 0.7}
)
```

##### `async handoff_to_agent(target_agent_id: str, context: Dict[str, Any], query: str) -> Dict[str, Any]`

Enhanced agent handoff with improved context preservation.

**Parameters:**

- `target_agent_id` (str): ID of the target agent
- `context` (Dict[str, Any]): Current conversation context
- `query` (str): Query to pass to the target agent

**Returns:**

- `Dict[str, Any]`: Agent response with preserved context

### WorkflowOrchestrator

Manages complex multi-step workflows with task dependencies.

#### Methods

##### `async create_workflow(workflow_id: str, workflow_type: str = "sequential") -> WorkflowExecution`

Create a new workflow execution instance.

**Parameters:**

- `workflow_id` (str): Unique workflow identifier
- `workflow_type` (str): Execution type ("sequential", "parallel", "pipeline", "conditional")

**Returns:**

- `WorkflowExecution`: Workflow execution instance

##### `async add_task(workflow_id: str, task_id: str, agent_id: str, query: str, dependencies: List[str] = None) -> WorkflowTask`

Add a task to an existing workflow.

**Parameters:**

- `workflow_id` (str): Workflow identifier
- `task_id` (str): Unique task identifier
- `agent_id` (str): Agent to execute the task
- `query` (str): Task query/instruction
- `dependencies` (List[str]): List of task IDs this task depends on

**Returns:**

- `WorkflowTask`: Created task instance

##### `async execute_workflow(workflow_id: str, timeout: int = 300) -> Dict[str, Any]`

Execute a workflow with all its tasks.

**Parameters:**

- `workflow_id` (str): Workflow to execute
- `timeout` (int): Maximum execution time in seconds

**Returns:**

- `Dict[str, Any]`: Workflow execution results

### CommunicationHub

Handles inter-agent communication and collaboration sessions.

#### Methods

##### `async send_message(from_agent: str, to_agent: str, message: AgentMessage) -> bool`

Send a direct message between agents.

**Parameters:**

- `from_agent` (str): Sender agent ID
- `to_agent` (str): Recipient agent ID
- `message` (AgentMessage): Message to send

**Returns:**

- `bool`: Success status

##### `async broadcast_message(from_agent: str, message: AgentMessage, target_capabilities: List[str] = None) -> List[str]`

Broadcast a message to multiple agents.

**Parameters:**

- `from_agent` (str): Sender agent ID
- `message` (AgentMessage): Message to broadcast
- `target_capabilities` (List[str]): Filter recipients by capabilities

**Returns:**

- `List[str]`: List of agent IDs that received the message

##### `async start_collaboration_session(session_id: str, participants: List[str], task_description: str) -> CollaborationSession`

Start a multi-agent collaboration session.

**Parameters:**

- `session_id` (str): Unique session identifier
- `participants` (List[str]): List of participant agent IDs
- `task_description` (str): Description of the collaborative task

**Returns:**

- `CollaborationSession`: Created collaboration session

##### `async request_response(from_agent: str, to_agent: str, query: str, timeout: int = 30) -> Dict[str, Any]`

Send a request and wait for a response.

**Parameters:**

- `from_agent` (str): Requesting agent ID
- `to_agent` (str): Target agent ID
- `query` (str): Request query
- `timeout` (int): Response timeout in seconds

**Returns:**

- `Dict[str, Any]`: Agent response

### EnhancedDelegationManager

Manages sophisticated agent delegation patterns.

#### Methods

##### `async delegate_by_capability(query: str, required_capabilities: List[str]) -> str`

Delegate based on required capabilities.

**Parameters:**

- `query` (str): Task description
- `required_capabilities` (List[str]): Required agent capabilities

**Returns:**

- `str`: Selected agent ID

##### `async delegate_load_balanced(query: str, candidate_agents: List[str] = None) -> str`

Delegate using load balancing.

**Parameters:**

- `query` (str): Task description
- `candidate_agents` (List[str]): Candidate agents (None for all)

**Returns:**

- `str`: Selected agent ID

##### `async delegate_expertise_weighted(query: str, expertise_weights: Dict[str, float]) -> str`

Delegate based on expertise weights.

**Parameters:**

- `query` (str): Task description
- `expertise_weights` (Dict[str, float]): Capability expertise weights

**Returns:**

- `str`: Selected agent ID

##### `async delegate_collaborative(query: str, min_agents: int = 2, max_agents: int = 4) -> List[str]`

Delegate to multiple agents for collaboration.

**Parameters:**

- `query` (str): Task description
- `min_agents` (int): Minimum number of agents
- `max_agents` (int): Maximum number of agents

**Returns:**

- `List[str]`: Selected agent IDs

##### `async delegate_consensus(query: str, candidate_agents: List[str], consensus_threshold: float = 0.6) -> List[str]`

Delegate requiring consensus from multiple agents.

**Parameters:**

- `query` (str): Task description
- `candidate_agents` (List[str]): Candidate agents
- `consensus_threshold` (float): Required agreement threshold

**Returns:**

- `List[str]`: Selected agent IDs

## Data Classes

### AgentMessage

Represents a message between agents.

#### Attributes

- `message_id` (str): Unique message identifier
- `message_type` (MessageType): Type of message
- `content` (str): Message content
- `metadata` (Dict[str, Any]): Additional message metadata
- `priority` (int): Message priority (1-10)
- `timestamp` (datetime): Message creation timestamp

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert message to dictionary format.

##### `from_dict(data: Dict[str, Any]) -> AgentMessage`

Create message from dictionary (class method).

### WorkflowTask

Represents a task within a workflow.

#### Attributes

- `task_id` (str): Unique task identifier
- `agent_id` (str): Assigned agent ID
- `query` (str): Task query/instruction
- `dependencies` (List[str]): Task dependencies
- `status` (TaskStatus): Current task status
- `result` (Dict[str, Any]): Task execution result
- `error` (str): Error message if failed
- `start_time` (datetime): Task start timestamp
- `end_time` (datetime): Task completion timestamp

### WorkflowExecution

Represents a workflow execution instance.

#### Attributes

- `workflow_id` (str): Unique workflow identifier
- `workflow_type` (str): Execution type
- `tasks` (Dict[str, WorkflowTask]): Workflow tasks
- `status` (WorkflowStatus): Current workflow status
- `results` (Dict[str, Any]): Aggregated results
- `start_time` (datetime): Workflow start timestamp
- `end_time` (datetime): Workflow completion timestamp

#### Methods

##### `add_task(task: WorkflowTask) -> None`

Add a task to the workflow.

##### `get_ready_tasks() -> List[WorkflowTask]`

Get tasks ready for execution (dependencies satisfied).

##### `is_complete() -> bool`

Check if workflow execution is complete.

### CollaborationSession

Represents a multi-agent collaboration session.

#### Attributes

- `session_id` (str): Unique session identifier
- `participants` (List[str]): Participant agent IDs
- `task_description` (str): Collaborative task description
- `messages` (List[AgentMessage]): Session messages
- `results` (Dict[str, Any]): Collaboration results
- `status` (SessionStatus): Current session status
- `start_time` (datetime): Session start timestamp
- `end_time` (datetime): Session completion timestamp

#### Methods

##### `add_participant(agent_id: str) -> None`

Add a participant to the session.

##### `send_message(message: AgentMessage) -> None`

Send a message in the session.

##### `get_aggregated_results() -> Dict[str, Any]`

Get aggregated results from all participants.

## Enums

### MessageType

Message types for agent communication.

- `TASK_REQUEST`: Task delegation request
- `INFORMATION_SHARING`: Information exchange
- `STATUS_UPDATE`: Progress update
- `COLLABORATION_INVITE`: Collaboration invitation
- `RESULT`: Task completion result
- `ERROR`: Error notification

### TaskStatus

Task execution status.

- `PENDING`: Task not started
- `RUNNING`: Task in progress
- `COMPLETED`: Task completed successfully
- `FAILED`: Task failed
- `CANCELLED`: Task was cancelled

### WorkflowStatus

Workflow execution status.

- `CREATED`: Workflow created but not started
- `RUNNING`: Workflow in progress
- `COMPLETED`: Workflow completed successfully
- `FAILED`: Workflow failed
- `CANCELLED`: Workflow was cancelled

### SessionStatus

Collaboration session status.

- `ACTIVE`: Session is active
- `COMPLETED`: Session completed successfully
- `FAILED`: Session failed
- `CANCELLED`: Session was cancelled

## Configuration Classes

### WorkflowConfig

Configuration for workflow orchestration.

#### Attributes

- `enabled` (bool): Enable workflow orchestration
- `max_concurrent_workflows` (int): Maximum concurrent workflows
- `default_timeout` (int): Default timeout in seconds
- `enable_parallel_execution` (bool): Enable parallel execution
- `enable_pipeline_execution` (bool): Enable pipeline execution
- `task_dependency_timeout` (int): Task dependency timeout

### CommunicationConfig

Configuration for inter-agent communication.

#### Attributes

- `enabled` (bool): Enable inter-agent communication
- `message_queue_size` (int): Message queue size
- `collaboration_timeout` (int): Collaboration timeout
- `broadcast_enabled` (bool): Enable broadcast messaging
- `priority_messaging` (bool): Enable priority messaging

### DelegationConfig

Configuration for enhanced delegation.

#### Attributes

- `enabled` (bool): Enable enhanced delegation
- `patterns` (List[str]): Available delegation patterns
- `consensus_threshold` (float): Consensus threshold
- `collaboration_min_agents` (int): Minimum agents for collaboration
- `expertise_weights` (Dict[str, float]): Default expertise weights

## Usage Examples

### Basic Workflow Creation

```python
from voice_agent.core.multi_agent_service import MultiAgentService
from voice_agent.core.multi_agent.workflow import WorkflowOrchestrator

# Initialize services
service = MultiAgentService()
orchestrator = WorkflowOrchestrator()

# Create workflow
workflow = await orchestrator.create_workflow("weather_workflow", "pipeline")

# Add tasks
await orchestrator.add_task(
    "weather_workflow",
    "get_weather",
    "information_agent",
    "get weather for London"
)

await orchestrator.add_task(
    "weather_workflow",
    "save_weather",
    "productivity_agent",
    "save weather data to file",
    dependencies=["get_weather"]
)

# Execute workflow
result = await orchestrator.execute_workflow("weather_workflow")
```

### Agent Collaboration

```python
from voice_agent.core.multi_agent.communication import CommunicationHub, AgentMessage

# Initialize communication hub
hub = CommunicationHub()

# Start collaboration session
session = await hub.start_collaboration_session(
    "analysis_session",
    ["information_agent", "utility_agent", "productivity_agent"],
    "comprehensive market analysis"
)

# Send collaboration message
message = AgentMessage(
    message_type=MessageType.COLLABORATION_INVITE,
    content="Please analyze renewable energy market trends",
    priority=5
)

await hub.send_message("information_agent", "utility_agent", message)
```

### Custom Delegation

```python
from voice_agent.core.multi_agent.communication import EnhancedDelegationManager

# Initialize delegation manager
manager = EnhancedDelegationManager()

# Capability-based delegation
agent_id = await manager.delegate_by_capability(
    "calculate compound interest",
    required_capabilities=["calculations"]
)

# Expertise-weighted delegation
agent_id = await manager.delegate_expertise_weighted(
    "financial analysis",
    expertise_weights={"calculations": 0.9, "web_search": 0.7}
)

# Collaborative delegation
agent_ids = await manager.delegate_collaborative(
    "comprehensive research project",
    min_agents=2,
    max_agents=4
)
```

## Error Handling

### Common Exceptions

#### `WorkflowExecutionError`

Raised when workflow execution fails.

#### `AgentCommunicationError`

Raised when agent communication fails.

#### `DelegationError`

Raised when task delegation fails.

#### `ContextPreservationError`

Raised when context preservation fails during handoffs.

### Error Handling Example

```python
try:
    result = await service.process_workflow(
        "complex multi-step task",
        workflow_type="pipeline"
    )
except WorkflowExecutionError as e:
    logger.error(f"Workflow failed: {e}")
    # Handle workflow failure
except AgentCommunicationError as e:
    logger.error(f"Communication failed: {e}")
    # Handle communication failure
```

## Best Practices

### Performance Optimization

1. **Use appropriate workflow types** for your use case
2. **Minimize context size** during agent handoffs
3. **Leverage parallel execution** when tasks are independent
4. **Cache frequently used results** in collaboration sessions
5. **Monitor agent load** and use load balancing

### Error Resilience

1. **Implement proper error handling** for all async operations
2. **Use timeouts** to prevent hanging operations
3. **Provide fallback mechanisms** for critical tasks
4. **Log detailed error information** for debugging
5. **Implement retry logic** for transient failures

### Security Considerations

1. **Validate all inputs** before processing
2. **Sanitize context data** during handoffs
3. **Implement access controls** for sensitive operations
4. **Monitor agent communications** for anomalies
5. **Use secure channels** for inter-agent messaging
