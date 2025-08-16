"""
Multi-agent workflow orchestration system.

Provides sophisticated workflow management for complex multi-agent tasks,
including task decomposition, parallel execution, result aggregation,
and advanced communication patterns.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object

    def Field(**kwargs):
        return None


from .message import AgentMessage, MessageType
from .agent_base import AgentBase, AgentCapability


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Status of individual tasks within a workflow."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskPriority(str, Enum):
    """Priority levels for workflow tasks."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ExecutionMode(str, Enum):
    """Execution modes for workflows."""

    SEQUENTIAL = "sequential"  # Tasks execute one after another
    PARALLEL = "parallel"  # Tasks execute simultaneously
    PIPELINE = "pipeline"  # Output of one task feeds into next
    CONDITIONAL = "conditional"  # Tasks execute based on conditions


@dataclass
class WorkflowTask:
    """Individual task within a workflow."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    agent_id: Optional[str] = None
    required_capabilities: List[AgentCapability] = field(default_factory=list)

    # Task execution
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL

    # Dependencies and ordering
    depends_on: List[str] = field(default_factory=list)  # Task IDs this depends on
    blocks: List[str] = field(default_factory=list)  # Task IDs this blocks

    # Execution constraints
    max_retries: int = 3
    timeout_seconds: float = 30.0
    retry_count: int = 0

    # Results and errors
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_ready_to_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute based on dependencies."""
        return self.status == TaskStatus.PENDING and all(
            dep_id in completed_tasks for dep_id in self.depends_on
        )

    def execution_time(self) -> Optional[float]:
        """Get task execution time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class WorkflowDefinition(BaseModel if BaseModel != object else dict):
    """Definition of a multi-agent workflow."""

    if BaseModel != object:
        workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        name: str
        description: str = ""
        version: str = "1.0"

        # Execution configuration
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
        max_parallel_tasks: int = 3
        overall_timeout_seconds: float = 300.0

        # Task definitions
        tasks: List[Dict[str, Any]] = Field(default_factory=list)

        # Workflow metadata
        created_by: Optional[str] = None
        created_at: datetime = Field(default_factory=datetime.utcnow)
        metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecution:
    """Manages the execution of a multi-agent workflow."""

    def __init__(
        self,
        workflow_def: WorkflowDefinition,
        agent_registry: Dict[str, AgentBase],
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize workflow execution.

        Args:
            workflow_def: Workflow definition to execute
            agent_registry: Available agents for task execution
            state_callback: Optional callback for status updates
        """
        self.workflow_def = workflow_def
        self.agent_registry = agent_registry
        self.logger = logging.getLogger(__name__)
        self._state_callback = state_callback

        # Execution state
        self.execution_id = str(uuid.uuid4())
        self.status = WorkflowStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Task management
        self.tasks: Dict[str, WorkflowTask] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.running_tasks: Dict[str, asyncio.Task] = {}

        # Results and communication
        self.workflow_results: Dict[str, Any] = {}
        self.task_communications: List[Dict[str, Any]] = []
        self.error_log: List[Dict[str, Any]] = []

        # Build task graph from definition
        self._build_task_graph()

    def _emit_state(self, state: str, message: Optional[str] = None) -> None:
        """Emit state change via callback."""
        if self._state_callback:
            try:
                self._state_callback("workflow", state, message)
            except Exception:
                self.logger.debug("Workflow state callback error", exc_info=True)

    def _build_task_graph(self) -> None:
        """Build task graph from workflow definition."""
        tasks_data = (
            self.workflow_def.tasks
            if hasattr(self.workflow_def, "tasks")
            else self.workflow_def.get("tasks", [])
        )

        for task_data in tasks_data:
            # Convert capability strings to enums
            capabilities = []
            for cap_str in task_data.get("required_capabilities", []):
                try:
                    capability = AgentCapability(cap_str)
                    capabilities.append(capability)
                except ValueError:
                    self.logger.warning(f"Unknown capability: {cap_str}")

            task = WorkflowTask(
                task_id=task_data.get("task_id", str(uuid.uuid4())),
                name=task_data.get("name", ""),
                description=task_data.get("description", ""),
                agent_id=task_data.get("agent_id"),
                required_capabilities=capabilities,
                input_data=task_data.get("input_data", {}),
                depends_on=task_data.get("depends_on", []),
                blocks=task_data.get("blocks", []),
                max_retries=task_data.get("max_retries", 3),
                timeout_seconds=task_data.get("timeout_seconds", 30.0),
                priority=TaskPriority(task_data.get("priority", "normal")),
                metadata=task_data.get("metadata", {}),
            )

            self.tasks[task.task_id] = task

        self.logger.info(f"Built workflow with {len(self.tasks)} tasks")

    async def execute(self) -> Dict[str, Any]:
        """
        Execute the workflow according to its execution mode.

        Returns:
            Workflow execution results
        """
        self.logger.info(
            f"Starting workflow execution: {self.workflow_def.name if hasattr(self.workflow_def, 'name') else self.workflow_def.get('name', 'unnamed')}"
        )
        self._emit_state("running", f"executing workflow {self.execution_id}")

        self.status = WorkflowStatus.RUNNING
        self.start_time = datetime.utcnow()

        try:
            execution_mode = (
                self.workflow_def.execution_mode
                if hasattr(self.workflow_def, "execution_mode")
                else ExecutionMode(
                    self.workflow_def.get("execution_mode", "sequential")
                )
            )

            if execution_mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential()
            elif execution_mode == ExecutionMode.PARALLEL:
                await self._execute_parallel()
            elif execution_mode == ExecutionMode.PIPELINE:
                await self._execute_pipeline()
            elif execution_mode == ExecutionMode.CONDITIONAL:
                await self._execute_conditional()
            else:
                raise ValueError(f"Unknown execution mode: {execution_mode}")

            # Check overall success
            if self.failed_tasks:
                self.status = WorkflowStatus.FAILED
                self._emit_state("failed", f"{len(self.failed_tasks)} tasks failed")
            else:
                self.status = WorkflowStatus.COMPLETED
                self._emit_state("completed", "all tasks completed successfully")

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            self.status = WorkflowStatus.FAILED
            self.error_log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                    "source": "workflow_execution",
                }
            )
            self._emit_state("failed", f"workflow error: {str(e)}")

        finally:
            self.end_time = datetime.utcnow()

        return self._build_results()

    async def _execute_sequential(self) -> None:
        """Execute tasks sequentially according to dependency order."""
        task_queue = self._build_execution_queue()

        for task in task_queue:
            if task.status != TaskStatus.PENDING:
                continue

            await self._execute_single_task(task)

            if (
                task.status == TaskStatus.FAILED
                and task.priority == TaskPriority.CRITICAL
            ):
                self.logger.error(
                    f"Critical task {task.task_id} failed, stopping workflow"
                )
                break

    async def _execute_parallel(self) -> None:
        """Execute tasks in parallel with concurrency limits."""
        max_parallel = (
            self.workflow_def.max_parallel_tasks
            if hasattr(self.workflow_def, "max_parallel_tasks")
            else self.workflow_def.get("max_parallel_tasks", 3)
        )
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_semaphore(task: WorkflowTask):
            async with semaphore:
                await self._execute_single_task(task)

        # Start all ready tasks
        while True:
            ready_tasks = [
                task
                for task in self.tasks.values()
                if task.is_ready_to_execute(self.completed_tasks)
                and task.task_id not in self.running_tasks
            ]

            if not ready_tasks and not self.running_tasks:
                break  # No more tasks to run

            # Start new tasks
            for task in ready_tasks:
                task_future = asyncio.create_task(execute_with_semaphore(task))
                self.running_tasks[task.task_id] = task_future

            # Wait for at least one task to complete
            if self.running_tasks:
                done, pending = await asyncio.wait(
                    self.running_tasks.values(), return_when=asyncio.FIRST_COMPLETED
                )

                # Clean up completed tasks
                for task_id in list(self.running_tasks.keys()):
                    if self.running_tasks[task_id] in done:
                        del self.running_tasks[task_id]

    async def _execute_pipeline(self) -> None:
        """Execute tasks in pipeline mode where outputs feed into inputs."""
        task_queue = self._build_execution_queue()
        pipeline_data = {}

        for task in task_queue:
            if task.status != TaskStatus.PENDING:
                continue

            # Merge pipeline data into task input
            task.input_data.update(pipeline_data)

            await self._execute_single_task(task)

            # Pass output to pipeline data
            if task.status == TaskStatus.COMPLETED and task.output_data:
                pipeline_data.update(task.output_data)

            if task.status == TaskStatus.FAILED:
                self.logger.error(f"Pipeline broken at task {task.task_id}")
                break

    async def _execute_conditional(self) -> None:
        """Execute tasks based on conditional logic."""
        # This is a simplified conditional execution
        # In a full implementation, you'd parse conditions from metadata
        task_queue = self._build_execution_queue()

        for task in task_queue:
            if task.status != TaskStatus.PENDING:
                continue

            # Check conditions (simplified example)
            conditions = task.metadata.get("conditions", {})
            if conditions:
                if not self._evaluate_conditions(conditions):
                    task.status = TaskStatus.CANCELLED
                    self.logger.info(f"Task {task.task_id} skipped due to conditions")
                    continue

            await self._execute_single_task(task)

    def _evaluate_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Evaluate task execution conditions."""
        # Simplified condition evaluation
        # In practice, this would be more sophisticated
        for condition_type, condition_value in conditions.items():
            if condition_type == "requires_success" and isinstance(
                condition_value, list
            ):
                for task_id in condition_value:
                    if task_id not in self.completed_tasks:
                        return False
            elif condition_type == "requires_failure" and isinstance(
                condition_value, list
            ):
                for task_id in condition_value:
                    if task_id not in self.failed_tasks:
                        return False

        return True

    def _build_execution_queue(self) -> List[WorkflowTask]:
        """Build ordered queue of tasks for execution."""
        # Topological sort to respect dependencies
        task_list = list(self.tasks.values())
        sorted_tasks = []
        remaining_tasks = task_list.copy()

        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task in remaining_tasks:
                if all(
                    dep_id in [t.task_id for t in sorted_tasks]
                    for dep_id in task.depends_on
                ):
                    ready_tasks.append(task)

            if not ready_tasks:
                # Circular dependency or other issue
                self.logger.warning(
                    "Circular dependency detected, executing remaining tasks"
                )
                ready_tasks = remaining_tasks

            # Sort by priority
            priority_order = {
                TaskPriority.CRITICAL: 0,
                TaskPriority.HIGH: 1,
                TaskPriority.NORMAL: 2,
                TaskPriority.LOW: 3,
            }
            ready_tasks.sort(key=lambda t: priority_order.get(t.priority, 2))

            # Add to execution queue
            sorted_tasks.extend(ready_tasks)
            for task in ready_tasks:
                remaining_tasks.remove(task)

        return sorted_tasks

    async def _execute_single_task(self, task: WorkflowTask) -> None:
        """Execute a single task with proper error handling and retries."""
        self.logger.info(f"Executing task {task.task_id}: {task.name}")
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.utcnow()

        for attempt in range(task.max_retries + 1):
            try:
                # Find appropriate agent
                agent = await self._find_agent_for_task(task)
                if not agent:
                    raise RuntimeError(
                        f"No suitable agent found for task {task.task_id}"
                    )

                task.agent_id = agent.agent_id
                task.status = TaskStatus.ASSIGNED

                # Create task message
                task_message = AgentMessage(
                    conversation_id=self.execution_id,
                    type=MessageType.TOOL_REQUEST,
                    content=f"Execute task: {task.description}",
                    metadata={
                        "task_id": task.task_id,
                        "workflow_id": (
                            self.workflow_def.workflow_id
                            if hasattr(self.workflow_def, "workflow_id")
                            else self.workflow_def.get("workflow_id")
                        ),
                        "input_data": task.input_data,
                        "is_workflow_task": True,
                    },
                    timeout_seconds=task.timeout_seconds,
                )

                # Execute task
                response = await asyncio.wait_for(
                    agent.process_message(task_message), timeout=task.timeout_seconds
                )

                # Process response
                if response.success:
                    task.status = TaskStatus.COMPLETED
                    task.result = response.content
                    task.output_data = response.context_updates
                    self.completed_tasks.add(task.task_id)

                    # Record task communication
                    self.task_communications.append(
                        {
                            "task_id": task.task_id,
                            "agent_id": agent.agent_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "input": task_message.content,
                            "output": response.content,
                            "metadata": response.metadata,
                        }
                    )

                    self.logger.info(f"Task {task.task_id} completed successfully")
                    break
                else:
                    raise RuntimeError(
                        response.error_message or "Task execution failed"
                    )

            except Exception as e:
                task.retry_count = attempt
                error_msg = (
                    f"Task {task.task_id} attempt {attempt + 1} failed: {str(e)}"
                )
                self.logger.warning(error_msg)

                if attempt < task.max_retries:
                    # Wait before retry with exponential backoff
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    # All retries exhausted
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    self.failed_tasks.add(task.task_id)

                    self.error_log.append(
                        {
                            "task_id": task.task_id,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                            "retry_count": attempt,
                        }
                    )

                    self.logger.error(
                        f"Task {task.task_id} failed after {attempt + 1} attempts"
                    )
                    break

        task.end_time = datetime.utcnow()

    async def _find_agent_for_task(self, task: WorkflowTask) -> Optional[AgentBase]:
        """Find the most suitable agent for a task."""
        # If specific agent is assigned, use it
        if task.agent_id and task.agent_id in self.agent_registry:
            return self.agent_registry[task.agent_id]

        # Find agents with required capabilities
        suitable_agents = []
        for agent_id, agent in self.agent_registry.items():
            if any(cap in agent.capabilities for cap in task.required_capabilities):
                suitable_agents.append(agent)

        if not suitable_agents:
            return None

        # Select agent with least current load (simple load balancing)
        # In practice, you'd integrate with the router's load balancing
        return suitable_agents[0]  # Simplified selection

    def _build_results(self) -> Dict[str, Any]:
        """Build comprehensive workflow results."""
        return {
            "execution_id": self.execution_id,
            "workflow_id": (
                self.workflow_def.workflow_id
                if hasattr(self.workflow_def, "workflow_id")
                else self.workflow_def.get("workflow_id")
            ),
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time": (
                (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time
                else None
            ),
            # Task results
            "task_results": {
                task_id: {
                    "status": task.status.value,
                    "result": task.result,
                    "output_data": task.output_data,
                    "execution_time": task.execution_time(),
                    "retry_count": task.retry_count,
                    "agent_id": task.agent_id,
                    "error": task.error,
                }
                for task_id, task in self.tasks.items()
            },
            # Summary statistics
            "summary": {
                "total_tasks": len(self.tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "success_rate": (
                    len(self.completed_tasks) / len(self.tasks) if self.tasks else 0.0
                ),
            },
            # Communication log
            "communications": self.task_communications,
            "errors": self.error_log,
            # Workflow data
            "workflow_results": self.workflow_results,
        }

    async def cancel(self) -> None:
        """Cancel workflow execution."""
        self.logger.info(f"Cancelling workflow execution {self.execution_id}")
        self.status = WorkflowStatus.CANCELLED

        # Cancel running tasks
        for task_future in self.running_tasks.values():
            task_future.cancel()

        # Update task statuses
        for task in self.tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.CANCELLED

        self._emit_state("cancelled", "workflow execution cancelled")

    def get_status(self) -> Dict[str, Any]:
        """Get current workflow execution status."""
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "progress": {
                "total_tasks": len(self.tasks),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "running": len(self.running_tasks),
                "pending": len(
                    [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
                ),
            },
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "current_time": datetime.utcnow().isoformat(),
            "running_tasks": list(self.running_tasks.keys()),
        }


class WorkflowOrchestrator:
    """
    Orchestrates complex multi-agent workflows.

    Provides high-level workflow management including:
    - Workflow definition and execution
    - Agent coordination and communication
    - Result aggregation and reporting
    - Error handling and recovery
    """

    def __init__(
        self,
        agent_registry: Dict[str, AgentBase],
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize workflow orchestrator.

        Args:
            agent_registry: Registry of available agents
            state_callback: Optional callback for state updates
        """
        self.agent_registry = agent_registry
        self.logger = logging.getLogger(__name__)
        self._state_callback = state_callback

        # Active workflows
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_history: List[Dict[str, Any]] = []

        # Orchestrator statistics
        self.stats = {
            "workflows_executed": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "total_tasks_executed": 0,
            "average_workflow_time": 0.0,
        }

    async def execute_workflow(
        self,
        workflow_def: Union[WorkflowDefinition, Dict[str, Any]],
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a workflow definition.

        Args:
            workflow_def: Workflow definition to execute
            conversation_id: Optional conversation context

        Returns:
            Workflow execution results
        """
        # Convert dict to WorkflowDefinition if needed
        if isinstance(workflow_def, dict):
            workflow_def = (
                WorkflowDefinition(**workflow_def)
                if hasattr(WorkflowDefinition, "workflow_id")
                else workflow_def
            )

        workflow = WorkflowExecution(
            workflow_def=workflow_def,
            agent_registry=self.agent_registry,
            state_callback=self._state_callback,
        )

        # Track active workflow
        self.active_workflows[workflow.execution_id] = workflow
        self.stats["workflows_executed"] += 1

        try:
            results = await workflow.execute()

            # Update statistics
            if workflow.status == WorkflowStatus.COMPLETED:
                self.stats["workflows_completed"] += 1
            elif workflow.status == WorkflowStatus.FAILED:
                self.stats["workflows_failed"] += 1

            self.stats["total_tasks_executed"] += len(workflow.tasks)

            # Update average execution time
            if workflow.start_time and workflow.end_time:
                execution_time = (
                    workflow.end_time - workflow.start_time
                ).total_seconds()
                current_avg = self.stats["average_workflow_time"]
                total_completed = (
                    self.stats["workflows_completed"] + self.stats["workflows_failed"]
                )
                self.stats["average_workflow_time"] = (
                    current_avg * (total_completed - 1) + execution_time
                ) / total_completed

            # Archive workflow
            self.workflow_history.append(
                {
                    "execution_id": workflow.execution_id,
                    "workflow_name": (
                        workflow_def.name
                        if hasattr(workflow_def, "name")
                        else workflow_def.get("name", "unnamed")
                    ),
                    "status": workflow.status.value,
                    "execution_time": (
                        execution_time if "execution_time" in locals() else None
                    ),
                    "task_count": len(workflow.tasks),
                    "completed_at": datetime.utcnow().isoformat(),
                }
            )

            # Keep history manageable
            if len(self.workflow_history) > 100:
                self.workflow_history = self.workflow_history[-80:]

            return results

        finally:
            # Remove from active workflows
            self.active_workflows.pop(workflow.execution_id, None)

    async def create_multi_step_workflow(
        self, user_request: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create and execute a multi-step workflow from a user request.

        This demonstrates the "get weather and save to file" type of workflows.

        Args:
            user_request: User's natural language request
            conversation_id: Optional conversation ID

        Returns:
            Workflow execution results
        """
        # Parse request and create workflow
        workflow_def = self._parse_user_request_to_workflow(user_request)

        if not workflow_def:
            return {
                "error": "Could not create workflow from request",
                "request": user_request,
            }

        return await self.execute_workflow(workflow_def, conversation_id)

    def _parse_user_request_to_workflow(
        self, user_request: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a user request into a workflow definition.

        This is a simplified parser - in practice you might use NLP or an LLM.
        """
        request_lower = user_request.lower()

        # Example: "get weather and save to file"
        if "weather" in request_lower and (
            "save" in request_lower or "file" in request_lower
        ):
            return {
                "workflow_id": str(uuid.uuid4()),
                "name": "Weather and Save Workflow",
                "description": f"Get weather information and save to file: {user_request}",
                "execution_mode": "pipeline",
                "tasks": [
                    {
                        "task_id": "get_weather",
                        "name": "Get Weather Information",
                        "description": "Retrieve current weather information",
                        "required_capabilities": ["weather_info"],
                        "input_data": {"query": user_request},
                        "depends_on": [],
                        "priority": "high",
                    },
                    {
                        "task_id": "save_to_file",
                        "name": "Save Weather to File",
                        "description": "Save weather information to a file",
                        "required_capabilities": ["file_operations"],
                        "depends_on": ["get_weather"],
                        "priority": "normal",
                        "metadata": {
                            "file_format": "text",
                            "filename_pattern": "weather_{timestamp}.txt",
                        },
                    },
                ],
            }

        # Example: "calculate expenses and create report"
        elif "calculate" in request_lower and (
            "report" in request_lower or "summary" in request_lower
        ):
            return {
                "workflow_id": str(uuid.uuid4()),
                "name": "Calculate and Report Workflow",
                "description": f"Calculate values and create report: {user_request}",
                "execution_mode": "sequential",
                "tasks": [
                    {
                        "task_id": "perform_calculations",
                        "name": "Perform Calculations",
                        "description": "Calculate requested values",
                        "required_capabilities": ["calculations"],
                        "input_data": {"query": user_request},
                        "depends_on": [],
                        "priority": "high",
                    },
                    {
                        "task_id": "create_report",
                        "name": "Create Report",
                        "description": "Create formatted report from calculations",
                        "required_capabilities": ["file_operations"],
                        "depends_on": ["perform_calculations"],
                        "priority": "normal",
                    },
                ],
            }

        # Add more workflow patterns as needed
        return None

    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active workflows."""
        return {
            execution_id: workflow.get_status()
            for execution_id, workflow in self.active_workflows.items()
        }

    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel an active workflow."""
        if execution_id in self.active_workflows:
            await self.active_workflows[execution_id].cancel()
            return True
        return False

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self.stats,
            "active_workflows": len(self.active_workflows),
            "workflow_history_count": len(self.workflow_history),
            "registered_agents": len(self.agent_registry),
        }

    async def cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        self.logger.info("Cleaning up workflow orchestrator")

        # Cancel all active workflows
        for workflow in self.active_workflows.values():
            await workflow.cancel()

        self.active_workflows.clear()
        self.workflow_history.clear()

        self.logger.info("Workflow orchestrator cleanup complete")
