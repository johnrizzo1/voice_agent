"""
ProductivityAgent for calendar and file operations.

This agent specializes in productivity and organization tasks, focusing on
file operations and calendar management (with future Google Calendar integration).
"""

import logging
from typing import Any, Dict, List, Optional, Set

from ..core.multi_agent.agent_base import (
    AgentBase,
    AgentCapability,
    AgentMessage,
    AgentResponse,
)


class ProductivityAgent(AgentBase):
    """
    Specialized agent for productivity and organization tasks.

    This agent is optimized for:
    - File system operations and document management
    - Calendar event management and scheduling
    - Task and reminder management (future expansion)
    - Note-taking and organization
    - Document processing and summarization
    - Productivity workflow optimization
    """

    def __init__(self, agent_id: str = "productivity_agent", **kwargs):
        """Initialize productivity agent with productivity-focused capabilities."""
        config = kwargs.pop(
            "config", {}
        )  # Remove config from kwargs to avoid duplicate
        if not hasattr(config, "capabilities"):
            config = type(
                "Config",
                (),
                {
                    "agent_id": agent_id,
                    "capabilities": [
                        AgentCapability.FILE_OPERATIONS,
                        AgentCapability.CALENDAR_MANAGEMENT,
                        AgentCapability.TOOL_EXECUTION,
                        AgentCapability.SYSTEM_INFO,
                        AgentCapability.CONVERSATION_MEMORY,
                        AgentCapability.TASK_PLANNING,
                    ],
                    "system_prompt": self._get_productivity_system_prompt(),
                    "max_concurrent_tasks": 4,
                    "timeout_seconds": 45.0,
                    "metadata": {"context_window": 3072},
                },
            )()

        super().__init__(
            agent_id=agent_id,
            config=config,
            llm_config=kwargs.get("llm_config"),
            state_callback=kwargs.get("state_callback"),
        )

        # Productivity-specific optimizations
        self._file_operation_context = {}
        self._calendar_context = {}
        self._productivity_metrics = {
            "files_processed": 0,
            "calendar_events_managed": 0,
            "tasks_organized": 0,
        }

    def _get_productivity_system_prompt(self) -> str:
        """Get the specialized system prompt for productivity tasks."""
        return """You are a ProductivityAgent, an AI assistant specialized in productivity and organization tasks. Your expertise includes:

**File Operations & Document Management:**
- Reading, writing, organizing, and managing files and directories
- Document processing, summarization, and content analysis
- File system navigation and organization recommendations
- Data extraction and file format conversions

**Calendar & Scheduling Management:**
- Creating, updating, and managing calendar events
- Scheduling optimization and conflict resolution
- Meeting coordination and availability checking
- Time management and productivity scheduling

**Task & Project Organization:**
- Breaking down complex tasks into manageable steps
- Project planning and workflow optimization
- Priority management and deadline tracking
- Productivity workflow recommendations

**Communication Style:**
- Be systematic and organized in your approach
- Provide clear, actionable steps for file and calendar operations
- Offer productivity tips and best practices when relevant
- Use structured formatting (lists, headers) for complex information
- Always explain what you're doing and why it's beneficial

**Tool Usage:**
- Use file_ops tool for all file system operations
- Use calendar tool for scheduling and event management
- Combine tools efficiently for complex productivity workflows
- Provide detailed feedback on operations performed

**Handoff Decisions:**
- Handle all file operations, document management, and calendar tasks
- Refer calculation-heavy tasks to tool_specialist
- Refer information retrieval (web search, weather, news) to information_agent
- Refer general conversation to general_agent when not productivity-related

Always prioritize user productivity and organization. When working with files or calendar, provide context about what you're doing and suggest improvements to their workflow when appropriate."""

    async def _agent_specific_initialize(self) -> None:
        """Productivity agent specific initialization."""
        self.logger.info(
            f"ProductivityAgent {self.agent_id} ready for productivity and organization tasks"
        )

        # Set up productivity processing optimizations
        await self._setup_productivity_processing()

    async def _setup_productivity_processing(self) -> None:
        """Set up productivity processing optimizations."""
        self.logger.debug("Setting up productivity processing optimizations")

        # Initialize context tracking for file operations
        self._file_operation_context = {
            "current_directory": None,
            "recent_files": [],
            "operation_history": [],
        }

        # Initialize context tracking for calendar operations
        self._calendar_context = {
            "recent_events": [],
            "preferred_meeting_times": [],
            "scheduling_preferences": {},
        }

        # Future: Set up integrations with productivity tools, file watchers, etc.

    async def _build_system_prompt(self) -> str:
        """Build enhanced system prompt for productivity tasks."""
        base_prompt = await super()._build_system_prompt()

        # Add productivity-specific instructions
        productivity_prompt = """

**Productivity Task Processing Guidelines:**

1. **File Operations Excellence:**
   - Always confirm file paths and operations before executing
   - Provide summaries of file contents when reading
   - Suggest organization improvements for file structures
   - Offer backup and safety recommendations for important operations
   - Use relative paths when possible for portability

2. **Calendar Management Best Practices:**
   - Confirm event details before creating/updating
   - Check for scheduling conflicts and suggest alternatives
   - Provide context about meeting preparation when relevant
   - Suggest optimal meeting times based on availability
   - Include location and attendee information clearly

3. **Workflow Optimization:**
   - Break complex tasks into clear, actionable steps
   - Suggest automation opportunities when relevant
   - Provide time estimates for file operations when possible
   - Recommend productivity tools and techniques
   - Track and report on completed productivity tasks

4. **Context Awareness:**
   - Remember recent file operations within the conversation
   - Build on previous calendar interactions for better scheduling
   - Maintain awareness of user's organizational preferences
   - Suggest related productivity improvements

5. **Error Handling & Safety:**
   - Always validate file paths and permissions
   - Confirm destructive operations before executing
   - Provide clear error messages with suggested solutions
   - Maintain data safety as the highest priority
"""

        return base_prompt + productivity_prompt

    async def _evaluate_handoff_need(
        self, message: AgentMessage, response: AgentResponse
    ) -> None:
        """Evaluate if this message should be handed off to another agent."""
        content_lower = message.content.lower()

        # Core productivity keywords - we handle these
        productivity_keywords = [
            "file",
            "folder",
            "directory",
            "document",
            "save",
            "load",
            "read",
            "write",
            "calendar",
            "schedule",
            "meeting",
            "appointment",
            "event",
            "reminder",
            "task",
            "organize",
            "productivity",
            "workflow",
            "project",
            "deadline",
            "note",
            "notes",
            "list",
            "todo",
            "manage",
            "plan",
        ]

        if any(keyword in content_lower for keyword in productivity_keywords):
            # This is clearly our domain, don't suggest handoff
            return

        # Check for calculation/math requests - handoff to tool specialist
        calc_keywords = [
            "calculate",
            "compute",
            "math",
            "equation",
            "formula",
            "sum",
            "multiply",
            "divide",
        ]
        if any(keyword in content_lower for keyword in calc_keywords):
            response.set_handoff(
                "tool_specialist",
                "mathematical calculations are better handled by the tool specialist",
            )
            return

        # Check for information requests - handoff to information agent
        info_keywords = [
            "weather",
            "forecast",
            "search",
            "find information",
            "look up",
            "research",
            "news",
            "current events",
            "what happened",
        ]
        if any(keyword in content_lower for keyword in info_keywords):
            response.set_handoff(
                "information_agent",
                "information retrieval is better handled by the information agent",
            )
            return

        # For general chat that doesn't involve productivity, suggest general agent
        general_indicators = [
            "hello",
            "hi",
            "how are you",
            "thanks",
            "explain",
            "what is",
        ]
        if any(
            indicator in content_lower for indicator in general_indicators
        ) and not any(keyword in content_lower for keyword in productivity_keywords):
            response.set_handoff(
                "general_agent",
                "general conversation is better handled by the general agent",
            )

    async def _process_with_agent(self, message: AgentMessage) -> AgentResponse:
        """Enhanced processing with productivity-specific context and formatting."""
        try:
            # Update productivity context before processing
            self._update_productivity_context(message)

            # Use parent processing
            response = await super()._process_with_agent(message)

            # Apply productivity-specific post-processing
            if response.success:
                response.content = await self._enhance_productivity_response(
                    message.content, response.content
                )

                # Update metrics
                self._update_productivity_metrics(message, response)

            return response

        except Exception as e:
            self.logger.error(f"ProductivityAgent processing error: {e}")
            return await self._fallback_process(message)

    def _update_productivity_context(self, message: AgentMessage) -> None:
        """Update productivity context based on the message."""
        content_lower = message.content.lower()

        # Track file operation context
        if any(
            keyword in content_lower
            for keyword in ["file", "folder", "directory", "save", "load"]
        ):
            self._file_operation_context["operation_history"].append(
                {
                    "message_id": message.id,
                    "content": message.content,
                    "timestamp": message.timestamp,
                }
            )

            # Keep only recent operations
            if len(self._file_operation_context["operation_history"]) > 10:
                self._file_operation_context["operation_history"] = (
                    self._file_operation_context["operation_history"][-5:]
                )

        # Track calendar operation context
        if any(
            keyword in content_lower
            for keyword in ["calendar", "schedule", "meeting", "event"]
        ):
            self._calendar_context["recent_events"].append(
                {
                    "message_id": message.id,
                    "content": message.content,
                    "timestamp": message.timestamp,
                }
            )

            # Keep only recent events
            if len(self._calendar_context["recent_events"]) > 10:
                self._calendar_context["recent_events"] = self._calendar_context[
                    "recent_events"
                ][-5:]

    async def _enhance_productivity_response(self, query: str, response: str) -> str:
        """Enhance response with productivity-specific formatting and tips."""
        enhanced_response = response
        query_lower = query.lower()

        # Add context for file operations
        if any(
            keyword in query_lower
            for keyword in ["file", "folder", "directory", "save", "load"]
        ):
            if "file" in response.lower() and (
                "created" in response.lower() or "saved" in response.lower()
            ):
                enhanced_response += "\n\n💡 *Productivity Tip: Consider organizing files with descriptive names and consistent folder structures for better long-term management.*"
            elif "read" in response.lower() or "content" in response.lower():
                enhanced_response += "\n\n💡 *Productivity Tip: For large documents, I can help summarize key points or extract specific information to save you time.*"

        # Add context for calendar operations
        if any(
            keyword in query_lower
            for keyword in ["calendar", "schedule", "meeting", "event"]
        ):
            if "event" in response.lower() and (
                "created" in response.lower() or "scheduled" in response.lower()
            ):
                enhanced_response += "\n\n💡 *Productivity Tip: Consider blocking prep time before important meetings and adding location/dial-in details to save time later.*"
            elif "busy" in response.lower() or "availability" in response.lower():
                enhanced_response += "\n\n💡 *Productivity Tip: I can help find optimal meeting times that work for all attendees and suggest time blocks for focused work.*"

        # Add context for task management
        if any(
            keyword in query_lower for keyword in ["task", "todo", "organize", "plan"]
        ):
            enhanced_response += "\n\n💡 *Productivity Tip: Break large tasks into smaller, actionable items and set deadlines to maintain momentum.*"

        return enhanced_response

    def _update_productivity_metrics(
        self, message: AgentMessage, response: AgentResponse
    ) -> None:
        """Update productivity metrics based on the interaction."""
        content_lower = message.content.lower()

        # Count file operations
        if any(
            keyword in content_lower
            for keyword in [
                "file",
                "folder",
                "directory",
                "save",
                "load",
                "read",
                "write",
            ]
        ):
            self._productivity_metrics["files_processed"] += 1

        # Count calendar operations
        if any(
            keyword in content_lower
            for keyword in ["calendar", "schedule", "meeting", "event"]
        ):
            self._productivity_metrics["calendar_events_managed"] += 1

        # Count task organization
        if any(
            keyword in content_lower
            for keyword in ["task", "organize", "plan", "todo", "workflow"]
        ):
            self._productivity_metrics["tasks_organized"] += 1

    def get_productivity_stats(self) -> Dict[str, Any]:
        """Get productivity-specific statistics."""
        base_stats = self.get_status_info()
        base_stats.update(
            {
                "productivity_metrics": self._productivity_metrics,
                "file_context": {
                    "operations_count": len(
                        self._file_operation_context.get("operation_history", [])
                    ),
                    "current_directory": self._file_operation_context.get(
                        "current_directory"
                    ),
                },
                "calendar_context": {
                    "events_count": len(self._calendar_context.get("recent_events", []))
                },
            }
        )
        return base_stats

    async def cleanup(self) -> None:
        """Cleanup productivity agent resources."""
        self.logger.info(f"Cleaning up ProductivityAgent {self.agent_id}")

        # Clear productivity context
        self._file_operation_context.clear()
        self._calendar_context.clear()
        self._productivity_metrics.clear()

        # Call parent cleanup
        await super().cleanup()
