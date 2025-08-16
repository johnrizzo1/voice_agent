"""
Base agent framework with LlamaIndex integration.

Provides the abstract base class and concrete implementations for agents
in the multi-agent system, with deep integration to LlamaIndex ReAct agents.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for development environments without pydantic
    BaseModel = object
    Field = lambda **kwargs: None

from .message import AgentMessage, AgentResponse, MessageType, MessageStatus

# LlamaIndex imports with fallback
try:
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.ollama import Ollama

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    ReActAgent = None
    FunctionTool = None
    Ollama = None
    LLAMAINDEX_AVAILABLE = False


class AgentCapability(str, Enum):
    """Capabilities that agents can have."""

    GENERAL_CHAT = "general_chat"
    TOOL_EXECUTION = "tool_execution"
    CODE_ANALYSIS = "code_analysis"
    FILE_OPERATIONS = "file_operations"
    WEB_SEARCH = "web_search"
    CALCULATIONS = "calculations"
    WEATHER_INFO = "weather_info"
    NEWS_INFO = "news_info"
    SYSTEM_INFO = "system_info"
    CONVERSATION_MEMORY = "conversation_memory"
    RAG_RETRIEVAL = "rag_retrieval"
    TASK_PLANNING = "task_planning"
    CALENDAR_MANAGEMENT = "calendar_management"


class AgentStatus(str, Enum):
    """Status of agent instances."""

    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentConfig(BaseModel if BaseModel != object else dict):
    """Configuration for individual agents."""

    if BaseModel != object:
        agent_id: str
        agent_type: str
        capabilities: List[AgentCapability] = Field(default_factory=list)
        tools: List[str] = Field(default_factory=list)
        system_prompt: Optional[str] = None
        max_concurrent_tasks: int = 3
        timeout_seconds: float = 30.0
        priority: int = Field(default=5, ge=1, le=10)
        metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentBase(ABC):
    """
    Abstract base class for all agents in the multi-agent system.

    Provides the core interface and integration with LlamaIndex ReAct agents,
    while allowing for specialized implementations based on specific use cases.
    """

    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        llm_config: Optional[Dict[str, Any]] = None,
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration
            llm_config: LLM configuration for LlamaIndex
            state_callback: Optional callback for state changes
        """
        self.agent_id = agent_id
        self.config = (
            config if hasattr(config, "agent_id") else type("Config", (), config)()
        )
        self.llm_config = llm_config or {}
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        self._state_callback = state_callback

        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.capabilities: Set[AgentCapability] = set(
            getattr(self.config, "capabilities", [])
        )
        self.active_tasks: Set[str] = set()
        self.total_messages_processed = 0
        self.last_activity_time = time.time()

        # LlamaIndex integration
        self.llm: Optional[Ollama] = None
        self.react_agent: Optional[ReActAgent] = None
        self.tools: List[FunctionTool] = []

        # Conversation context
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_window_size = getattr(self.config, "metadata", {}).get(
            "context_window", 4096
        )

        # Performance metrics
        self.response_times: List[float] = []
        self.success_count = 0
        self.error_count = 0

    def _emit_state(self, state: str, message: Optional[str] = None) -> None:
        """Emit state change via callback."""
        if self._state_callback:
            try:
                self._state_callback(self.agent_id, state, message)
            except Exception:
                self.logger.debug(
                    f"State callback error for {self.agent_id}", exc_info=True
                )

    async def initialize(self) -> None:
        """Initialize the agent and its LlamaIndex components."""
        if not LLAMAINDEX_AVAILABLE:
            self.logger.warning(f"LlamaIndex not available for agent {self.agent_id}")
            self.status = AgentStatus.ERROR
            self._emit_state("error", "LlamaIndex not available")
            return

        try:
            self.logger.info(f"Initializing agent {self.agent_id}")
            self._emit_state("initializing", "setting up LlamaIndex components")

            # Initialize LLM
            self.llm = Ollama(
                model=self.llm_config.get("model", "mistral:7b"),
                temperature=self.llm_config.get("temperature", 0.7),
                base_url=self.llm_config.get("base_url", "http://localhost:11434"),
                request_timeout=self.llm_config.get("request_timeout", 60.0),
            )

            # Initialize tools (will be populated by tool adapter)
            self.tools = []

            # Create ReAct agent
            await self._create_react_agent()

            # Run any agent-specific initialization
            await self._agent_specific_initialize()

            self.status = AgentStatus.READY
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            self._emit_state("ready", f"agent ready with {len(self.tools)} tools")

        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            self._emit_state("error", f"initialization failed: {e}")
            raise

    async def _create_react_agent(self) -> None:
        """Create the LlamaIndex ReAct agent."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")

        # Build system prompt
        system_prompt = await self._build_system_prompt()

        # Create ReAct agent with tools
        self.react_agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            system_prompt=system_prompt,
            max_iterations=10,
        )

        self.logger.debug(
            f"Created ReAct agent for {self.agent_id} with {len(self.tools)} tools"
        )

    async def _build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        base_prompt = f"""You are {self.agent_id}, a specialized AI agent with the following capabilities:
{', '.join([cap.value for cap in self.capabilities])}

Your role is to assist users by leveraging your specialized capabilities and available tools.
Always be helpful, accurate, and concise in your responses.

When using tools, explain what you're doing and why.
If you need to hand off to another agent, clearly explain the reason.
"""

        # Add agent-specific prompt if configured
        if hasattr(self.config, "system_prompt") and self.config.system_prompt:
            base_prompt += f"\n\nAdditional instructions:\n{self.config.system_prompt}"

        return base_prompt

    @abstractmethod
    async def _agent_specific_initialize(self) -> None:
        """Agent-specific initialization logic (to be implemented by subclasses)."""
        pass

    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Process an incoming message and generate a response.

        Args:
            message: The message to process

        Returns:
            Agent response with results and any routing decisions
        """
        if self.status != AgentStatus.READY:
            error_msg = f"Agent {self.agent_id} not ready (status: {self.status})"
            self.logger.warning(error_msg)
            return AgentResponse(
                request_id=message.id,
                agent_id=self.agent_id,
                success=False,
                error_message=error_msg,
            )

        # Check if we can handle more tasks
        max_tasks = getattr(self.config, "max_concurrent_tasks", 3)
        if len(self.active_tasks) >= max_tasks:
            return AgentResponse(
                request_id=message.id,
                agent_id=self.agent_id,
                success=False,
                error_message=f"Agent {self.agent_id} at capacity ({len(self.active_tasks)}/{max_tasks} tasks)",
            )

        start_time = time.time()
        self.active_tasks.add(message.id)
        self.status = AgentStatus.BUSY
        self._emit_state("active", f"processing message {message.id}")

        try:
            # Add to conversation history
            self._add_to_conversation_history(message)

            # Process with ReAct agent or fallback
            response = await self._process_with_agent(message)

            # Update metrics
            processing_time = time.time() - start_time
            response.processing_time_seconds = processing_time
            self.response_times.append(processing_time)
            self.success_count += 1
            self.total_messages_processed += 1
            self.last_activity_time = time.time()

            # Add response to conversation history
            self._add_response_to_conversation_history(response)

            return response

        except Exception as e:
            self.logger.error(f"Error processing message in agent {self.agent_id}: {e}")
            self.error_count += 1

            error_response = AgentResponse(
                request_id=message.id,
                agent_id=self.agent_id,
                success=False,
                error_message=str(e),
                processing_time_seconds=time.time() - start_time,
            )
            return error_response

        finally:
            self.active_tasks.discard(message.id)
            if len(self.active_tasks) == 0:
                self.status = AgentStatus.READY
                self._emit_state("ready", None)

    async def _process_with_agent(self, message: AgentMessage) -> AgentResponse:
        """Process message using LlamaIndex ReAct agent."""
        if not self.react_agent:
            # Fallback to direct processing
            return await self._fallback_process(message)

        try:
            # Convert message context for agent
            context = self._build_agent_context(message)

            # Query the ReAct agent
            agent_response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.react_agent.chat(message.content)
            )

            # Build response
            response = AgentResponse(
                request_id=message.id,
                agent_id=self.agent_id,
                content=str(agent_response),
                success=True,
            )

            # Check if we should suggest handoff
            await self._evaluate_handoff_need(message, response)

            return response

        except Exception as e:
            self.logger.error(f"ReAct agent error: {e}")
            return await self._fallback_process(message)

    async def _fallback_process(self, message: AgentMessage) -> AgentResponse:
        """Fallback processing when ReAct agent is not available."""
        response_content = f"I'm {self.agent_id}, but I'm currently unable to process your request properly. "
        response_content += "The multi-agent system is not fully operational."

        return AgentResponse(
            request_id=message.id,
            agent_id=self.agent_id,
            content=response_content,
            success=True,
            should_handoff=True,
            suggested_agent="general_agent",
            handoff_reason="fallback_mode",
        )

    def _build_agent_context(self, message: AgentMessage) -> str:
        """Build context string for the agent from conversation history."""
        context_parts = []

        # Add recent conversation history
        recent_history = self.conversation_history[-5:]  # Last 5 exchanges
        for entry in recent_history:
            if entry.get("type") == "user":
                context_parts.append(f"User: {entry.get('content', '')}")
            elif entry.get("type") == "agent":
                context_parts.append(f"Assistant: {entry.get('content', '')}")

        # Add current message context
        if message.context:
            context_parts.append(f"Context: {message.context}")

        return "\n".join(context_parts) if context_parts else ""

    async def _evaluate_handoff_need(
        self, message: AgentMessage, response: AgentResponse
    ) -> None:
        """Evaluate if this message should be handed off to another agent."""
        # Basic capability matching - can be overridden by subclasses
        if message.type == MessageType.TOOL_REQUEST:
            required_tools = message.metadata.get("required_tools", [])
            available_tools = {tool.metadata.get("name", "") for tool in self.tools}

            missing_tools = set(required_tools) - available_tools
            if missing_tools:
                response.set_handoff(
                    "tool_specialist",
                    f"requires tools not available: {', '.join(missing_tools)}",
                )

    def _add_to_conversation_history(self, message: AgentMessage) -> None:
        """Add message to conversation history."""
        entry = {
            "id": message.id,
            "timestamp": message.timestamp.isoformat(),
            "type": "user",
            "content": message.content,
            "metadata": message.metadata,
        }
        self.conversation_history.append(entry)

        # Trim history if too long
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-40:]

    def _add_response_to_conversation_history(self, response: AgentResponse) -> None:
        """Add response to conversation history."""
        entry = {
            "id": response.id,
            "timestamp": response.timestamp.isoformat(),
            "type": "agent",
            "agent_id": self.agent_id,
            "content": response.content,
            "metadata": response.metadata,
        }
        self.conversation_history.append(entry)

    def set_tools(self, tools: List[FunctionTool]) -> None:
        """Set the tools available to this agent."""
        self.tools = tools
        self.logger.info(f"Agent {self.agent_id} configured with {len(tools)} tools")

        # Recreate ReAct agent if it exists
        if self.react_agent and self.llm:
            asyncio.create_task(self._create_react_agent())

    def can_handle_capability(self, capability: AgentCapability) -> bool:
        """Check if this agent can handle a specific capability."""
        return capability in self.capabilities

    def get_status_info(self) -> Dict[str, Any]:
        """Get detailed status information about this agent."""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times
            else 0.0
        )

        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "active_tasks": len(self.active_tasks),
            "total_processed": self.total_messages_processed,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "avg_response_time": avg_response_time,
            "tools_count": len(self.tools),
            "last_activity": self.last_activity_time,
            "has_react_agent": self.react_agent is not None,
        }

    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        self.logger.info(f"Cleaning up agent {self.agent_id}")
        self.status = AgentStatus.SHUTDOWN

        # Cancel active tasks
        self.active_tasks.clear()

        # Cleanup LlamaIndex components
        self.react_agent = None
        self.llm = None
        self.tools.clear()

        # Clear conversation history
        self.conversation_history.clear()

        self._emit_state("shutdown", "agent cleanup complete")


class GeneralAgent(AgentBase):
    """
    General-purpose agent for handling common conversational tasks.

    This agent serves as the default handler for general chat and
    basic tool execution tasks.
    """

    def __init__(self, agent_id: str = "general_agent", **kwargs):
        """Initialize general agent with broad capabilities."""
        config = kwargs.get("config", {})
        if not hasattr(config, "capabilities"):
            config = type(
                "Config",
                (),
                {
                    "agent_id": agent_id,
                    "capabilities": [
                        AgentCapability.GENERAL_CHAT,
                        AgentCapability.TOOL_EXECUTION,
                        AgentCapability.CONVERSATION_MEMORY,
                    ],
                    "system_prompt": "You are a helpful general-purpose AI assistant.",
                    "max_concurrent_tasks": 5,
                    "timeout_seconds": 30.0,
                    "metadata": {"context_window": 4096},
                },
            )()

        super().__init__(
            agent_id=agent_id,
            config=config,
            llm_config=kwargs.get("llm_config"),
            state_callback=kwargs.get("state_callback"),
        )

    async def _agent_specific_initialize(self) -> None:
        """General agent specific initialization."""
        self.logger.info(f"General agent {self.agent_id} ready for broad task handling")


class ToolSpecialistAgent(AgentBase):
    """
    Specialized agent for complex tool execution and file operations.

    This agent is optimized for tasks that require multiple tool calls
    or complex file system operations.
    """

    def __init__(self, agent_id: str = "tool_specialist", **kwargs):
        """Initialize tool specialist agent."""
        config = kwargs.get("config", {})
        if not hasattr(config, "capabilities"):
            config = type(
                "Config",
                (),
                {
                    "agent_id": agent_id,
                    "capabilities": [
                        AgentCapability.TOOL_EXECUTION,
                        AgentCapability.FILE_OPERATIONS,
                        AgentCapability.SYSTEM_INFO,
                        AgentCapability.CALCULATIONS,
                    ],
                    "system_prompt": "You are a specialist in tool execution and file operations. Be precise and thorough.",
                    "max_concurrent_tasks": 3,
                    "timeout_seconds": 60.0,
                    "metadata": {"context_window": 2048},
                },
            )()

        super().__init__(
            agent_id=agent_id,
            config=config,
            llm_config=kwargs.get("llm_config"),
            state_callback=kwargs.get("state_callback"),
        )

    async def _agent_specific_initialize(self) -> None:
        """Tool specialist specific initialization."""
        self.logger.info(
            f"Tool specialist {self.agent_id} ready for complex tool operations"
        )


class InformationAgent(AgentBase):
    """
    Specialized agent for information retrieval and external data sources.

    This agent is optimized for:
    - Weather information queries
    - Web search and research tasks
    - News and current events (future)
    - Information synthesis from multiple sources
    - Context-aware responses for information queries
    """

    def __init__(self, agent_id: str = "information_agent", **kwargs):
        """Initialize information agent with information-focused capabilities."""
        config = kwargs.get("config", {})
        if not hasattr(config, "capabilities"):
            config = type(
                "Config",
                (),
                {
                    "agent_id": agent_id,
                    "capabilities": [
                        AgentCapability.WEATHER_INFO,
                        AgentCapability.WEB_SEARCH,
                        AgentCapability.NEWS_INFO,
                        AgentCapability.TOOL_EXECUTION,
                        AgentCapability.CONVERSATION_MEMORY,
                    ],
                    "system_prompt": (
                        "You are an information specialist focused on retrieving and presenting "
                        "accurate, up-to-date information from various sources. Your expertise includes:\n\n"
                        "- Weather information: Provide detailed, context-aware weather reports\n"
                        "- Web search: Find relevant information and synthesize results\n"
                        "- Information synthesis: Combine data from multiple sources coherently\n"
                        "- Current events: Stay informed about news and developments\n\n"
                        "Always cite your sources and provide context for the information you present. "
                        "When using tools, explain what information you're gathering and why it's relevant. "
                        "Present information in a clear, organized manner that directly addresses the user's needs."
                    ),
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

        # Information-specific caching and optimization
        self._weather_cache = {}
        self._search_cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL

    async def _agent_specific_initialize(self) -> None:
        """Information agent specific initialization."""
        self.logger.info(
            f"Information agent {self.agent_id} ready for information retrieval tasks"
        )

        # Set up information processing optimizations
        await self._setup_information_processing()

    async def _setup_information_processing(self) -> None:
        """Set up information processing optimizations."""
        # Configure for information-focused tasks
        self.logger.debug("Setting up information processing optimizations")

        # Future: Set up caching, result optimization, etc.

    async def _build_system_prompt(self) -> str:
        """Build enhanced system prompt for information tasks."""
        base_prompt = await super()._build_system_prompt()

        # Add information-specific instructions
        info_prompt = """

When processing information requests:

1. **Weather Queries**: Provide comprehensive weather information including current conditions,
   forecasts, and relevant context (e.g., clothing recommendations, travel advisories).

2. **Web Search**: Summarize search results clearly, highlighting the most relevant information.
   Always mention sources and recency of information when available.

3. **Information Synthesis**: When combining multiple sources, clearly indicate which information
   comes from which source and highlight any conflicting data.

4. **Response Format**: Structure responses with clear headings and bullet points when appropriate.
   Use emojis sparingly but effectively to enhance readability.

5. **Accuracy First**: If information is uncertain or sources conflict, clearly state this rather
   than presenting potentially incorrect information as fact.
"""

        return base_prompt + info_prompt

    async def _evaluate_handoff_need(
        self, message: AgentMessage, response: AgentResponse
    ) -> None:
        """Evaluate if this message should be handed off to another agent."""
        # Check if the request is outside our information domain
        content_lower = message.content.lower()

        # Don't handoff for information-related queries
        info_keywords = [
            "weather",
            "forecast",
            "temperature",
            "rain",
            "snow",
            "storm",
            "search",
            "find",
            "look up",
            "research",
            "information about",
            "news",
            "current events",
            "latest",
            "recent",
            "what happened",
        ]

        if any(keyword in content_lower for keyword in info_keywords):
            # This is clearly our domain, don't suggest handoff
            return

        # Check for calculation/math requests - handoff to tool specialist
        calc_keywords = ["calculate", "compute", "math", "equation", "formula"]
        if any(keyword in content_lower for keyword in calc_keywords):
            response.set_handoff(
                "tool_specialist",
                "mathematical calculations are better handled by the tool specialist",
            )
            return

        # Check for file operations - handoff to tool specialist
        file_keywords = [
            "file",
            "directory",
            "folder",
            "save",
            "load",
            "read file",
            "write file",
        ]
        if any(keyword in content_lower for keyword in file_keywords):
            response.set_handoff(
                "tool_specialist",
                "file operations are better handled by the tool specialist",
            )
            return

        # For general chat that doesn't need information retrieval, suggest general agent
        if not any(
            keyword in content_lower
            for keyword in info_keywords + ["weather", "search", "find"]
        ):
            response.set_handoff(
                "general_agent",
                "general conversation is better handled by the general agent",
            )

    def _format_weather_response(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data into a user-friendly response."""
        if not weather_data.get("success"):
            return f"Sorry, I couldn't retrieve weather information: {weather_data.get('error', 'Unknown error')}"

        result = weather_data.get("result", {})
        if not result:
            return "No weather data available."

        # Build formatted response
        location = result.get("location", "Unknown location")
        temp = result.get("temperature", "N/A")
        temp_unit = result.get("temperature_unit", "")
        condition = result.get(
            "condition", result.get("weather_description", "Unknown")
        )
        emoji = result.get("emoji", "🌤️")

        response_parts = [
            f"{emoji} **Weather for {location}**",
            f"Current: {temp}{temp_unit} - {condition}",
        ]

        # Add additional details if available
        if result.get("feels_like"):
            response_parts.append(f"Feels like: {result['feels_like']}{temp_unit}")

        if result.get("humidity"):
            response_parts.append(f"Humidity: {result['humidity']}%")

        if result.get("wind_speed"):
            wind_unit = result.get("wind_unit", "km/h")
            response_parts.append(f"Wind: {result['wind_speed']} {wind_unit}")

        # Add source attribution
        source = result.get("source", "Weather service")
        response_parts.append(f"\n*Source: {source}*")

        return "\n".join(response_parts)

    def _format_search_response(self, search_data: Dict[str, Any]) -> str:
        """Format web search results into a user-friendly response."""
        if not search_data.get("success"):
            return f"Sorry, I couldn't perform the web search: {search_data.get('error', 'Unknown error')}"

        result = search_data.get("result", {})
        if not result:
            return "No search results available."

        query = result.get("query", "your search")
        results = result.get("results", [])
        source = result.get("source", "Web search")

        if not results:
            return f"No results found for '{query}'."

        response_parts = [f"🔍 **Search results for: {query}**\n"]

        # Format top results
        for i, search_result in enumerate(results[:5], 1):
            title = search_result.get("title", "No title")
            snippet = search_result.get("snippet", "No description available")
            url = search_result.get("url", "")
            result_source = search_result.get("source", "")

            # Clean up snippet
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."

            response_parts.append(f"**{i}. {title}**")
            response_parts.append(f"{snippet}")
            if result_source:
                response_parts.append(f"*Source: {result_source}*")
            response_parts.append("")  # Add spacing

        response_parts.append(f"*Search provided by: {source}*")

        return "\n".join(response_parts)

    async def _process_with_agent(self, message: AgentMessage) -> AgentResponse:
        """Enhanced processing with information-specific formatting."""
        try:
            # Use parent processing
            response = await super()._process_with_agent(message)

            # Apply information-specific post-processing
            if response.success:
                response.content = await self._enhance_information_response(
                    message.content, response.content
                )

            return response

        except Exception as e:
            self.logger.error(f"Information agent processing error: {e}")
            return await self._fallback_process(message)

    async def _enhance_information_response(self, query: str, response: str) -> str:
        """Enhance response with information-specific formatting and context."""
        # This method can be extended to apply consistent formatting,
        # add relevant context, or optimize information presentation

        # For now, ensure proper formatting and add helpful context
        enhanced_response = response

        # Add context for weather queries
        if any(
            keyword in query.lower()
            for keyword in ["weather", "forecast", "temperature"]
        ):
            if "weather" in response.lower() and "°" in response:
                enhanced_response += "\n\n💡 *Tip: Weather conditions can change quickly. Check for updates if planning outdoor activities.*"

        # Add context for search queries
        if any(keyword in query.lower() for keyword in ["search", "find", "look up"]):
            if "search results" in response.lower() or "found" in response.lower():
                enhanced_response += "\n\n💡 *Tip: Information accuracy can vary by source. Cross-reference important details when needed.*"

        return enhanced_response
