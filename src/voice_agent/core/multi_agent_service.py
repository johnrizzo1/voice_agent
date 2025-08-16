"""
Multi-agent service integration.

Main service class that orchestrates the multi-agent system, providing
backward compatibility with the existing voice agent while adding
multi-agent capabilities when enabled.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable

from .config import Config, MultiAgentConfig
from .llamaindex_service import LlamaIndexService
from .tool_executor import ToolExecutor

# Multi-agent imports with fallback
try:
    from .multi_agent import (
        AgentMessage,
        AgentResponse,
        MessageType,
        MessageStatus,
        AgentBase,
        GeneralAgent,
        ToolSpecialistAgent,
        InformationAgent,
        AgentRouter,
        RoutingStrategy,
        SharedContextManager,
        ToolAdapter,
    )
    from .multi_agent.workflow import WorkflowOrchestrator, WorkflowDefinition
    from .multi_agent.communication import CommunicationHub, EnhancedDelegationManager
    from ..agents import ProductivityAgent, UtilityAgent

    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False
    AgentMessage = None
    AgentResponse = None
    MessageType = None
    MessageStatus = None
    AgentBase = None
    GeneralAgent = None
    ToolSpecialistAgent = None
    InformationAgent = None
    ProductivityAgent = None
    UtilityAgent = None
    AgentRouter = None
    RoutingStrategy = None
    SharedContextManager = None
    ToolAdapter = None
    WorkflowOrchestrator = None
    WorkflowDefinition = None
    CommunicationHub = None
    EnhancedDelegationManager = None


class MultiAgentService:
    """
    Main multi-agent service that orchestrates the entire multi-agent system.

    Provides:
    - Backward compatibility with existing single-agent mode
    - Multi-agent routing and coordination when enabled
    - Seamless integration with existing voice agent components
    - Feature flag support for gradual rollout
    """

    def __init__(
        self,
        config: Config,
        tool_executor: ToolExecutor,
        llamaindex_service: Optional[LlamaIndexService] = None,
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize the multi-agent service.

        Args:
            config: Main configuration object
            tool_executor: Existing tool executor instance
            llamaindex_service: Optional LlamaIndex service for single-agent mode
            state_callback: Optional callback for state changes
        """
        self.config = config
        self.multi_agent_config = config.multi_agent
        self.tool_executor = tool_executor
        self.llamaindex_service = llamaindex_service
        self.logger = logging.getLogger(__name__)
        self._state_callback = state_callback

        # Service state
        self.is_initialized = False
        self.multi_agent_enabled = (
            MULTI_AGENT_AVAILABLE and self.multi_agent_config.enabled
        )

        # Multi-agent components (None if disabled)
        self.agent_router: Optional[AgentRouter] = None
        self.context_manager: Optional[SharedContextManager] = None
        self.tool_adapter: Optional[ToolAdapter] = None

        # Enhanced components
        self.workflow_orchestrator: Optional[WorkflowOrchestrator] = None
        self.communication_hub: Optional[CommunicationHub] = None
        self.delegation_manager: Optional[EnhancedDelegationManager] = None

        # Active agents
        self.agents: Dict[str, AgentBase] = {}

        # Performance tracking
        self.message_count = 0
        self.agent_switches = 0
        self.routing_stats: Dict[str, int] = {}
        self.workflow_count = 0
        self.collaboration_count = 0

    def _emit_state(self, state: str, message: Optional[str] = None) -> None:
        """Emit state change via callback."""
        if self._state_callback:
            try:
                self._state_callback("multi_agent_service", state, message)
            except Exception:
                self.logger.debug(
                    "Multi-agent service state callback error", exc_info=True
                )

    async def initialize(self) -> None:
        """Initialize the multi-agent service."""
        self.logger.info("Initializing multi-agent service")
        self._emit_state("initializing", "setting up multi-agent system")

        if not self.multi_agent_enabled:
            self.logger.info("Multi-agent system disabled - using single-agent mode")
            if self.llamaindex_service and not self.llamaindex_service.is_initialized:
                await self.llamaindex_service.initialize()
            self._emit_state("ready", "single-agent mode active")
            self.is_initialized = True
            return

        if not MULTI_AGENT_AVAILABLE:
            self.logger.error(
                "Multi-agent system requested but not available - falling back to single-agent"
            )
            self.multi_agent_enabled = False
            if self.llamaindex_service and not self.llamaindex_service.is_initialized:
                await self.llamaindex_service.initialize()
            self._emit_state("ready", "fallback to single-agent mode")
            self.is_initialized = True
            return

        try:
            # Initialize core multi-agent components
            await self._initialize_multi_agent_components()

            # Initialize agents
            await self._initialize_agents()

            # Setup routing rules
            await self._setup_routing_rules()

            self.logger.info(
                f"Multi-agent service initialized with {len(self.agents)} agents"
            )
            self._emit_state(
                "ready", f"multi-agent mode with {len(self.agents)} agents"
            )
            self.is_initialized = True

        except Exception as e:
            self.logger.error(f"Failed to initialize multi-agent service: {e}")
            self.logger.info("Falling back to single-agent mode")

            # Fallback to single-agent mode
            self.multi_agent_enabled = False
            if self.llamaindex_service and not self.llamaindex_service.is_initialized:
                await self.llamaindex_service.initialize()

            self._emit_state("ready", "fallback to single-agent mode after error")
            self.is_initialized = True

    async def _initialize_multi_agent_components(self) -> None:
        """Initialize core multi-agent components."""
        # Initialize tool adapter
        self.tool_adapter = ToolAdapter(self.tool_executor)
        await self.tool_adapter.initialize()

        # Initialize enhanced context manager with better preservation
        self.context_manager = SharedContextManager(
            max_conversations=100,
            default_context_window=self.multi_agent_config.context_window_size,
            enable_rag_integration=True,  # Enable for better context sharing
            max_slice_age_hours=48,  # Keep context longer for complex workflows
        )

        # Initialize communication hub
        if MULTI_AGENT_AVAILABLE and CommunicationHub:
            self.communication_hub = CommunicationHub(
                agent_registry={},  # Will be populated as agents are registered
                state_callback=self._state_callback,
            )

        # Initialize router
        routing_strategy = RoutingStrategy(self.multi_agent_config.routing_strategy)
        self.agent_router = AgentRouter(
            default_agent=self.multi_agent_config.default_agent,
            strategy=routing_strategy,
            embedding_model=self.multi_agent_config.embedding_model,
            confidence_threshold=self.multi_agent_config.confidence_threshold,
            state_callback=self._state_callback,
        )
        await self.agent_router.initialize()

        # Initialize workflow orchestrator
        if MULTI_AGENT_AVAILABLE and WorkflowOrchestrator:
            self.workflow_orchestrator = WorkflowOrchestrator(
                agent_registry={},  # Will be populated as agents are registered
                state_callback=self._state_callback,
            )

        # Initialize enhanced delegation manager
        if (
            MULTI_AGENT_AVAILABLE
            and EnhancedDelegationManager
            and self.communication_hub
        ):
            self.delegation_manager = EnhancedDelegationManager(
                agent_registry={},  # Will be populated as agents are registered
                communication_hub=self.communication_hub,
                state_callback=self._state_callback,
            )

    async def _initialize_agents(self) -> None:
        """Initialize configured agents."""
        agent_configs = self.multi_agent_config.agents

        for agent_id, agent_config in agent_configs.items():
            try:
                agent_type = agent_config.get("type", "GeneralAgent")

                # Create agent configuration
                agent_config_obj = self._create_agent_config(agent_id, agent_config)
                llm_config = self._get_llm_config()

                # Create agent based on type
                if agent_type == "GeneralAgent":
                    agent = GeneralAgent(
                        agent_id=agent_id,
                        config=agent_config_obj,
                        llm_config=llm_config,
                        state_callback=self._state_callback,
                    )
                elif agent_type == "ToolSpecialistAgent":
                    agent = ToolSpecialistAgent(
                        agent_id=agent_id,
                        config=agent_config_obj,
                        llm_config=llm_config,
                        state_callback=self._state_callback,
                    )
                elif agent_type == "InformationAgent":
                    agent = InformationAgent(
                        agent_id=agent_id,
                        config=agent_config_obj,
                        llm_config=llm_config,
                        state_callback=self._state_callback,
                    )
                elif agent_type == "ProductivityAgent":
                    agent = ProductivityAgent(
                        agent_id=agent_id,
                        config=agent_config_obj,
                        llm_config=llm_config,
                        state_callback=self._state_callback,
                    )
                elif agent_type == "UtilityAgent":
                    agent = UtilityAgent(
                        agent_id=agent_id,
                        config=agent_config_obj,
                        llm_config=llm_config,
                        state_callback=self._state_callback,
                    )
                else:
                    self.logger.warning(
                        f"Unknown agent type {agent_type}, using GeneralAgent"
                    )
                    agent = GeneralAgent(
                        agent_id=agent_id,
                        config=agent_config_obj,
                        llm_config=llm_config,
                        state_callback=self._state_callback,
                    )

                # Initialize agent
                await agent.initialize()

                # Set up tools for agent
                agent_tools = await self._get_tools_for_agent(agent_id, agent_config)
                if agent_tools:
                    agent.set_tools(agent_tools)
                    self.logger.info(
                        f"Configured {len(agent_tools)} tools for agent {agent_id}"
                    )
                else:
                    # If no tools are available, make sure we still log this properly
                    self.logger.warning(f"No tools configured for agent {agent_id}")

                # Register with router
                self.agent_router.register_agent(agent)

                # Register with enhanced components
                if self.communication_hub:
                    self.communication_hub.agent_registry[agent_id] = agent
                    # Subscribe to relevant channels
                    from .multi_agent.communication import CommunicationChannel

                    self.communication_hub.subscribe_to_channel(
                        agent_id, CommunicationChannel.COORDINATION
                    )
                    self.communication_hub.subscribe_to_channel(
                        agent_id, CommunicationChannel.STATUS_UPDATES
                    )

                if self.workflow_orchestrator:
                    self.workflow_orchestrator.agent_registry[agent_id] = agent

                if self.delegation_manager:
                    self.delegation_manager.agent_registry[agent_id] = agent

                # Store agent
                self.agents[agent_id] = agent

                self.logger.info(
                    f"Initialized agent {agent_id} ({agent_type}) with {len(agent_tools)} tools"
                )

            except Exception as e:
                self.logger.error(f"Failed to initialize agent {agent_id}: {e}")
                # Continue with other agents

    def _create_agent_config(self, agent_id: str, agent_config: Dict[str, Any]):
        """Create agent configuration object."""
        from .multi_agent.agent_base import AgentCapability

        # Convert capability strings to AgentCapability enums
        capabilities = []
        for cap_str in agent_config.get("capabilities", []):
            try:
                capability = AgentCapability(cap_str)
                capabilities.append(capability)
            except ValueError:
                self.logger.warning(
                    f"Unknown capability '{cap_str}' for agent {agent_id}"
                )

        # This is a simple dict-based config since we're avoiding pydantic import issues
        return type(
            "AgentConfig",
            (),
            {
                "agent_id": agent_id,
                "agent_type": agent_config.get("type", "GeneralAgent"),
                "capabilities": capabilities,
                "tools": agent_config.get("tools", []),
                "system_prompt": agent_config.get("system_prompt"),
                "max_concurrent_tasks": agent_config.get("max_concurrent_tasks", 3),
                "timeout_seconds": agent_config.get("timeout_seconds", 30.0),
                "metadata": {
                    "context_window": self.multi_agent_config.context_window_size
                },
            },
        )()

    def _get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for agents."""
        return {
            "model": self.config.llm.model,
            "temperature": self.config.llm.temperature,
            "base_url": "http://localhost:11434",
            "request_timeout": 60.0,
        }

    async def _get_tools_for_agent(
        self, agent_id: str, agent_config: Dict[str, Any]
    ) -> List:
        """Get tools configured for a specific agent."""
        if not self.tool_adapter:
            return []

        # Get all available tools
        all_tools = await self.tool_adapter.get_llamaindex_tools(agent_id)

        # Filter based on agent configuration
        agent_tool_names = agent_config.get("tools", [])
        agent_capabilities = agent_config.get("capabilities", [])

        if agent_tool_names:
            # Filter by specific tool names
            filtered_tools = [
                tool
                for tool in all_tools
                if tool.metadata.get("name", "") in agent_tool_names
            ]
        else:
            # Filter by capabilities
            filtered_tools = self.tool_adapter.filter_tools_for_agent(
                all_tools, agent_capabilities
            )

        return filtered_tools

    async def _setup_routing_rules(self) -> None:
        """Setup routing rules from configuration."""
        if not self.agent_router:
            return

        from .multi_agent.router import RoutingRule, AgentCapability

        routing_rules_config = self.multi_agent_config.routing_rules

        for rule_config in routing_rules_config:
            try:
                # Convert capability strings to enums
                capabilities = []
                for cap_str in rule_config.get("capabilities", []):
                    try:
                        capability = AgentCapability(cap_str)
                        capabilities.append(capability)
                    except ValueError:
                        self.logger.warning(f"Unknown capability: {cap_str}")

                rule = RoutingRule(
                    name=rule_config["name"],
                    target_agent=rule_config["target_agent"],
                    patterns=rule_config["patterns"],
                    capabilities=capabilities,
                    priority=rule_config.get("priority", 5),
                    confidence=rule_config.get("confidence", 0.8),
                )

                self.agent_router.add_routing_rule(rule)

            except Exception as e:
                self.logger.error(
                    f"Failed to create routing rule {rule_config.get('name', 'unknown')}: {e}"
                )

    async def process_message(
        self, user_input: str, conversation_id: Optional[str] = None
    ) -> str:
        """
        Process a user message through the multi-agent system.

        Args:
            user_input: User's input message
            conversation_id: Optional conversation identifier

        Returns:
            Response string
        """
        self.message_count += 1

        if not self.is_initialized:
            await self.initialize()

        # Single-agent fallback
        if not self.multi_agent_enabled:
            return await self._process_single_agent(user_input)

        try:
            conversation_id = conversation_id or f"conv_{self.message_count}"

            # Create agent message
            message = AgentMessage(
                conversation_id=conversation_id,
                type=MessageType.USER_INPUT,
                content=user_input,
                requires_response=True,
            )

            # Add to context
            if self.context_manager:
                self.context_manager.add_message_to_context(conversation_id, message)

            # Route message to appropriate agent
            route_decision = await self.agent_router.route_message(message)
            target_agent_id = (
                route_decision.target_agent
                if hasattr(route_decision, "target_agent")
                else route_decision["target_agent"]
            )

            # Update routing stats
            if target_agent_id not in self.routing_stats:
                self.routing_stats[target_agent_id] = 0
            self.routing_stats[target_agent_id] += 1

            # Get target agent
            target_agent = self.agents.get(target_agent_id)
            if not target_agent:
                self.logger.error(f"Target agent {target_agent_id} not found")
                return await self._process_single_agent(user_input)

            # Process message with agent
            response = await target_agent.process_message(message)

            # Add response to context
            if self.context_manager:
                self.context_manager.add_response_to_context(conversation_id, response)

            # Handle agent handoff if requested
            if response.should_handoff and response.suggested_agent:
                self.agent_switches += 1
                return await self._handle_agent_handoff(
                    message, response, conversation_id
                )

            # Update agent load tracking
            self.agent_router.update_agent_load(target_agent_id, -1)

            return response.content

        except Exception as e:
            self.logger.error(f"Error in multi-agent processing: {e}")
            # Fallback to single-agent mode for this message
            return await self._process_single_agent(user_input)

    async def process_workflow(
        self, workflow_request: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a request that requires workflow orchestration.

        Args:
            workflow_request: User request that needs multi-step coordination
            conversation_id: Optional conversation identifier

        Returns:
            Workflow execution results
        """
        if not self.multi_agent_enabled or not self.workflow_orchestrator:
            return {"error": "Workflow orchestration not available"}

        try:
            # Create and execute workflow
            results = await self.workflow_orchestrator.create_multi_step_workflow(
                workflow_request, conversation_id
            )

            self.workflow_count += 1
            return results

        except Exception as e:
            self.logger.error(f"Workflow processing failed: {e}")
            return {"error": f"Workflow processing failed: {str(e)}"}

    async def request_agent_collaboration(
        self,
        initiator_agent: str,
        target_agents: List[str],
        task_description: str,
        collaboration_type: str = "problem_solving",
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Request collaboration between multiple agents.

        Args:
            initiator_agent: Agent requesting collaboration
            target_agents: List of agents to collaborate with
            task_description: Description of collaborative task
            collaboration_type: Type of collaboration needed
            context_data: Shared context for collaboration

        Returns:
            Collaboration session information
        """
        if not self.multi_agent_enabled or not self.communication_hub:
            return {"error": "Agent collaboration not available"}

        try:
            session_id = await self.communication_hub.request_collaboration(
                initiator_agent=initiator_agent,
                target_agents=target_agents,
                task_description=task_description,
                collaboration_type=collaboration_type,
                context_data=context_data,
            )

            self.collaboration_count += 1

            return {
                "success": True,
                "session_id": session_id,
                "participants": [initiator_agent] + target_agents,
                "collaboration_type": collaboration_type,
            }

        except Exception as e:
            self.logger.error(f"Collaboration request failed: {e}")
            return {"error": f"Collaboration request failed: {str(e)}"}

    async def delegate_complex_task(
        self,
        task_description: str,
        required_capabilities: List[str],
        delegation_pattern: str = "capability_based",
        priority: str = "normal",
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Delegate a complex task using enhanced delegation patterns.

        Args:
            task_description: Description of the task
            required_capabilities: Required agent capabilities
            delegation_pattern: Delegation pattern to use
            priority: Task priority level
            context_data: Additional context data

        Returns:
            Delegation results
        """
        if not self.multi_agent_enabled or not self.delegation_manager:
            return {"error": "Enhanced delegation not available"}

        try:
            # Convert capability strings to enums
            from .multi_agent.agent_base import AgentCapability
            from .multi_agent.communication import MessagePriority

            capabilities = []
            for cap_str in required_capabilities:
                try:
                    capability = AgentCapability(cap_str)
                    capabilities.append(capability)
                except ValueError:
                    self.logger.warning(f"Unknown capability: {cap_str}")

            # Convert priority string to enum
            try:
                priority_enum = MessagePriority(priority)
            except ValueError:
                priority_enum = MessagePriority.NORMAL

            # Delegate the task
            results = await self.delegation_manager.delegate_task(
                task_description=task_description,
                required_capabilities=capabilities,
                delegation_pattern=delegation_pattern,
                priority=priority_enum,
                context_data=context_data,
            )

            return results

        except Exception as e:
            self.logger.error(f"Task delegation failed: {e}")
            return {"error": f"Task delegation failed: {str(e)}"}

    async def _process_single_agent(self, user_input: str) -> str:
        """Process message using single-agent mode (fallback)."""
        if self.llamaindex_service and self.llamaindex_service.is_available:
            return await self.llamaindex_service.chat(user_input)
        else:
            return "I'm sorry, but I'm currently unable to process your request. The multi-agent system is not available and no fallback service is configured."

    async def _handle_agent_handoff(
        self,
        original_message: AgentMessage,
        response: AgentResponse,
        conversation_id: str,
    ) -> str:
        """Handle agent handoff scenario with enhanced context preservation."""
        try:
            suggested_agent_id = response.suggested_agent
            suggested_agent = self.agents.get(suggested_agent_id)

            if not suggested_agent:
                self.logger.warning(
                    f"Suggested agent {suggested_agent_id} not found for handoff"
                )
                return response.content

            # Enhanced context sharing with better preservation
            context_shared = False
            if self.context_manager:
                context_shared = self.context_manager.share_context_between_agents(
                    response.agent_id, suggested_agent_id, conversation_id
                )

                # Create enriched context slice for handoff
                handoff_context = self.context_manager.create_context_slice_for_agent(
                    suggested_agent_id,
                    conversation_id,
                    "handoff",
                    max_messages=15,  # More context for handoffs
                )

            # Notify via communication hub if available
            if self.communication_hub:
                try:
                    from .multi_agent.communication import CommunicationChannel

                    await self.communication_hub.send_message(
                        from_agent=response.agent_id,
                        to_agent=suggested_agent_id,
                        content=f"Handoff: {response.handoff_reason}",
                        channel=CommunicationChannel.COORDINATION,
                        subject="Agent Handoff",
                        data_payload={
                            "original_message": original_message.content,
                            "handoff_reason": response.handoff_reason,
                            "conversation_id": conversation_id,
                            "context_preserved": context_shared,
                        },
                    )
                except Exception as comm_error:
                    self.logger.debug(
                        f"Communication hub notification failed: {comm_error}"
                    )

            # Create enhanced handoff message with preserved context
            handoff_message = AgentMessage(
                conversation_id=conversation_id,
                type=MessageType.AGENT_HANDOFF,
                content=original_message.content,
                from_agent=response.agent_id,
                to_agent=suggested_agent_id,
                metadata={
                    "handoff_reason": response.handoff_reason,
                    "original_response": response.content,
                    "context_preserved": context_shared,
                    "handoff_timestamp": "placeholder_timestamp",
                    "conversation_history_available": bool(self.context_manager),
                    "previous_agent_capabilities": (
                        list(self.agents[response.agent_id].capabilities)
                        if response.agent_id in self.agents
                        else []
                    ),
                },
            )

            # Add conversation context to handoff message if available
            if self.context_manager and context_shared:
                recent_context = self.context_manager.get_conversation_context(
                    conversation_id
                )
                if recent_context:
                    handoff_message.context = {
                        "recent_messages": recent_context.get_recent_messages(5),
                        "conversation_summary": recent_context.get_context_text(
                            include_system=True
                        )[
                            -1000:
                        ],  # Last 1000 chars
                    }

            # Process with suggested agent
            handoff_response = await suggested_agent.process_message(handoff_message)

            # Update load tracking
            self.agent_router.update_agent_load(response.agent_id, -1)
            self.agent_router.update_agent_load(suggested_agent_id, -1)

            # Update delegation manager performance if available
            if self.delegation_manager:
                self.delegation_manager.update_agent_performance(
                    response.agent_id,
                    task_success=True,  # Successful handoff
                    response_time=1.0,  # Placeholder
                    quality_score=0.8,  # Handoff quality score
                )

            # Create comprehensive combined response
            handoff_info = f"\n\n🔄 **Task transferred from {response.agent_id} to {suggested_agent_id}**"
            if response.handoff_reason:
                handoff_info += f"\n*Reason: {response.handoff_reason}*"
            if context_shared:
                handoff_info += f"\n*Context preserved: ✅*"

            combined_response = (
                f"{response.content}{handoff_info}\n\n{handoff_response.content}"
            )

            return combined_response

        except Exception as e:
            self.logger.error(f"Error in agent handoff: {e}")
            return response.content

    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the enhanced multi-agent service."""
        base_info = {
            "multi_agent_enabled": self.multi_agent_enabled,
            "multi_agent_available": MULTI_AGENT_AVAILABLE,
            "is_initialized": self.is_initialized,
            "service_initialized": self.is_initialized,  # Add explicit service_initialized flag
            "message_count": self.message_count,
            "agent_switches": self.agent_switches,
            "workflow_count": self.workflow_count,
            "collaboration_count": self.collaboration_count,
            "routing_stats": self.routing_stats.copy(),
        }

        if self.multi_agent_enabled and self.agent_router:
            base_info.update(
                {
                    "active_agents": len(self.agents),
                    "router_stats": self.agent_router.get_routing_stats(),
                    "agent_status": {
                        agent_id: agent.get_status_info()
                        for agent_id, agent in self.agents.items()
                        if hasattr(agent, "get_status_info")
                    },
                }
            )

            # Enhanced context manager stats
            if self.context_manager:
                base_info["context_stats"] = self.context_manager.get_context_stats()

            # Tool adapter stats
            if self.tool_adapter:
                try:
                    base_info["tool_adapter_stats"] = (
                        self.tool_adapter.get_adapter_stats()
                    )
                except Exception:
                    base_info["tool_adapter_stats"] = {"error": "stats unavailable"}

            # Workflow orchestrator stats
            if self.workflow_orchestrator:
                base_info["workflow_stats"] = (
                    self.workflow_orchestrator.get_orchestrator_stats()
                )

            # Communication hub stats
            if self.communication_hub:
                base_info["communication_stats"] = (
                    self.communication_hub.get_communication_stats()
                )

            # Enhanced delegation manager stats
            if self.delegation_manager:
                base_info["delegation_stats"] = (
                    self.delegation_manager.get_delegation_stats()
                )

            # Enhanced capabilities summary
            base_info["enhanced_features"] = {
                "workflow_orchestration": self.workflow_orchestrator is not None,
                "inter_agent_communication": self.communication_hub is not None,
                "enhanced_delegation": self.delegation_manager is not None,
                "context_preservation": self.context_manager is not None,
                "collaboration_support": self.communication_hub is not None,
                "multi_step_workflows": self.workflow_orchestrator is not None,
            }

        return base_info

    async def cleanup(self) -> None:
        """Cleanup multi-agent service resources including enhanced components."""
        self.logger.info("Cleaning up enhanced multi-agent service")

        # Cleanup agents
        for agent in self.agents.values():
            try:
                await agent.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up agent {agent.agent_id}: {e}")

        self.agents.clear()

        # Cleanup enhanced components
        if self.workflow_orchestrator:
            await self.workflow_orchestrator.cleanup()
            self.workflow_orchestrator = None

        if self.communication_hub:
            await self.communication_hub.cleanup()
            self.communication_hub = None

        if self.delegation_manager:
            await self.delegation_manager.cleanup()
            self.delegation_manager = None

        # Cleanup core components
        if self.agent_router:
            await self.agent_router.cleanup()
            self.agent_router = None

        if self.context_manager:
            await self.context_manager.cleanup()
            self.context_manager = None

        if self.tool_adapter:
            await self.tool_adapter.cleanup()
            self.tool_adapter = None

        # Cleanup single-agent fallback
        if self.llamaindex_service:
            await self.llamaindex_service.cleanup()

        self.is_initialized = False
        self.multi_agent_enabled = False

        self.logger.info("Enhanced multi-agent service cleanup complete")
