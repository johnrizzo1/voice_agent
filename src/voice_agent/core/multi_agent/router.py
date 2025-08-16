"""
Agent router with hybrid routing strategy.

Provides intelligent routing of messages to appropriate agents using
a combination of rule-based matching, embedding similarity, and LLM fallback.
"""

import asyncio
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object
    Field = lambda **kwargs: None

from .message import AgentMessage, AgentResponse, MessageType
from .agent_base import AgentBase, AgentCapability

# LlamaIndex imports for embeddings
try:
    from llama_index.embeddings.ollama import OllamaEmbedding
    import numpy as np

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    OllamaEmbedding = None
    np = None
    EMBEDDINGS_AVAILABLE = False


class RoutingStrategy(str, Enum):
    """Available routing strategies."""

    RULES_ONLY = "rules_only"
    EMBEDDINGS_ONLY = "embeddings_only"
    HYBRID = "hybrid"
    LLM_FALLBACK = "llm_fallback"


class RouteDecision(BaseModel if BaseModel != object else dict):
    """Decision result from the routing process."""

    if BaseModel != object:
        target_agent: str
        confidence: float = Field(ge=0.0, le=1.0)
        strategy_used: RoutingStrategy
        reasoning: str
        alternatives: List[Tuple[str, float]] = Field(default_factory=list)
        metadata: Dict[str, Any] = Field(default_factory=dict)


class RoutingRule:
    """Rule-based routing configuration."""

    def __init__(
        self,
        name: str,
        target_agent: str,
        patterns: List[str],
        capabilities: Optional[List[AgentCapability]] = None,
        priority: int = 5,
        confidence: float = 0.8,
    ):
        """
        Initialize a routing rule.

        Args:
            name: Rule identifier
            target_agent: Target agent for this rule
            patterns: Regex patterns to match against message content
            capabilities: Required capabilities (optional)
            priority: Rule priority (1=highest, 10=lowest)
            confidence: Confidence score when rule matches
        """
        self.name = name
        self.target_agent = target_agent
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        self.capabilities = capabilities or []
        self.priority = priority
        self.confidence = confidence

    def matches(self, message: AgentMessage) -> bool:
        """Check if this rule matches the given message."""
        # Check content patterns
        for pattern in self.patterns:
            if pattern.search(message.content):
                return True

        # Check capability requirements in metadata
        if self.capabilities and message.metadata.get("required_capabilities"):
            required_caps = set(message.metadata["required_capabilities"])
            rule_caps = set(self.capabilities)
            if required_caps.intersection(rule_caps):
                return True

        return False


class AgentRouter:
    """
    Hybrid agent router combining rule-based, embedding-based, and LLM-based routing.

    Routing process:
    1. Rule-based matching for explicit patterns
    2. Embedding similarity for semantic matching
    3. LLM fallback for complex routing decisions
    4. Load balancing among capable agents
    """

    def __init__(
        self,
        default_agent: str = "general_agent",
        strategy: RoutingStrategy = RoutingStrategy.HYBRID,
        embedding_model: str = "nomic-embed-text",
        confidence_threshold: float = 0.7,
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize the agent router.

        Args:
            default_agent: Default agent when no specific route found
            strategy: Primary routing strategy to use
            embedding_model: Model name for embedding-based routing
            confidence_threshold: Minimal confidence for routing decisions
            state_callback: Optional callback for state changes
        """
        self.default_agent = default_agent
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        self._state_callback = state_callback

        # Registered agents and their capabilities
        self.agents: Dict[str, AgentBase] = {}
        self.agent_capabilities: Dict[str, Set[AgentCapability]] = {}
        self.agent_load: Dict[str, int] = {}  # Current task count per agent

        # Routing rules
        self.routing_rules: List[RoutingRule] = []

        # Embedding-based routing
        self.embedding_model: Optional[OllamaEmbedding] = None
        self.capability_embeddings: Dict[str, List[float]] = {}
        self.agent_embeddings: Dict[str, List[float]] = {}

        # Routing statistics
        self.routing_stats: Dict[str, int] = {}
        self.routing_history: List[Dict[str, Any]] = []

        # Initialize with default rules
        self._setup_default_rules()

    def _emit_state(self, state: str, message: Optional[str] = None) -> None:
        """Emit state change via callback."""
        if self._state_callback:
            try:
                self._state_callback("router", state, message)
            except Exception:
                self.logger.debug("Router state callback error", exc_info=True)

    async def initialize(self) -> None:
        """Initialize the router and embedding model."""
        self.logger.info("Initializing agent router")
        self._emit_state("initializing", "setting up routing components")

        if self.strategy in [RoutingStrategy.EMBEDDINGS_ONLY, RoutingStrategy.HYBRID]:
            await self._initialize_embeddings()

        self.logger.info(f"Router initialized with strategy: {self.strategy}")
        self._emit_state("ready", f"router ready with {len(self.agents)} agents")

    async def _initialize_embeddings(self) -> None:
        """Initialize embedding model and precompute embeddings."""
        if not EMBEDDINGS_AVAILABLE:
            self.logger.warning(
                "Embeddings not available - falling back to rules-only routing"
            )
            self.strategy = RoutingStrategy.RULES_ONLY
            return

        try:
            self.embedding_model = OllamaEmbedding(
                model_name="nomic-embed-text", base_url="http://localhost:11434"
            )

            # Precompute capability embeddings
            await self._compute_capability_embeddings()

            self.logger.info("Embedding model initialized for semantic routing")

        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {e}")
            self.strategy = RoutingStrategy.RULES_ONLY

    async def _compute_capability_embeddings(self) -> None:
        """Precompute embeddings for agent capabilities."""
        if not self.embedding_model:
            return

        capability_descriptions = {
            AgentCapability.GENERAL_CHAT: "general conversation and chat",
            AgentCapability.TOOL_EXECUTION: "executing tools and functions",
            AgentCapability.CODE_ANALYSIS: "analyzing and reviewing code",
            AgentCapability.FILE_OPERATIONS: "file and directory operations",
            AgentCapability.WEB_SEARCH: "searching the web for information",
            AgentCapability.CALCULATIONS: "mathematical calculations and computations",
            AgentCapability.WEATHER_INFO: "weather information and forecasts",
            AgentCapability.SYSTEM_INFO: "system information and monitoring",
            AgentCapability.CONVERSATION_MEMORY: "remembering conversation context",
            AgentCapability.RAG_RETRIEVAL: "retrieving information from knowledge base",
            AgentCapability.TASK_PLANNING: "planning and organizing tasks",
        }

        for capability, description in capability_descriptions.items():
            try:
                # Get embedding for capability description
                embedding = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.embedding_model.get_text_embedding(description)
                )
                self.capability_embeddings[capability.value] = embedding

            except Exception as e:
                self.logger.error(f"Failed to compute embedding for {capability}: {e}")

    def register_agent(self, agent: AgentBase) -> None:
        """
        Register an agent with the router.

        Args:
            agent: Agent instance to register
        """
        self.agents[agent.agent_id] = agent
        self.agent_capabilities[agent.agent_id] = agent.capabilities.copy()
        self.agent_load[agent.agent_id] = 0

        self.logger.info(
            f"Registered agent {agent.agent_id} with capabilities: {agent.capabilities}"
        )

        # Compute agent embedding if embeddings are available
        if self.embedding_model and agent.capabilities:
            asyncio.create_task(self._compute_agent_embedding(agent.agent_id))

    async def _compute_agent_embedding(self, agent_id: str) -> None:
        """Compute embedding for an agent based on its capabilities."""
        if not self.embedding_model or agent_id not in self.agent_capabilities:
            return

        try:
            capabilities = self.agent_capabilities[agent_id]
            capability_text = " ".join(
                [cap.value.replace("_", " ") for cap in capabilities]
            )

            embedding = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.embedding_model.get_text_embedding(capability_text)
            )
            self.agent_embeddings[agent_id] = embedding

        except Exception as e:
            self.logger.error(f"Failed to compute embedding for agent {agent_id}: {e}")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the router.

        Args:
            agent_id: ID of agent to unregister
        """
        self.agents.pop(agent_id, None)
        self.agent_capabilities.pop(agent_id, None)
        self.agent_load.pop(agent_id, None)
        self.agent_embeddings.pop(agent_id, None)

        self.logger.info(f"Unregistered agent {agent_id}")

    def _setup_default_rules(self) -> None:
        """Setup default routing rules."""
        default_rules = [
            RoutingRule(
                name="tool_requests",
                target_agent="tool_specialist",
                patterns=[
                    r"calculate|compute|math|equation",
                    r"file|directory|folder|path",
                    r"weather|temperature|forecast",
                    r"search|find|look up",
                ],
                capabilities=[
                    AgentCapability.TOOL_EXECUTION,
                    AgentCapability.CALCULATIONS,
                    AgentCapability.FILE_OPERATIONS,
                ],
                priority=2,
                confidence=0.8,
            ),
            RoutingRule(
                name="code_analysis",
                target_agent="code_specialist",
                patterns=[
                    r"code|programming|function|class|method",
                    r"debug|error|bug|syntax",
                    r"refactor|optimize|review",
                ],
                capabilities=[AgentCapability.CODE_ANALYSIS],
                priority=2,
                confidence=0.85,
            ),
            RoutingRule(
                name="general_chat",
                target_agent="general_agent",
                patterns=[
                    r"hello|hi|hey|greetings",
                    r"how are you|what.*up",
                    r"tell me about|explain|what is",
                ],
                capabilities=[AgentCapability.GENERAL_CHAT],
                priority=8,
                confidence=0.6,
            ),
        ]

        self.routing_rules.extend(default_rules)
        self.logger.info(f"Initialized with {len(default_rules)} default routing rules")

    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add a custom routing rule."""
        self.routing_rules.append(rule)
        # Sort by priority (lower number = higher priority)
        self.routing_rules.sort(key=lambda r: r.priority)
        self.logger.info(f"Added routing rule: {rule.name}")

    async def route_message(self, message: AgentMessage) -> RouteDecision:
        """
        Route a message to the most appropriate agent.

        Args:
            message: Message to route

        Returns:
            Routing decision with target agent and confidence
        """
        self.logger.debug(f"Routing message {message.id}: {message.content[:100]}...")
        self._emit_state("active", f"routing message {message.id}")

        try:
            decision = None

            # Try routing strategies in order
            if self.strategy in [RoutingStrategy.RULES_ONLY, RoutingStrategy.HYBRID]:
                decision = await self._route_by_rules(message)
                if decision and decision.confidence >= self.confidence_threshold:
                    decision.strategy_used = RoutingStrategy.RULES_ONLY
                    self._record_routing_decision(decision)
                    return decision

            if self.strategy in [
                RoutingStrategy.EMBEDDINGS_ONLY,
                RoutingStrategy.HYBRID,
            ]:
                decision = await self._route_by_embeddings(message)
                if decision and decision.confidence >= self.confidence_threshold:
                    decision.strategy_used = RoutingStrategy.EMBEDDINGS_ONLY
                    self._record_routing_decision(decision)
                    return decision

            if self.strategy in [RoutingStrategy.LLM_FALLBACK, RoutingStrategy.HYBRID]:
                decision = await self._route_by_llm(message)
                if decision and decision.confidence >= self.confidence_threshold:
                    decision.strategy_used = RoutingStrategy.LLM_FALLBACK
                    self._record_routing_decision(decision)
                    return decision

            # Fallback to default agent with load balancing
            target_agent = self._select_by_load_balancing([self.default_agent])
            decision = (
                RouteDecision(
                    target_agent=target_agent,
                    confidence=0.5,
                    strategy_used=RoutingStrategy.RULES_ONLY,
                    reasoning="fallback to default agent",
                    metadata={"fallback": True},
                )
                if hasattr(RouteDecision, "target_agent")
                else {
                    "target_agent": target_agent,
                    "confidence": 0.5,
                    "strategy_used": RoutingStrategy.RULES_ONLY,
                    "reasoning": "fallback to default agent",
                    "alternatives": [],
                    "metadata": {"fallback": True},
                }
            )

            self._record_routing_decision(decision)
            return decision

        except Exception as e:
            self.logger.error(f"Error routing message: {e}")
            # Emergency fallback
            return (
                RouteDecision(
                    target_agent=self.default_agent,
                    confidence=0.1,
                    strategy_used=RoutingStrategy.RULES_ONLY,
                    reasoning=f"error fallback: {str(e)}",
                    metadata={"error": True},
                )
                if hasattr(RouteDecision, "target_agent")
                else {
                    "target_agent": self.default_agent,
                    "confidence": 0.1,
                    "strategy_used": RoutingStrategy.RULES_ONLY,
                    "reasoning": f"error fallback: {str(e)}",
                    "alternatives": [],
                    "metadata": {"error": True},
                }
            )

        finally:
            self._emit_state("ready", None)

    async def _route_by_rules(self, message: AgentMessage) -> Optional[RouteDecision]:
        """Route message using rule-based matching."""
        best_rule = None
        best_priority = float("inf")

        for rule in self.routing_rules:
            if rule.matches(message) and rule.priority < best_priority:
                # Check if target agent exists and is available
                if rule.target_agent in self.agents:
                    best_rule = rule
                    best_priority = rule.priority

        if best_rule:
            # Apply load balancing among agents with same capabilities
            capable_agents = self._find_capable_agents(best_rule.capabilities or [])
            target_agent = self._select_by_load_balancing(
                capable_agents or [best_rule.target_agent]
            )

            decision = (
                RouteDecision(
                    target_agent=target_agent,
                    confidence=best_rule.confidence,
                    strategy_used=RoutingStrategy.RULES_ONLY,
                    reasoning=f"matched rule: {best_rule.name}",
                    metadata={"rule": best_rule.name, "priority": best_rule.priority},
                )
                if hasattr(RouteDecision, "target_agent")
                else {
                    "target_agent": target_agent,
                    "confidence": best_rule.confidence,
                    "strategy_used": RoutingStrategy.RULES_ONLY,
                    "reasoning": f"matched rule: {best_rule.name}",
                    "alternatives": [],
                    "metadata": {
                        "rule": best_rule.name,
                        "priority": best_rule.priority,
                    },
                }
            )

            self.logger.debug(
                f"Rule-based routing: {message.id} -> {target_agent} (rule: {best_rule.name})"
            )
            return decision

        return None

    async def _route_by_embeddings(
        self, message: AgentMessage
    ) -> Optional[RouteDecision]:
        """Route message using embedding similarity."""
        if not self.embedding_model or not self.agent_embeddings:
            return None

        try:
            # Get message embedding
            message_embedding = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.embedding_model.get_text_embedding(message.content)
            )

            # Calculate similarity with each agent
            similarities = []
            for agent_id, agent_embedding in self.agent_embeddings.items():
                if agent_id in self.agents:  # Ensure agent is still registered
                    similarity = self._cosine_similarity(
                        message_embedding, agent_embedding
                    )
                    similarities.append((agent_id, similarity))

            if similarities:
                # Sort by similarity (highest first)
                similarities.sort(key=lambda x: x[1], reverse=True)

                best_agent, best_similarity = similarities[0]
                alternatives = similarities[1:4]  # Top 3 alternatives

                # Apply load balancing among top candidates
                top_candidates = [
                    agent
                    for agent, sim in similarities[:3]
                    if sim >= best_similarity * 0.9
                ]
                target_agent = self._select_by_load_balancing(top_candidates)

                decision = (
                    RouteDecision(
                        target_agent=target_agent,
                        confidence=min(best_similarity, 0.95),  # Cap confidence
                        strategy_used=RoutingStrategy.EMBEDDINGS_ONLY,
                        reasoning=f"embedding similarity: {best_similarity:.3f}",
                        alternatives=alternatives,
                        metadata={"similarity": best_similarity},
                    )
                    if hasattr(RouteDecision, "target_agent")
                    else {
                        "target_agent": target_agent,
                        "confidence": min(best_similarity, 0.95),
                        "strategy_used": RoutingStrategy.EMBEDDINGS_ONLY,
                        "reasoning": f"embedding similarity: {best_similarity:.3f}",
                        "alternatives": alternatives,
                        "metadata": {"similarity": best_similarity},
                    }
                )

                self.logger.debug(
                    f"Embedding-based routing: {message.id} -> {target_agent} (sim: {best_similarity:.3f})"
                )
                return decision

        except Exception as e:
            self.logger.error(f"Embedding routing error: {e}")

        return None

    async def _route_by_llm(self, message: AgentMessage) -> Optional[RouteDecision]:
        """Route message using LLM-based decision making."""
        # This is a placeholder for LLM-based routing
        # In a full implementation, you would query an LLM to make routing decisions
        self.logger.debug("LLM-based routing not implemented yet")
        return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not np:
            # Fallback implementation without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)

        # Numpy implementation
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        return np.dot(vec1_np, vec2_np) / (
            np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)
        )

    def _find_capable_agents(
        self, required_capabilities: List[AgentCapability]
    ) -> List[str]:
        """Find agents that have the required capabilities."""
        capable_agents = []

        for agent_id, capabilities in self.agent_capabilities.items():
            if agent_id in self.agents:  # Ensure agent is still registered
                if any(cap in capabilities for cap in required_capabilities):
                    capable_agents.append(agent_id)

        return capable_agents

    def _select_by_load_balancing(self, candidate_agents: List[str]) -> str:
        """Select agent from candidates based on current load."""
        if not candidate_agents:
            return self.default_agent

        # Filter to only registered agents
        available_agents = [
            agent_id for agent_id in candidate_agents if agent_id in self.agents
        ]
        if not available_agents:
            return self.default_agent

        # Select agent with lowest current load
        min_load = min(
            self.agent_load.get(agent_id, 0) for agent_id in available_agents
        )
        low_load_agents = [
            agent_id
            for agent_id in available_agents
            if self.agent_load.get(agent_id, 0) == min_load
        ]

        # If multiple agents have same low load, pick first one
        selected = low_load_agents[0]

        # Update load tracking
        self.agent_load[selected] = self.agent_load.get(selected, 0) + 1

        return selected

    def update_agent_load(self, agent_id: str, delta: int) -> None:
        """
        Update the load tracking for an agent.

        Args:
            agent_id: Agent identifier
            delta: Change in load (+1 for new task, -1 for completed task)
        """
        if agent_id in self.agent_load:
            self.agent_load[agent_id] = max(0, self.agent_load[agent_id] + delta)

    def _record_routing_decision(self, decision) -> None:
        """Record routing decision for statistics and analysis."""
        target_agent = (
            decision.target_agent
            if hasattr(decision, "target_agent")
            else decision["target_agent"]
        )
        strategy = (
            decision.strategy_used
            if hasattr(decision, "strategy_used")
            else decision["strategy_used"]
        )
        confidence = (
            decision.confidence
            if hasattr(decision, "confidence")
            else decision["confidence"]
        )

        # Update statistics
        if target_agent not in self.routing_stats:
            self.routing_stats[target_agent] = 0
        self.routing_stats[target_agent] += 1

        # Record in history (keep last 100 decisions)
        history_entry = {
            "timestamp": asyncio.get_event_loop().time(),
            "target_agent": target_agent,
            "strategy": strategy.value if hasattr(strategy, "value") else strategy,
            "confidence": confidence,
        }
        self.routing_history.append(history_entry)

        if len(self.routing_history) > 100:
            self.routing_history = self.routing_history[-80:]  # Keep most recent 80

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics."""
        total_routes = sum(self.routing_stats.values())

        return {
            "total_routes": total_routes,
            "routes_per_agent": self.routing_stats.copy(),
            "registered_agents": len(self.agents),
            "routing_rules": len(self.routing_rules),
            "strategy": self.strategy.value,
            "has_embeddings": self.embedding_model is not None,
            "current_load": self.agent_load.copy(),
        }

    async def cleanup(self) -> None:
        """Cleanup router resources."""
        self.logger.info("Cleaning up agent router")

        self.agents.clear()
        self.agent_capabilities.clear()
        self.agent_load.clear()
        self.routing_rules.clear()
        self.capability_embeddings.clear()
        self.agent_embeddings.clear()
        self.routing_history.clear()
        self.routing_stats.clear()

        self.embedding_model = None

        self._emit_state("shutdown", "router cleanup complete")
