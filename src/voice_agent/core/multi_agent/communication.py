"""
Advanced inter-agent communication protocol.

Provides sophisticated communication patterns for multi-agent coordination,
including direct agent messaging, broadcast channels, event subscriptions,
and collaborative reasoning workflows.
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


class CommunicationType(str, Enum):
    """Types of inter-agent communication."""

    DIRECT_MESSAGE = "direct_message"
    BROADCAST = "broadcast"
    REQUEST_RESPONSE = "request_response"
    COLLABORATION = "collaboration"
    STATUS_UPDATE = "status_update"
    RESULT_SHARING = "result_sharing"
    EVENT_NOTIFICATION = "event_notification"


class MessagePriority(str, Enum):
    """Priority levels for agent communications."""

    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class CommunicationChannel(str, Enum):
    """Communication channels for different types of interactions."""

    COORDINATION = "coordination"  # Task coordination and planning
    DATA_SHARING = "data_sharing"  # Sharing results and data
    STATUS_UPDATES = "status_updates"  # Agent status and health
    COLLABORATION = "collaboration"  # Joint problem solving
    EMERGENCY = "emergency"  # Critical alerts and errors


@dataclass
class AgentCommunication:
    """Enhanced communication message between agents."""

    communication_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = ""
    to_agent: Optional[str] = None  # None for broadcast
    channel: CommunicationChannel = CommunicationChannel.COORDINATION
    communication_type: CommunicationType = CommunicationType.DIRECT_MESSAGE
    priority: MessagePriority = MessagePriority.NORMAL

    # Message content
    subject: str = ""
    content: Any = None
    data_payload: Dict[str, Any] = field(default_factory=dict)

    # Communication flow
    conversation_id: str = ""
    reply_to: Optional[str] = None
    expects_reply: bool = False
    timeout_seconds: Optional[float] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Response tracking
    responses: List[str] = field(default_factory=list)  # Response IDs
    is_delivered: bool = False
    is_acknowledged: bool = False


class CommunicationHub:
    """
    Central hub for managing inter-agent communications.

    Provides message routing, channel management, subscription handling,
    and coordination patterns for complex multi-agent workflows.
    """

    def __init__(
        self,
        agent_registry: Dict[str, AgentBase],
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize communication hub.

        Args:
            agent_registry: Registry of available agents
            state_callback: Optional callback for state updates
        """
        self.agent_registry = agent_registry
        self.logger = logging.getLogger(__name__)
        self._state_callback = state_callback

        # Communication channels
        self.channels: Dict[CommunicationChannel, Set[str]] = {
            channel: set() for channel in CommunicationChannel
        }

        # Message storage and routing
        self.message_queue: Dict[str, List[AgentCommunication]] = {}
        self.active_communications: Dict[str, AgentCommunication] = {}
        self.communication_history: List[AgentCommunication] = []

        # Subscriptions and listeners
        self.channel_subscribers: Dict[CommunicationChannel, Set[str]] = {
            channel: set() for channel in CommunicationChannel
        }
        self.event_listeners: Dict[str, List[Callable]] = {}

        # Collaboration sessions
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "broadcasts_sent": 0,
            "collaborations_initiated": 0,
            "average_response_time": 0.0,
        }

    def _emit_state(self, state: str, message: Optional[str] = None) -> None:
        """Emit state change via callback."""
        if self._state_callback:
            try:
                self._state_callback("communication_hub", state, message)
            except Exception:
                self.logger.debug(
                    "Communication hub state callback error", exc_info=True
                )

    async def send_message(
        self,
        from_agent: str,
        to_agent: Optional[str],
        content: Any,
        channel: CommunicationChannel = CommunicationChannel.COORDINATION,
        communication_type: CommunicationType = CommunicationType.DIRECT_MESSAGE,
        priority: MessagePriority = MessagePriority.NORMAL,
        subject: str = "",
        data_payload: Optional[Dict[str, Any]] = None,
        expects_reply: bool = False,
        timeout_seconds: Optional[float] = None,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Send a message between agents.

        Args:
            from_agent: Sender agent ID
            to_agent: Recipient agent ID (None for broadcast)
            content: Message content
            channel: Communication channel
            communication_type: Type of communication
            priority: Message priority
            subject: Message subject
            data_payload: Additional data
            expects_reply: Whether a reply is expected
            timeout_seconds: Timeout for response
            conversation_id: Conversation context
            **kwargs: Additional metadata

        Returns:
            Communication ID
        """
        communication = AgentCommunication(
            from_agent=from_agent,
            to_agent=to_agent,
            channel=channel,
            communication_type=communication_type,
            priority=priority,
            subject=subject,
            content=content,
            data_payload=data_payload or {},
            expects_reply=expects_reply,
            timeout_seconds=timeout_seconds,
            conversation_id=conversation_id or str(uuid.uuid4()),
            metadata=kwargs,
        )

        # Store communication
        self.active_communications[communication.communication_id] = communication

        # Route message
        if to_agent:
            # Direct message
            await self._deliver_message(communication)
            self.stats["messages_sent"] += 1
        else:
            # Broadcast
            await self._broadcast_message(communication)
            self.stats["broadcasts_sent"] += 1

        self.logger.debug(
            f"Sent communication {communication.communication_id} from {from_agent} to {to_agent or 'broadcast'}"
        )

        return communication.communication_id

    async def _deliver_message(self, communication: AgentCommunication) -> bool:
        """Deliver a direct message to a specific agent."""
        try:
            # Add to recipient's message queue
            if communication.to_agent not in self.message_queue:
                self.message_queue[communication.to_agent] = []

            self.message_queue[communication.to_agent].append(communication)
            communication.is_delivered = True

            # Notify recipient if they have a message handler
            if communication.to_agent in self.agent_registry:
                agent = self.agent_registry[communication.to_agent]
                if hasattr(agent, "_handle_agent_communication"):
                    await agent._handle_agent_communication(communication)

            self.stats["messages_delivered"] += 1
            self._emit_state(
                "message_delivered", f"delivered to {communication.to_agent}"
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to deliver message {communication.communication_id}: {e}"
            )
            return False

    async def _broadcast_message(self, communication: AgentCommunication) -> None:
        """Broadcast a message to all subscribed agents."""
        channel_subscribers = self.channel_subscribers.get(communication.channel, set())

        for agent_id in channel_subscribers:
            if agent_id != communication.from_agent:  # Don't send to sender
                # Create individual copy for each recipient
                individual_comm = AgentCommunication(
                    communication_id=str(uuid.uuid4()),
                    from_agent=communication.from_agent,
                    to_agent=agent_id,
                    channel=communication.channel,
                    communication_type=communication.communication_type,
                    priority=communication.priority,
                    subject=f"[BROADCAST] {communication.subject}",
                    content=communication.content,
                    data_payload=communication.data_payload.copy(),
                    conversation_id=communication.conversation_id,
                    metadata={
                        **communication.metadata,
                        "is_broadcast": True,
                        "original_id": communication.communication_id,
                    },
                )

                await self._deliver_message(individual_comm)

        self.logger.info(
            f"Broadcast {communication.communication_id} sent to {len(channel_subscribers)} agents"
        )

    async def request_collaboration(
        self,
        initiator_agent: str,
        target_agents: List[str],
        task_description: str,
        collaboration_type: str = "problem_solving",
        timeout_seconds: float = 120.0,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Initiate a collaboration session between multiple agents.

        Args:
            initiator_agent: Agent initiating the collaboration
            target_agents: List of agents to invite
            task_description: Description of the collaboration task
            collaboration_type: Type of collaboration
            timeout_seconds: Session timeout
            context_data: Shared context data

        Returns:
            Collaboration session ID
        """
        session_id = str(uuid.uuid4())

        collaboration_session = {
            "session_id": session_id,
            "initiator": initiator_agent,
            "participants": [initiator_agent] + target_agents,
            "task_description": task_description,
            "collaboration_type": collaboration_type,
            "status": "active",
            "created_at": datetime.utcnow(),
            "timeout_seconds": timeout_seconds,
            "context_data": context_data or {},
            "messages": [],
            "shared_workspace": {},
            "results": {},
        }

        self.active_collaborations[session_id] = collaboration_session

        # Send collaboration invitations
        for agent_id in target_agents:
            await self.send_message(
                from_agent=initiator_agent,
                to_agent=agent_id,
                content=f"Collaboration invitation: {task_description}",
                channel=CommunicationChannel.COLLABORATION,
                communication_type=CommunicationType.COLLABORATION,
                priority=MessagePriority.HIGH,
                subject="Collaboration Invitation",
                data_payload={
                    "session_id": session_id,
                    "collaboration_type": collaboration_type,
                    "context_data": context_data,
                },
                expects_reply=True,
                timeout_seconds=30.0,
            )

        self.stats["collaborations_initiated"] += 1
        self.logger.info(
            f"Initiated collaboration {session_id} with {len(target_agents)} agents"
        )

        return session_id

    async def join_collaboration(
        self, agent_id: str, session_id: str, acceptance_message: Optional[str] = None
    ) -> bool:
        """Join an active collaboration session."""
        if session_id not in self.active_collaborations:
            return False

        session = self.active_collaborations[session_id]

        if agent_id not in session["participants"]:
            session["participants"].append(agent_id)

        # Notify other participants
        await self.send_message(
            from_agent=agent_id,
            to_agent=None,  # Broadcast to session participants
            content=acceptance_message or f"Agent {agent_id} joined the collaboration",
            channel=CommunicationChannel.COLLABORATION,
            communication_type=CommunicationType.STATUS_UPDATE,
            subject="Collaboration Join",
            data_payload={"session_id": session_id, "action": "join"},
        )

        return True

    async def contribute_to_collaboration(
        self,
        agent_id: str,
        session_id: str,
        contribution_type: str,
        content: Any,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Contribute to a collaboration session."""
        if session_id not in self.active_collaborations:
            return False

        session = self.active_collaborations[session_id]

        if agent_id not in session["participants"]:
            return False

        contribution = {
            "id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "type": contribution_type,
            "content": content,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        session["messages"].append(contribution)

        # Update shared workspace if applicable
        if contribution_type == "shared_data":
            session["shared_workspace"].update(data or {})

        # Notify other participants
        other_participants = [p for p in session["participants"] if p != agent_id]
        for participant in other_participants:
            await self.send_message(
                from_agent=agent_id,
                to_agent=participant,
                content=content,
                channel=CommunicationChannel.COLLABORATION,
                communication_type=CommunicationType.COLLABORATION,
                subject=f"Collaboration Contribution: {contribution_type}",
                data_payload={
                    "session_id": session_id,
                    "contribution_id": contribution["id"],
                    "contribution_type": contribution_type,
                    "contribution_data": data,
                },
            )

        return True

    async def aggregate_collaboration_results(
        self, session_id: str, aggregation_method: str = "consensus"
    ) -> Dict[str, Any]:
        """
        Aggregate results from a collaboration session.

        Args:
            session_id: Collaboration session ID
            aggregation_method: Method for aggregating results

        Returns:
            Aggregated results
        """
        if session_id not in self.active_collaborations:
            return {"error": "Session not found"}

        session = self.active_collaborations[session_id]

        # Extract results from contributions
        results = []
        data_contributions = []

        for message in session["messages"]:
            if message["type"] == "result":
                results.append(message)
            elif message["type"] == "shared_data":
                data_contributions.append(message)

        # Aggregate based on method
        if aggregation_method == "consensus":
            # Simple consensus: most common result
            result_counts = {}
            for result in results:
                content_str = str(result["content"])
                result_counts[content_str] = result_counts.get(content_str, 0) + 1

            if result_counts:
                consensus_result = max(result_counts, key=result_counts.get)
                confidence = result_counts[consensus_result] / len(results)
            else:
                consensus_result = "No consensus reached"
                confidence = 0.0

            aggregated = {
                "method": "consensus",
                "result": consensus_result,
                "confidence": confidence,
                "participant_count": len(session["participants"]),
                "contribution_count": len(results),
            }

        elif aggregation_method == "weighted_average":
            # For numerical results, compute weighted average
            numerical_results = []
            weights = []

            for result in results:
                try:
                    value = float(result["content"])
                    numerical_results.append(value)
                    # Simple weight based on agent capabilities (placeholder)
                    weights.append(1.0)
                except (ValueError, TypeError):
                    continue

            if numerical_results:
                weighted_sum = sum(v * w for v, w in zip(numerical_results, weights))
                total_weight = sum(weights)
                aggregated_result = (
                    weighted_sum / total_weight if total_weight > 0 else 0
                )
            else:
                aggregated_result = 0

            aggregated = {
                "method": "weighted_average",
                "result": aggregated_result,
                "numerical_results": numerical_results,
                "weights": weights,
            }

        else:
            # Simple concatenation
            all_results = [result["content"] for result in results]
            aggregated = {
                "method": "concatenation",
                "results": all_results,
                "shared_data": session["shared_workspace"],
            }

        # Store final results
        session["results"] = aggregated
        session["status"] = "completed"

        return aggregated

    def subscribe_to_channel(
        self, agent_id: str, channel: CommunicationChannel
    ) -> bool:
        """Subscribe an agent to a communication channel."""
        self.channel_subscribers[channel].add(agent_id)
        self.logger.debug(f"Agent {agent_id} subscribed to channel {channel.value}")
        return True

    def unsubscribe_from_channel(
        self, agent_id: str, channel: CommunicationChannel
    ) -> bool:
        """Unsubscribe an agent from a communication channel."""
        self.channel_subscribers[channel].discard(agent_id)
        self.logger.debug(f"Agent {agent_id} unsubscribed from channel {channel.value}")
        return True

    def get_pending_messages(self, agent_id: str) -> List[AgentCommunication]:
        """Get pending messages for an agent."""
        return self.message_queue.get(agent_id, [])

    def acknowledge_message(self, agent_id: str, communication_id: str) -> bool:
        """Acknowledge receipt of a message."""
        if communication_id in self.active_communications:
            communication = self.active_communications[communication_id]
            if communication.to_agent == agent_id:
                communication.is_acknowledged = True
                return True
        return False

    async def send_status_update(
        self,
        agent_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        broadcast: bool = True,
    ) -> None:
        """Send agent status update."""
        await self.send_message(
            from_agent=agent_id,
            to_agent=None if broadcast else "system",
            content=f"Status: {status}",
            channel=CommunicationChannel.STATUS_UPDATES,
            communication_type=CommunicationType.STATUS_UPDATE,
            priority=MessagePriority.NORMAL,
            subject="Agent Status Update",
            data_payload={
                "status": status,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication hub statistics."""
        return {
            **self.stats,
            "active_communications": len(self.active_communications),
            "active_collaborations": len(self.active_collaborations),
            "total_subscribers": sum(
                len(subs) for subs in self.channel_subscribers.values()
            ),
            "channel_activity": {
                channel.value: len(subscribers)
                for channel, subscribers in self.channel_subscribers.items()
            },
            "queued_messages": sum(len(queue) for queue in self.message_queue.values()),
        }

    async def cleanup_expired_communications(self) -> int:
        """Clean up expired communications and return count."""
        now = datetime.utcnow()
        cleaned_count = 0

        # Clean up active communications
        expired_ids = []
        for comm_id, communication in self.active_communications.items():
            if communication.timeout_seconds:
                age = (now - communication.timestamp).total_seconds()
                if age > communication.timeout_seconds:
                    expired_ids.append(comm_id)

        for comm_id in expired_ids:
            del self.active_communications[comm_id]
            cleaned_count += 1

        # Clean up old collaboration sessions
        expired_sessions = []
        for session_id, session in self.active_collaborations.items():
            session_age = (now - session["created_at"]).total_seconds()
            if session_age > session["timeout_seconds"]:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.active_collaborations[session_id]
            cleaned_count += 1

        # Trim communication history
        if len(self.communication_history) > 1000:
            self.communication_history = self.communication_history[-800:]
            cleaned_count += 200

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} expired communications")

        return cleaned_count

    async def cleanup(self) -> None:
        """Cleanup communication hub resources."""
        self.logger.info("Cleaning up communication hub")

        self.message_queue.clear()
        self.active_communications.clear()
        self.communication_history.clear()
        self.active_collaborations.clear()

        for channel_subs in self.channel_subscribers.values():
            channel_subs.clear()

        self.event_listeners.clear()

        self.logger.info("Communication hub cleanup complete")


class EnhancedDelegationManager:
    """
    Enhanced delegation manager with sophisticated patterns.

    Provides advanced delegation patterns including task decomposition,
    capability-based routing, load balancing, and failure recovery.
    """

    def __init__(
        self,
        agent_registry: Dict[str, AgentBase],
        communication_hub: CommunicationHub,
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize enhanced delegation manager.

        Args:
            agent_registry: Registry of available agents
            communication_hub: Communication hub for agent coordination
            state_callback: Optional callback for state updates
        """
        self.agent_registry = agent_registry
        self.communication_hub = communication_hub
        self.logger = logging.getLogger(__name__)
        self._state_callback = state_callback

        # Delegation patterns and strategies
        self.delegation_patterns: Dict[str, Callable] = {
            "capability_based": self._delegate_by_capability,
            "load_balanced": self._delegate_by_load,
            "expertise_weighted": self._delegate_by_expertise,
            "collaborative": self._delegate_collaborative,
            "hierarchical": self._delegate_hierarchical,
            "consensus": self._delegate_consensus,
        }

        # Agent performance tracking
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        self.delegation_history: List[Dict[str, Any]] = []

        # Delegation statistics
        self.stats = {
            "delegations_made": 0,
            "successful_delegations": 0,
            "failed_delegations": 0,
            "average_delegation_time": 0.0,
            "pattern_usage": {pattern: 0 for pattern in self.delegation_patterns},
        }

    async def delegate_task(
        self,
        task_description: str,
        required_capabilities: List[AgentCapability],
        delegation_pattern: str = "capability_based",
        priority: MessagePriority = MessagePriority.NORMAL,
        context_data: Optional[Dict[str, Any]] = None,
        fallback_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Delegate a task using the specified pattern.

        Args:
            task_description: Description of the task
            required_capabilities: Required agent capabilities
            delegation_pattern: Delegation pattern to use
            priority: Task priority
            context_data: Additional context data
            fallback_patterns: Fallback patterns if primary fails

        Returns:
            Delegation result
        """
        start_time = datetime.utcnow()
        delegation_id = str(uuid.uuid4())

        self.logger.info(
            f"Delegating task {delegation_id} using pattern: {delegation_pattern}"
        )

        # Record delegation attempt
        delegation_record = {
            "delegation_id": delegation_id,
            "task_description": task_description,
            "required_capabilities": [cap.value for cap in required_capabilities],
            "pattern": delegation_pattern,
            "priority": priority.value,
            "start_time": start_time,
            "context_data": context_data or {},
        }

        self.stats["delegations_made"] += 1
        self.stats["pattern_usage"][delegation_pattern] += 1

        try:
            # Try primary delegation pattern
            if delegation_pattern in self.delegation_patterns:
                result = await self.delegation_patterns[delegation_pattern](
                    task_description, required_capabilities, priority, context_data
                )

                if result.get("success", False):
                    delegation_record.update(
                        {
                            "success": True,
                            "assigned_agents": result.get("assigned_agents", []),
                            "end_time": datetime.utcnow(),
                        }
                    )
                    self.stats["successful_delegations"] += 1
                    self.delegation_history.append(delegation_record)
                    return result

            # Try fallback patterns
            if fallback_patterns:
                for fallback_pattern in fallback_patterns:
                    if fallback_pattern in self.delegation_patterns:
                        self.logger.info(f"Trying fallback pattern: {fallback_pattern}")
                        result = await self.delegation_patterns[fallback_pattern](
                            task_description,
                            required_capabilities,
                            priority,
                            context_data,
                        )

                        if result.get("success", False):
                            delegation_record.update(
                                {
                                    "success": True,
                                    "assigned_agents": result.get(
                                        "assigned_agents", []
                                    ),
                                    "fallback_pattern": fallback_pattern,
                                    "end_time": datetime.utcnow(),
                                }
                            )
                            self.stats["successful_delegations"] += 1
                            self.delegation_history.append(delegation_record)
                            return result

            # All patterns failed
            delegation_record.update(
                {
                    "success": False,
                    "error": "No suitable agents found with any pattern",
                    "end_time": datetime.utcnow(),
                }
            )
            self.stats["failed_delegations"] += 1
            self.delegation_history.append(delegation_record)

            return {
                "success": False,
                "error": "Delegation failed with all patterns",
                "delegation_id": delegation_id,
            }

        except Exception as e:
            self.logger.error(f"Delegation error: {e}")
            delegation_record.update(
                {"success": False, "error": str(e), "end_time": datetime.utcnow()}
            )
            self.stats["failed_delegations"] += 1
            self.delegation_history.append(delegation_record)

            return {
                "success": False,
                "error": f"Delegation exception: {str(e)}",
                "delegation_id": delegation_id,
            }

    async def _delegate_by_capability(
        self,
        task_description: str,
        required_capabilities: List[AgentCapability],
        priority: MessagePriority,
        context_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Delegate based on agent capabilities."""
        suitable_agents = []

        for agent_id, agent in self.agent_registry.items():
            if any(cap in agent.capabilities for cap in required_capabilities):
                suitable_agents.append(agent_id)

        if not suitable_agents:
            return {"success": False, "error": "No agents with required capabilities"}

        # Select best agent (simplified - could be more sophisticated)
        selected_agent = suitable_agents[0]

        # Send task to agent
        comm_id = await self.communication_hub.send_message(
            from_agent="delegation_manager",
            to_agent=selected_agent,
            content=task_description,
            channel=CommunicationChannel.COORDINATION,
            communication_type=CommunicationType.REQUEST_RESPONSE,
            priority=priority,
            subject="Task Delegation",
            data_payload=context_data or {},
            expects_reply=True,
        )

        return {
            "success": True,
            "assigned_agents": [selected_agent],
            "communication_id": comm_id,
            "delegation_method": "capability_based",
        }

    async def _delegate_by_load(
        self,
        task_description: str,
        required_capabilities: List[AgentCapability],
        priority: MessagePriority,
        context_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Delegate based on agent load balancing."""
        # Find capable agents
        capable_agents = []
        for agent_id, agent in self.agent_registry.items():
            if any(cap in agent.capabilities for cap in required_capabilities):
                current_load = self._get_agent_load(agent_id)
                capable_agents.append((agent_id, current_load))

        if not capable_agents:
            return {"success": False, "error": "No capable agents available"}

        # Select agent with lowest load
        capable_agents.sort(key=lambda x: x[1])
        selected_agent = capable_agents[0][0]

        comm_id = await self.communication_hub.send_message(
            from_agent="delegation_manager",
            to_agent=selected_agent,
            content=task_description,
            channel=CommunicationChannel.COORDINATION,
            communication_type=CommunicationType.REQUEST_RESPONSE,
            priority=priority,
            subject="Load-Balanced Task Delegation",
            data_payload=context_data or {},
            expects_reply=True,
        )

        return {
            "success": True,
            "assigned_agents": [selected_agent],
            "communication_id": comm_id,
            "delegation_method": "load_balanced",
            "selected_load": capable_agents[0][1],
        }

    async def _delegate_by_expertise(
        self,
        task_description: str,
        required_capabilities: List[AgentCapability],
        priority: MessagePriority,
        context_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Delegate based on agent expertise and performance history."""
        expert_agents = []

        for agent_id, agent in self.agent_registry.items():
            if any(cap in agent.capabilities for cap in required_capabilities):
                expertise_score = self._calculate_expertise_score(
                    agent_id, required_capabilities
                )
                expert_agents.append((agent_id, expertise_score))

        if not expert_agents:
            return {"success": False, "error": "No expert agents available"}

        # Select most expert agent
        expert_agents.sort(key=lambda x: x[1], reverse=True)
        selected_agent = expert_agents[0][0]

        comm_id = await self.communication_hub.send_message(
            from_agent="delegation_manager",
            to_agent=selected_agent,
            content=task_description,
            channel=CommunicationChannel.COORDINATION,
            communication_type=CommunicationType.REQUEST_RESPONSE,
            priority=priority,
            subject="Expertise-Based Task Delegation",
            data_payload=context_data or {},
            expects_reply=True,
        )

        return {
            "success": True,
            "assigned_agents": [selected_agent],
            "communication_id": comm_id,
            "delegation_method": "expertise_weighted",
            "expertise_score": expert_agents[0][1],
        }

    async def _delegate_collaborative(
        self,
        task_description: str,
        required_capabilities: List[AgentCapability],
        priority: MessagePriority,
        context_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Delegate as a collaborative task requiring multiple agents."""
        # Find multiple agents with complementary capabilities
        selected_agents = []
        covered_capabilities = set()

        for agent_id, agent in self.agent_registry.items():
            agent_caps = set(agent.capabilities)
            if (
                agent_caps.intersection(set(required_capabilities))
                - covered_capabilities
            ):
                selected_agents.append(agent_id)
                covered_capabilities.update(agent_caps)

                # Stop when all capabilities are covered
                if covered_capabilities.issuperset(set(required_capabilities)):
                    break

        if not selected_agents:
            return {"success": False, "error": "No collaborative agents available"}

        # Initiate collaboration session
        session_id = await self.communication_hub.request_collaboration(
            initiator_agent="delegation_manager",
            target_agents=selected_agents,
            task_description=task_description,
            collaboration_type="task_completion",
            context_data=context_data,
        )

        return {
            "success": True,
            "assigned_agents": selected_agents,
            "collaboration_session": session_id,
            "delegation_method": "collaborative",
        }

    async def _delegate_hierarchical(
        self,
        task_description: str,
        required_capabilities: List[AgentCapability],
        priority: MessagePriority,
        context_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Delegate using hierarchical task breakdown."""
        # This is a simplified hierarchical delegation
        # In practice, this would involve task decomposition

        # Find a lead agent
        lead_agent = None
        for agent_id, agent in self.agent_registry.items():
            if AgentCapability.TASK_PLANNING in agent.capabilities:
                lead_agent = agent_id
                break

        if not lead_agent:
            # Fallback to regular capability-based delegation
            return await self._delegate_by_capability(
                task_description, required_capabilities, priority, context_data
            )

        # Send task to lead agent for further delegation
        comm_id = await self.communication_hub.send_message(
            from_agent="delegation_manager",
            to_agent=lead_agent,
            content=f"Lead this hierarchical task: {task_description}",
            channel=CommunicationChannel.COORDINATION,
            communication_type=CommunicationType.REQUEST_RESPONSE,
            priority=priority,
            subject="Hierarchical Task Leadership",
            data_payload={
                **(context_data or {}),
                "delegation_type": "hierarchical",
                "required_capabilities": [cap.value for cap in required_capabilities],
            },
            expects_reply=True,
        )

        return {
            "success": True,
            "assigned_agents": [lead_agent],
            "communication_id": comm_id,
            "delegation_method": "hierarchical",
            "lead_agent": lead_agent,
        }

    async def _delegate_consensus(
        self,
        task_description: str,
        required_capabilities: List[AgentCapability],
        priority: MessagePriority,
        context_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Delegate task requiring consensus from multiple agents."""
        # Find all capable agents
        capable_agents = []
        for agent_id, agent in self.agent_registry.items():
            if any(cap in agent.capabilities for cap in required_capabilities):
                capable_agents.append(agent_id)

        if len(capable_agents) < 2:
            return {"success": False, "error": "Need at least 2 agents for consensus"}

        # Limit to reasonable number for consensus
        consensus_agents = (
            capable_agents[:5] if len(capable_agents) > 5 else capable_agents
        )

        # Start consensus collaboration
        session_id = await self.communication_hub.request_collaboration(
            initiator_agent="delegation_manager",
            target_agents=consensus_agents,
            task_description=f"Reach consensus on: {task_description}",
            collaboration_type="consensus_building",
            context_data=context_data,
        )

        return {
            "success": True,
            "assigned_agents": consensus_agents,
            "collaboration_session": session_id,
            "delegation_method": "consensus",
        }

    def _get_agent_load(self, agent_id: str) -> float:
        """Get current load for an agent (placeholder implementation)."""
        # In practice, this would check actual agent load
        performance = self.agent_performance.get(agent_id, {})
        return performance.get("current_load", 0.0)

    def _calculate_expertise_score(
        self, agent_id: str, capabilities: List[AgentCapability]
    ) -> float:
        """Calculate expertise score for an agent on specific capabilities."""
        performance = self.agent_performance.get(agent_id, {})

        # Base score from capability match
        agent = self.agent_registry.get(agent_id)
        if not agent:
            return 0.0

        capability_match = len(set(capabilities).intersection(set(agent.capabilities)))
        base_score = capability_match / len(capabilities) if capabilities else 0

        # Adjust based on historical performance
        success_rate = performance.get("success_rate", 0.5)
        response_time_factor = 1.0 / (performance.get("avg_response_time", 10.0) + 1)

        expertise_score = base_score * success_rate * response_time_factor
        return min(expertise_score, 1.0)  # Cap at 1.0

    def update_agent_performance(
        self,
        agent_id: str,
        task_success: bool,
        response_time: float,
        quality_score: Optional[float] = None,
    ) -> None:
        """Update agent performance metrics."""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "success_rate": 0.0,
                "total_response_time": 0.0,
                "avg_response_time": 0.0,
                "quality_scores": [],
            }

        perf = self.agent_performance[agent_id]
        perf["total_tasks"] += 1

        if task_success:
            perf["successful_tasks"] += 1

        perf["success_rate"] = perf["successful_tasks"] / perf["total_tasks"]
        perf["total_response_time"] += response_time
        perf["avg_response_time"] = perf["total_response_time"] / perf["total_tasks"]

        if quality_score is not None:
            perf["quality_scores"].append(quality_score)
            # Keep only recent scores
            if len(perf["quality_scores"]) > 100:
                perf["quality_scores"] = perf["quality_scores"][-80:]

    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get delegation manager statistics."""
        return {
            **self.stats,
            "tracked_agents": len(self.agent_performance),
            "delegation_history_count": len(self.delegation_history),
            "agent_performance_summary": {
                agent_id: {
                    "success_rate": perf["success_rate"],
                    "avg_response_time": perf["avg_response_time"],
                    "total_tasks": perf["total_tasks"],
                }
                for agent_id, perf in self.agent_performance.items()
            },
        }

    async def cleanup(self) -> None:
        """Cleanup delegation manager resources."""
        self.logger.info("Cleaning up delegation manager")

        self.agent_performance.clear()
        self.delegation_history.clear()

        self.logger.info("Delegation manager cleanup complete")
