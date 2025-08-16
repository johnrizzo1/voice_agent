"""
Shared context manager for multi-agent conversations.

Manages conversation history, context slicing, and inter-agent context sharing
to maintain coherent conversations across agent handoffs.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object
    Field = lambda **kwargs: None

from .message import AgentMessage, AgentResponse


class ConversationSlice(BaseModel if BaseModel != object else dict):
    """A slice of conversation history relevant to a specific context."""

    if BaseModel != object:
        slice_id: str
        conversation_id: str
        agent_id: Optional[str] = None
        start_time: datetime
        end_time: Optional[datetime] = None
        messages: List[Dict[str, Any]] = Field(default_factory=list)
        context_metadata: Dict[str, Any] = Field(default_factory=dict)
        token_count: int = 0
        relevance_score: float = 1.0


class ContextWindow:
    """Manages a sliding window of conversation context."""

    def __init__(
        self, max_tokens: int = 4000, max_messages: int = 50, max_age_hours: int = 24
    ):
        """
        Initialize context window.

        Args:
            max_tokens: Maximum tokens to keep in context
            max_messages: Maximum number of messages to keep
            max_age_hours: Maximum age of messages in hours
        """
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.max_age_hours = max_age_hours

        self.messages: List[Dict[str, Any]] = []
        self.current_tokens = 0

    def add_message(self, message_dict: Dict[str, Any]) -> None:
        """Add a message to the context window."""
        # Estimate token count (rough approximation: 4 chars = 1 token)
        content = message_dict.get("content", "")
        estimated_tokens = len(content) // 4

        message_dict["estimated_tokens"] = estimated_tokens
        message_dict["timestamp"] = message_dict.get(
            "timestamp", datetime.utcnow().isoformat()
        )

        self.messages.append(message_dict)
        self.current_tokens += estimated_tokens

        # Trim if necessary
        self._trim_context()

    def _trim_context(self) -> None:
        """Trim context to stay within limits."""
        now = datetime.utcnow()
        cutoff_time = now - timedelta(hours=self.max_age_hours)

        # Remove old messages first
        self.messages = [
            msg
            for msg in self.messages
            if datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
            > cutoff_time
        ]

        # Trim by message count
        if len(self.messages) > self.max_messages:
            removed_count = len(self.messages) - self.max_messages
            self.messages = self.messages[removed_count:]

        # Trim by token count (remove oldest messages)
        while self.current_tokens > self.max_tokens and self.messages:
            removed_msg = self.messages.pop(0)
            self.current_tokens -= removed_msg.get("estimated_tokens", 0)

        # Recalculate token count
        self.current_tokens = sum(
            msg.get("estimated_tokens", 0) for msg in self.messages
        )

    def get_context_text(self, include_system: bool = True) -> str:
        """Get formatted context text for LLM consumption."""
        context_parts = []

        for msg in self.messages:
            msg_type = msg.get("type", "unknown")
            content = msg.get("content", "")
            agent_id = msg.get("agent_id")

            if msg_type == "user":
                context_parts.append(f"User: {content}")
            elif msg_type == "agent":
                agent_label = f" ({agent_id})" if agent_id else ""
                context_parts.append(f"Assistant{agent_label}: {content}")
            elif msg_type == "system" and include_system:
                context_parts.append(f"System: {content}")
            elif msg_type == "tool":
                tool_name = msg.get("tool_name", "unknown")
                context_parts.append(f"Tool ({tool_name}): {content}")

        return "\n".join(context_parts)

    def get_recent_messages(self, count: int) -> List[Dict[str, Any]]:
        """Get the most recent N messages."""
        return (
            self.messages[-count:]
            if count <= len(self.messages)
            else self.messages.copy()
        )

    def clear(self) -> None:
        """Clear all context."""
        self.messages.clear()
        self.current_tokens = 0


class SharedContextManager:
    """
    Manages shared context across multiple agents in conversations.

    Provides conversation history slicing, context sharing between agents,
    and intelligent context window management for optimal LLM performance.
    """

    def __init__(
        self,
        max_conversations: int = 100,
        default_context_window: int = 4000,
        max_slice_age_hours: int = 24,
        enable_rag_integration: bool = False,
    ):
        """
        Initialize the shared context manager.

        Args:
            max_conversations: Maximum number of active conversations to track
            default_context_window: Default context window size in tokens
            max_slice_age_hours: Maximum age for context slices in hours
            enable_rag_integration: Enable RAG integration for context retrieval
        """
        self.max_conversations = max_conversations
        self.default_context_window = default_context_window
        self.max_slice_age_hours = max_slice_age_hours
        self.enable_rag_integration = enable_rag_integration
        self.logger = logging.getLogger(__name__)

        # Active conversations and their context windows
        self.conversation_contexts: Dict[str, ContextWindow] = {}

        # Agent-specific context slices
        self.agent_contexts: Dict[str, Dict[str, ConversationSlice]] = {}

        # Cross-conversation context (for RAG integration)
        self.global_context_store: List[Dict[str, Any]] = []

        # Context sharing relationships (which agents share context)
        self.context_sharing_groups: Dict[str, Set[str]] = {}

        # Performance tracking
        self.context_stats: Dict[str, int] = {
            "conversations_created": 0,
            "context_slices_created": 0,
            "context_retrievals": 0,
            "context_cleanups": 0,
        }

    def create_conversation_context(
        self, conversation_id: str, context_window_size: Optional[int] = None
    ) -> ContextWindow:
        """
        Create a new conversation context.

        Args:
            conversation_id: Unique conversation identifier
            context_window_size: Custom context window size

        Returns:
            Created context window
        """
        if conversation_id in self.conversation_contexts:
            self.logger.warning(
                f"Conversation context {conversation_id} already exists"
            )
            return self.conversation_contexts[conversation_id]

        window_size = context_window_size or self.default_context_window
        context_window = ContextWindow(max_tokens=window_size)

        self.conversation_contexts[conversation_id] = context_window
        self.context_stats["conversations_created"] += 1

        self.logger.info(
            f"Created conversation context {conversation_id} with {window_size} token window"
        )

        # Clean up old conversations if needed
        if len(self.conversation_contexts) > self.max_conversations:
            self._cleanup_old_conversations()

        return context_window

    def get_conversation_context(self, conversation_id: str) -> Optional[ContextWindow]:
        """Get conversation context by ID."""
        return self.conversation_contexts.get(conversation_id)

    def add_message_to_context(
        self, conversation_id: str, message: AgentMessage
    ) -> None:
        """
        Add a message to conversation context.

        Args:
            conversation_id: Conversation identifier
            message: Message to add
        """
        context_window = self.conversation_contexts.get(conversation_id)
        if not context_window:
            context_window = self.create_conversation_context(conversation_id)

        message_dict = {
            "id": message.id,
            "type": "user" if message.type.value == "user_input" else "system",
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "agent_id": message.from_agent,
            "metadata": message.metadata,
        }

        context_window.add_message(message_dict)

        # Add to global context store for RAG
        if self.enable_rag_integration:
            self.global_context_store.append(message_dict)
            # Trim global store if it gets too large
            if len(self.global_context_store) > 1000:
                self.global_context_store = self.global_context_store[-800:]

    def add_response_to_context(
        self, conversation_id: str, response: AgentResponse
    ) -> None:
        """
        Add an agent response to conversation context.

        Args:
            conversation_id: Conversation identifier
            response: Agent response to add
        """
        context_window = self.conversation_contexts.get(conversation_id)
        if not context_window:
            context_window = self.create_conversation_context(conversation_id)

        response_dict = {
            "id": response.id,
            "type": "agent",
            "content": response.content,
            "timestamp": response.timestamp.isoformat(),
            "agent_id": response.agent_id,
            "metadata": {
                **response.metadata,
                "tool_calls": len(response.tool_calls),
                "confidence": response.confidence_score,
            },
        }

        context_window.add_message(response_dict)

        # Add tool calls as separate context entries
        for tool_call in response.tool_calls:
            tool_dict = {
                "id": tool_call.get("id", "unknown"),
                "type": "tool",
                "content": f"Called {tool_call.get('tool_name', 'unknown')} with parameters: {tool_call.get('parameters', {})}",
                "timestamp": tool_call.get("timestamp", response.timestamp.isoformat()),
                "tool_name": tool_call.get("tool_name"),
                "metadata": {"tool_call": True},
            }
            context_window.add_message(tool_dict)

        # Add to global context store
        if self.enable_rag_integration:
            self.global_context_store.append(response_dict)

    def create_context_slice_for_agent(
        self,
        agent_id: str,
        conversation_id: str,
        slice_type: str = "general",
        max_messages: int = 20,
    ) -> ConversationSlice:
        """
        Create a context slice tailored for a specific agent.

        Args:
            agent_id: Target agent identifier
            conversation_id: Source conversation
            slice_type: Type of slice (general, tool_focused, etc.)
            max_messages: Maximum messages to include

        Returns:
            Created context slice
        """
        context_window = self.conversation_contexts.get(conversation_id)
        if not context_window:
            # Create empty slice if no context exists
            slice_dict = {
                "slice_id": f"{agent_id}_{conversation_id}_{slice_type}",
                "conversation_id": conversation_id,
                "agent_id": agent_id,
                "start_time": datetime.utcnow(),
                "messages": [],
                "context_metadata": {"slice_type": slice_type},
                "token_count": 0,
                "relevance_score": 0.0,
            }
        else:
            # Get relevant messages based on slice type
            relevant_messages = self._filter_messages_for_agent(
                context_window.get_recent_messages(max_messages), agent_id, slice_type
            )

            token_count = sum(
                msg.get("estimated_tokens", 0) for msg in relevant_messages
            )

            slice_dict = {
                "slice_id": f"{agent_id}_{conversation_id}_{slice_type}",
                "conversation_id": conversation_id,
                "agent_id": agent_id,
                "start_time": datetime.utcnow(),
                "messages": relevant_messages,
                "context_metadata": {
                    "slice_type": slice_type,
                    "original_message_count": len(context_window.messages),
                    "filtered_message_count": len(relevant_messages),
                },
                "token_count": token_count,
                "relevance_score": self._calculate_relevance_score(
                    relevant_messages, agent_id
                ),
            }

        # Create ConversationSlice object
        context_slice = (
            ConversationSlice(**slice_dict)
            if hasattr(ConversationSlice, "slice_id")
            else slice_dict
        )

        # Store in agent contexts
        if agent_id not in self.agent_contexts:
            self.agent_contexts[agent_id] = {}
        self.agent_contexts[agent_id][conversation_id] = context_slice

        self.context_stats["context_slices_created"] += 1

        self.logger.debug(
            f"Created context slice for agent {agent_id}: {len(relevant_messages)} messages, {token_count} tokens"
        )

        return context_slice

    def _filter_messages_for_agent(
        self, messages: List[Dict[str, Any]], agent_id: str, slice_type: str
    ) -> List[Dict[str, Any]]:
        """Filter messages based on agent capabilities and slice type."""
        if slice_type == "tool_focused":
            # Include tool-related messages and recent context
            relevant_messages = []
            for msg in messages:
                if (
                    msg.get("type") == "tool"
                    or msg.get("metadata", {}).get("tool_call")
                    or "tool" in msg.get("content", "").lower()
                    or "calculate" in msg.get("content", "").lower()
                    or "search" in msg.get("content", "").lower()
                ):
                    relevant_messages.append(msg)
                elif (
                    msg.get("type") == "user"
                ):  # Always include user messages for context
                    relevant_messages.append(msg)

            # Add recent agent responses for continuity
            for msg in messages[-5:]:
                if msg.get("type") == "agent" and msg not in relevant_messages:
                    relevant_messages.append(msg)

            return sorted(relevant_messages, key=lambda x: x.get("timestamp", ""))

        elif slice_type == "handoff":
            # Include recent conversation flow for agent handoffs
            return messages[-10:]  # Last 10 messages for context continuity

        else:  # general slice
            # Include all message types with slight preference for recent messages
            return messages

    def _calculate_relevance_score(
        self, messages: List[Dict[str, Any]], agent_id: str
    ) -> float:
        """Calculate relevance score for a context slice."""
        if not messages:
            return 0.0

        relevance_factors = []

        # Factor 1: Recency (more recent = higher score)
        now = datetime.utcnow()
        avg_age_hours = 0
        valid_timestamps = 0

        for msg in messages:
            try:
                msg_time = datetime.fromisoformat(
                    msg["timestamp"].replace("Z", "+00:00")
                )
                age_hours = (now - msg_time).total_seconds() / 3600
                avg_age_hours += age_hours
                valid_timestamps += 1
            except (ValueError, KeyError):
                continue

        if valid_timestamps > 0:
            avg_age_hours /= valid_timestamps
            recency_score = max(
                0.0, 1.0 - (avg_age_hours / 24.0)
            )  # Decay over 24 hours
            relevance_factors.append(recency_score)

        # Factor 2: Message type diversity
        message_types = set(msg.get("type", "unknown") for msg in messages)
        diversity_score = min(1.0, len(message_types) / 4.0)  # Normalize to max 4 types
        relevance_factors.append(diversity_score)

        # Factor 3: Content richness (longer messages generally more informative)
        avg_length = sum(len(msg.get("content", "")) for msg in messages) / len(
            messages
        )
        length_score = min(1.0, avg_length / 200.0)  # Normalize to 200 chars
        relevance_factors.append(length_score)

        # Calculate weighted average
        weights = [0.4, 0.3, 0.3]  # Prefer recency, then diversity, then length
        if len(relevance_factors) == len(weights):
            return sum(
                score * weight for score, weight in zip(relevance_factors, weights)
            )
        else:
            return (
                sum(relevance_factors) / len(relevance_factors)
                if relevance_factors
                else 0.0
            )

    def get_context_for_agent(
        self, agent_id: str, conversation_id: str, slice_type: str = "general"
    ) -> Optional[ConversationSlice]:
        """
        Get existing context slice for an agent or create new one.

        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            slice_type: Type of context slice

        Returns:
            Context slice for the agent
        """
        self.context_stats["context_retrievals"] += 1

        # Check if we have an existing slice
        if (
            agent_id in self.agent_contexts
            and conversation_id in self.agent_contexts[agent_id]
        ):

            existing_slice = self.agent_contexts[agent_id][conversation_id]

            # Check if slice is still fresh (less than 1 hour old)
            slice_age = (
                datetime.utcnow() - existing_slice.start_time
                if hasattr(existing_slice, "start_time")
                else datetime.utcnow()
                - datetime.fromisoformat(existing_slice["start_time"])
            )
            if slice_age.total_seconds() < 3600:  # 1 hour
                return existing_slice

        # Create new slice
        return self.create_context_slice_for_agent(
            agent_id, conversation_id, slice_type
        )

    def share_context_between_agents(
        self, from_agent: str, to_agent: str, conversation_id: str
    ) -> bool:
        """
        Share context from one agent to another during handoffs.

        Args:
            from_agent: Source agent identifier
            to_agent: Target agent identifier
            conversation_id: Conversation identifier

        Returns:
            True if context was shared successfully
        """
        try:
            # Get source agent's context
            source_slice = self.get_context_for_agent(from_agent, conversation_id)
            if not source_slice:
                return False

            # Create enhanced context slice for target agent
            enhanced_slice = self.create_context_slice_for_agent(
                to_agent, conversation_id, "handoff"
            )

            # Add handoff metadata
            handoff_metadata = {
                "handoff_from": from_agent,
                "handoff_time": datetime.utcnow().isoformat(),
                "source_slice_id": (
                    source_slice.slice_id
                    if hasattr(source_slice, "slice_id")
                    else source_slice["slice_id"]
                ),
            }

            if hasattr(enhanced_slice, "context_metadata"):
                enhanced_slice.context_metadata.update(handoff_metadata)
            else:
                enhanced_slice["context_metadata"].update(handoff_metadata)

            self.logger.info(
                f"Shared context from {from_agent} to {to_agent} for conversation {conversation_id}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to share context between agents: {e}")
            return False

    def _cleanup_old_conversations(self) -> None:
        """Clean up old conversation contexts to free memory."""
        if len(self.conversation_contexts) <= self.max_conversations:
            return

        # Sort conversations by last activity (approximate via newest message timestamp)
        conversations_by_activity = []

        for conv_id, context_window in self.conversation_contexts.items():
            last_activity = datetime.min
            if context_window.messages:
                try:
                    last_msg_time = context_window.messages[-1]["timestamp"]
                    last_activity = datetime.fromisoformat(
                        last_msg_time.replace("Z", "+00:00")
                    )
                except (ValueError, KeyError):
                    pass

            conversations_by_activity.append((conv_id, last_activity))

        # Sort by activity (oldest first)
        conversations_by_activity.sort(key=lambda x: x[1])

        # Remove oldest conversations
        conversations_to_remove = conversations_by_activity[
            : len(conversations_by_activity) - self.max_conversations + 10
        ]

        for conv_id, _ in conversations_to_remove:
            # Clean up conversation context
            del self.conversation_contexts[conv_id]

            # Clean up agent contexts for this conversation
            for agent_contexts in self.agent_contexts.values():
                agent_contexts.pop(conv_id, None)

            self.context_stats["context_cleanups"] += 1

        self.logger.info(
            f"Cleaned up {len(conversations_to_remove)} old conversation contexts"
        )

    async def cleanup_expired_contexts(self) -> int:
        """Clean up expired context slices and return count of cleaned items."""
        cleaned_count = 0
        cutoff_time = datetime.utcnow() - timedelta(hours=self.max_slice_age_hours)

        # Clean up agent contexts
        for agent_id in list(self.agent_contexts.keys()):
            agent_slices = self.agent_contexts[agent_id]

            for conv_id in list(agent_slices.keys()):
                slice_obj = agent_slices[conv_id]
                slice_time = (
                    slice_obj.start_time
                    if hasattr(slice_obj, "start_time")
                    else datetime.fromisoformat(slice_obj["start_time"])
                )

                if slice_time < cutoff_time:
                    del agent_slices[conv_id]
                    cleaned_count += 1

            # Remove empty agent contexts
            if not agent_slices:
                del self.agent_contexts[agent_id]

        # Clean up global context store
        if self.enable_rag_integration:
            original_count = len(self.global_context_store)
            self.global_context_store = [
                msg
                for msg in self.global_context_store
                if datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
                > cutoff_time
            ]
            cleaned_count += original_count - len(self.global_context_store)

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} expired context items")

        return cleaned_count

    def get_context_stats(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        return {
            **self.context_stats,
            "active_conversations": len(self.conversation_contexts),
            "agent_contexts": len(self.agent_contexts),
            "total_agent_slices": sum(
                len(slices) for slices in self.agent_contexts.values()
            ),
            "global_context_items": len(self.global_context_store),
            "context_sharing_groups": len(self.context_sharing_groups),
        }

    async def cleanup(self) -> None:
        """Cleanup all context manager resources."""
        self.logger.info("Cleaning up shared context manager")

        self.conversation_contexts.clear()
        self.agent_contexts.clear()
        self.global_context_store.clear()
        self.context_sharing_groups.clear()

        self.context_stats = {key: 0 for key in self.context_stats}

        self.logger.info("Context manager cleanup complete")
