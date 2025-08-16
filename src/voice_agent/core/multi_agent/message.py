"""
Message system for multi-agent communication.

Provides structured message passing between agents with routing,
correlation, and error handling capabilities.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages in the multi-agent system."""

    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    AGENT_HANDOFF = "agent_handoff"
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"
    SYSTEM_NOTIFICATION = "system_notification"
    ERROR = "error"


class MessageStatus(str, Enum):
    """Status of messages in the system."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROUTED = "routed"
    EXPIRED = "expired"


class AgentMessage(BaseModel):
    """
    Core message structure for multi-agent communication.

    Provides a standardized envelope for all inter-agent messages
    with routing, correlation, and metadata capabilities.
    """

    # Core identifiers
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Message metadata
    type: MessageType = MessageType.USER_INPUT
    status: MessageStatus = MessageStatus.PENDING
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Routing information
    from_agent: Optional[str] = None
    to_agent: Optional[str] = None
    requires_response: bool = True

    # Message content
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

    # Processing information
    processing_history: List[Dict[str, Any]] = Field(default_factory=list)
    error_info: Optional[Dict[str, Any]] = None

    # Timing constraints
    timeout_seconds: Optional[float] = None
    priority: int = Field(default=5, ge=1, le=10)  # 1=highest, 10=lowest

    def add_processing_step(
        self,
        agent_id: str,
        action: str,
        result: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a processing step to the message history."""
        step = {
            "agent_id": agent_id,
            "action": action,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.processing_history.append(step)

    def set_error(
        self,
        error_type: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set error information on the message."""
        self.status = MessageStatus.FAILED
        self.error_info = {
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {},
        }

    def is_expired(self) -> bool:
        """Check if the message has expired based on timeout."""
        if not self.timeout_seconds:
            return False

        elapsed = (datetime.utcnow() - self.timestamp).total_seconds()
        return elapsed > self.timeout_seconds

    def to_conversation_entry(self) -> Dict[str, Any]:
        """Convert message to conversation history entry format."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.type.value,
            "from_agent": self.from_agent,
            "content": self.content,
            "metadata": self.metadata,
        }


class AgentResponse(BaseModel):
    """
    Response structure for agent interactions.

    Standardized response format that includes the processed message,
    any generated content, tool calls, and routing decisions.
    """

    # Response identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str  # ID of the original request message
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Response content
    content: str = ""
    response_type: MessageType = MessageType.AGENT_RESPONSE

    # Processing results
    success: bool = True
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None

    # Tool interactions
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Routing and handoff
    should_handoff: bool = False
    suggested_agent: Optional[str] = None
    handoff_reason: Optional[str] = None

    # Context and metadata
    context_updates: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Confidence and quality metrics
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)

    def add_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Optional[Any] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Add a tool call to the response."""
        tool_call = {
            "id": str(uuid.uuid4()),
            "tool_name": tool_name,
            "parameters": parameters,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.tool_calls.append(tool_call)

        tool_result = {
            "call_id": tool_call["id"],
            "success": success,
            "result": result,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.tool_results.append(tool_result)

    def set_handoff(self, target_agent: str, reason: str) -> None:
        """Set agent handoff information."""
        self.should_handoff = True
        self.suggested_agent = target_agent
        self.handoff_reason = reason

    def to_message(self, conversation_id: str) -> AgentMessage:
        """Convert response back to a message for further processing."""
        return AgentMessage(
            correlation_id=self.request_id,
            conversation_id=conversation_id,
            type=self.response_type,
            status=MessageStatus.COMPLETED if self.success else MessageStatus.FAILED,
            from_agent=self.agent_id,
            content=self.content,
            metadata={
                **self.metadata,
                "tool_calls": len(self.tool_calls),
                "processing_time": self.processing_time_seconds,
                "confidence": self.confidence_score,
            },
            context=self.context_updates,
        )


class MessageRouter:
    """
    Utility class for message routing and correlation tracking.

    Provides helper methods for managing message flows and maintaining
    conversation context across agent handoffs.
    """

    def __init__(self):
        """Initialize the message router."""
        self.active_conversations: Dict[str, List[AgentMessage]] = {}
        self.message_correlations: Dict[str, List[str]] = {}

    def route_message(
        self,
        message: AgentMessage,
        target_agent: str,
        routing_reason: Optional[str] = None,
    ) -> AgentMessage:
        """
        Route a message to a specific agent.

        Args:
            message: Message to route
            target_agent: Target agent identifier
            routing_reason: Optional reason for routing

        Returns:
            Routed message with updated metadata
        """
        routed_message = message.model_copy(deep=True)
        routed_message.to_agent = target_agent
        routed_message.status = MessageStatus.ROUTED

        # Add routing information to processing history
        routing_metadata = {"routing_reason": routing_reason} if routing_reason else {}
        routed_message.add_processing_step(
            agent_id="router",
            action="route_message",
            result=f"routed to {target_agent}",
            metadata=routing_metadata,
        )

        # Track conversation
        if routed_message.conversation_id not in self.active_conversations:
            self.active_conversations[routed_message.conversation_id] = []
        self.active_conversations[routed_message.conversation_id].append(routed_message)

        # Track correlation
        if routed_message.correlation_id not in self.message_correlations:
            self.message_correlations[routed_message.correlation_id] = []
        self.message_correlations[routed_message.correlation_id].append(
            routed_message.id
        )

        return routed_message

    def get_conversation_history(self, conversation_id: str) -> List[AgentMessage]:
        """Get all messages in a conversation."""
        return self.active_conversations.get(conversation_id, [])

    def get_correlated_messages(self, correlation_id: str) -> List[AgentMessage]:
        """Get all messages with the same correlation ID."""
        message_ids = self.message_correlations.get(correlation_id, [])
        correlated_messages = []

        for conversation_messages in self.active_conversations.values():
            for message in conversation_messages:
                if message.id in message_ids:
                    correlated_messages.append(message)

        return correlated_messages

    def cleanup_expired_messages(self) -> int:
        """Remove expired messages and return count of cleaned up messages."""
        cleaned_count = 0

        for conversation_id in list(self.active_conversations.keys()):
            messages = self.active_conversations[conversation_id]
            active_messages = []

            for message in messages:
                if message.is_expired():
                    cleaned_count += 1
                    # Clean up correlation tracking
                    if message.correlation_id in self.message_correlations:
                        correlation_messages = self.message_correlations[
                            message.correlation_id
                        ]
                        if message.id in correlation_messages:
                            correlation_messages.remove(message.id)
                        if not correlation_messages:
                            del self.message_correlations[message.correlation_id]
                else:
                    active_messages.append(message)

            if active_messages:
                self.active_conversations[conversation_id] = active_messages
            else:
                del self.active_conversations[conversation_id]

        return cleaned_count
