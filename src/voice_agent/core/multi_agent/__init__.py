"""
Multi-agent framework for the voice agent.

This module provides a comprehensive multi-agent system built on LlamaIndex,
enabling specialized agents to handle different types of queries and tasks.

Key components:
- Message system for inter-agent communication
- Base agent framework with LlamaIndex ReAct integration
- Hybrid routing system (rules + embeddings + LLM fallback)
- Tool adapter for seamless integration with existing tools
- Shared context manager for conversation history
- Configuration extensions for multi-agent setup
"""

from .message import AgentMessage, AgentResponse, MessageType, MessageStatus
from .agent_base import (
    AgentBase,
    AgentCapability,
    GeneralAgent,
    ToolSpecialistAgent,
    InformationAgent,
)
from .router import AgentRouter, RouteDecision, RoutingStrategy
from .context import SharedContextManager, ConversationSlice
from .tool_adapter import ToolAdapter

__all__ = [
    "AgentMessage",
    "AgentResponse",
    "MessageType",
    "MessageStatus",
    "AgentBase",
    "AgentCapability",
    "GeneralAgent",
    "ToolSpecialistAgent",
    "InformationAgent",
    "AgentRouter",
    "RouteDecision",
    "RoutingStrategy",
    "SharedContextManager",
    "ConversationSlice",
    "ToolAdapter",
]
