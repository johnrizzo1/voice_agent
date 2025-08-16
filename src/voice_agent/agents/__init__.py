"""
Agent implementations for the multi-agent voice system.

This module contains specialized agent implementations that extend
the base AgentBase class for specific use cases.
"""

from ..core.multi_agent.agent_base import (
    AgentBase,
    AgentCapability,
    AgentStatus,
    AgentConfig,
    GeneralAgent,
    ToolSpecialistAgent,
    InformationAgent,
)

from .utility_agent import UtilityAgent

from .productivity_agent import ProductivityAgent

__all__ = [
    "AgentBase",
    "AgentCapability",
    "AgentStatus",
    "AgentConfig",
    "GeneralAgent",
    "ToolSpecialistAgent",
    "InformationAgent",
    "ProductivityAgent",
    "UtilityAgent",
]
