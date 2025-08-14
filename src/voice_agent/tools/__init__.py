"""
Tool framework for the voice agent.
"""

from .base import Tool
from .registry import get_tool, list_tools, register_tool, tool

__all__ = ["Tool", "tool", "register_tool", "get_tool", "list_tools"]
