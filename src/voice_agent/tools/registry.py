"""
Tool registry for the voice agent.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .base import SimpleTool, Tool


class ToolRegistry:
    """
    Central registry for managing voice agent tools.

    Provides:
    - Tool registration and discovery
    - Tool lifecycle management
    - Tool metadata access
    """

    def __init__(self):
        """Initialize the tool registry."""
        self.logger = logging.getLogger(__name__)
        self._tools: Dict[str, Tool] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool instance.

        Args:
            tool: Tool instance to register
        """
        if not isinstance(tool, Tool):
            raise TypeError("Tool must be an instance of Tool class")

        tool_name = tool.name

        if tool_name in self._tools:
            self.logger.warning(f"Tool '{tool_name}' already registered, overriding")

        self._tools[tool_name] = tool
        self._tool_metadata[tool_name] = tool.get_info()

        self.logger.info(f"Registered tool: {tool_name}")

    def register_function(
        self, name: str, func: Callable, description: str = "", parameters_schema=None
    ) -> None:
        """
        Register a function as a tool.

        Args:
            name: Tool name
            func: Function to register
            description: Tool description
            parameters_schema: Pydantic model for parameter validation
        """
        tool = SimpleTool(
            name=name,
            description=description or func.__doc__ or "No description",
            func=func,
            parameters_schema=parameters_schema,
        )

        self.register_tool(tool)

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            del self._tool_metadata[name]
            self.logger.info(f"Unregistered tool: {name}")
            return True

        return False

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool.

        Args:
            name: Tool name

        Returns:
            Tool information dictionary or None if not found
        """
        return self._tool_metadata.get(name)

    def get_all_tools_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered tools.

        Returns:
            List of tool information dictionaries
        """
        return list(self._tool_metadata.values())

    def search_tools(self, query: str) -> List[str]:
        """
        Search for tools by name or description.

        Args:
            query: Search query

        Returns:
            List of matching tool names
        """
        query = query.lower()
        matching_tools = []

        for name, metadata in self._tool_metadata.items():
            if (
                query in name.lower()
                or query in metadata.get("description", "").lower()
            ):
                matching_tools.append(name)

        return matching_tools

    async def initialize_all_tools(self) -> None:
        """Initialize all registered tools."""
        self.logger.info("Initializing all registered tools...")

        for name, tool in self._tools.items():
            try:
                if hasattr(tool, "initialize"):
                    await tool.initialize()
                    self.logger.debug(f"Initialized tool: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize tool '{name}': {e}")

        self.logger.info("Tool initialization complete")

    async def cleanup_all_tools(self) -> None:
        """Cleanup all registered tools."""
        self.logger.info("Cleaning up all registered tools...")

        for name, tool in self._tools.items():
            try:
                if hasattr(tool, "cleanup"):
                    await tool.cleanup()
                    self.logger.debug(f"Cleaned up tool: {name}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup tool '{name}': {e}")

        self.logger.info("Tool cleanup complete")

    def get_tools_by_category(self, category: str) -> List[str]:
        """
        Get tools by category (if tools have category metadata).

        Args:
            category: Category name

        Returns:
            List of tool names in the category
        """
        matching_tools = []

        for name, metadata in self._tool_metadata.items():
            tool_categories = metadata.get("categories", [])
            if category.lower() in [cat.lower() for cat in tool_categories]:
                matching_tools.append(name)

        return matching_tools

    def validate_tool_parameters(
        self, name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate parameters for a tool.

        Args:
            name: Tool name
            parameters: Parameters to validate

        Returns:
            Validated parameters

        Raises:
            ValueError: If tool not found or parameters invalid
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        return tool.validate_parameters(parameters)


# Global tool registry instance
_global_registry = ToolRegistry()


def register_tool(tool: Tool) -> None:
    """
    Register a tool in the global registry.

    Args:
        tool: Tool instance to register
    """
    _global_registry.register_tool(tool)


def register_function(
    name: str, func: Callable, description: str = "", parameters_schema=None
) -> None:
    """
    Register a function as a tool in the global registry.

    Args:
        name: Tool name
        func: Function to register
        description: Tool description
        parameters_schema: Pydantic model for parameter validation
    """
    _global_registry.register_function(name, func, description, parameters_schema)


def tool(name: str, description: str = "", parameters=None):
    """
    Decorator to register a function as a tool.

    Args:
        name: Tool name
        description: Tool description
        parameters: Pydantic model for parameter validation

    Returns:
        Decorated function
    """

    def decorator(func):
        register_function(name, func, description, parameters)
        return func

    return decorator


def get_tool(name: str) -> Optional[Tool]:
    """
    Get a tool from the global registry.

    Args:
        name: Tool name

    Returns:
        Tool instance or None if not found
    """
    return _global_registry.get_tool(name)


def list_tools() -> List[str]:
    """
    List all registered tools in the global registry.

    Returns:
        List of tool names
    """
    return _global_registry.list_tools()


def get_tool_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get tool information from the global registry.

    Args:
        name: Tool name

    Returns:
        Tool information dictionary or None if not found
    """
    return _global_registry.get_tool_info(name)


def get_all_tools_info() -> List[Dict[str, Any]]:
    """
    Get information about all tools in the global registry.

    Returns:
        List of tool information dictionaries
    """
    return _global_registry.get_all_tools_info()


def search_tools(query: str) -> List[str]:
    """
    Search for tools in the global registry.

    Args:
        query: Search query

    Returns:
        List of matching tool names
    """
    return _global_registry.search_tools(query)


# Expose the global registry for advanced usage
def get_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.

    Returns:
        Global ToolRegistry instance
    """
    return _global_registry
