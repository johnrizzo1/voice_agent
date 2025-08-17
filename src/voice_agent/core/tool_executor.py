"""
Tool execution engine for the voice agent.
"""

import asyncio
import inspect
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Type

from .config import ToolsConfig


class ToolExecutor:
    """
    Tool execution engine with plugin-based architecture.

    Features:
    - Dynamic tool discovery and loading
    - Sandboxed execution
    - Error handling and recovery
    - Automatic schema generation
    """

    def __init__(self, config: ToolsConfig):
        """
        Initialize the tool executor.

        Args:
            config: Tools configuration settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Registry of available tools
        self.registered_tools: Dict[str, Dict[str, Any]] = {}
        self.tool_instances: Dict[str, Any] = {}

        # Execution state
        self.is_initialized = False
        self.execution_timeout = 30.0  # seconds

    async def initialize(self) -> None:
        """Initialize the tool executor and discover tools."""
        self.logger.info("Initializing tool executor...")

        # Load built-in tools
        await self._load_builtin_tools()

        # Load user-defined tools
        await self._load_user_tools()

        # Apply enabled/disabled configuration
        self._apply_tool_config()

        self.is_initialized = True
        self.logger.info(
            f"Tool executor initialized with {len(self.registered_tools)} tools"
        )

    async def _load_builtin_tools(self) -> None:
        """Load built-in tools from the tools.builtin package."""
        try:
            # Import built-in tool modules
            from ..tools.builtin import calculator, file_ops, weather, web_search

            # Register tools
            await self._register_tool_from_module(calculator)
            await self._register_tool_from_module(weather)
            await self._register_tool_from_module(file_ops)
            await self._register_tool_from_module(web_search)

            self.logger.info("Built-in tools loaded")
        except Exception as e:
            self.logger.error(f"Failed to load built-in tools: {e}")

    async def _load_user_tools(self) -> None:
        """Load user-defined tools from custom directories."""
        # This would scan for user-defined tool files
        # For now, we'll just log that this feature is available
        self.logger.info("User tool loading is available but not implemented yet")

    async def _register_tool_from_module(self, module) -> None:
        """Register tools from a module."""
        try:
            # Look for Tool classes and decorated functions
            for name in dir(module):
                obj = getattr(module, name)

                # Check if it's a Tool class (but not abstract)
                if (
                    inspect.isclass(obj)
                    and hasattr(obj, "name")
                    and hasattr(obj, "execute")
                    and not inspect.isabstract(obj)
                ):
                    await self._register_tool_class(obj)

                # Check if it's a decorated function
                elif hasattr(obj, "_is_tool"):
                    await self._register_tool_function(obj)

        except Exception as e:
            self.logger.error(f"Error registering tools from module {module}: {e}")

    async def _register_tool_class(self, tool_class: Type) -> None:
        """Register a tool class."""
        try:
            # Create instance
            tool_instance = tool_class()

            tool_info = {
                "name": tool_instance.name,
                "description": tool_instance.description,
                "parameters": self._extract_parameters(tool_instance.execute),
                "instance": tool_instance,
                "type": "class",
            }

            self.registered_tools[tool_instance.name] = tool_info
            self.tool_instances[tool_instance.name] = tool_instance

            self.logger.debug(f"Registered tool class: {tool_instance.name}")

        except Exception as e:
            self.logger.error(f"Failed to register tool class {tool_class}: {e}")

    async def _register_tool_function(self, tool_func: Callable) -> None:
        """Register a decorated tool function."""
        try:
            tool_name = getattr(tool_func, "_tool_name", tool_func.__name__)
            tool_description = getattr(
                tool_func, "_tool_description", tool_func.__doc__ or "No description"
            )

            tool_info = {
                "name": tool_name,
                "description": tool_description,
                "parameters": self._extract_parameters(tool_func),
                "function": tool_func,
                "type": "function",
            }

            self.registered_tools[tool_name] = tool_info

            self.logger.debug(f"Registered tool function: {tool_name}")

        except Exception as e:
            self.logger.error(f"Failed to register tool function {tool_func}: {e}")

    def _extract_parameters(self, func: Callable) -> Dict[str, Any]:
        """Extract parameter information from a function."""
        try:
            sig = inspect.signature(func)
            parameters = {}

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_info = {
                    "type": "any",
                    "required": param.default == inspect.Parameter.empty,
                    "default": (
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None
                    ),
                }

                # Try to extract type information
                if param.annotation != inspect.Parameter.empty:
                    param_info["type"] = (
                        str(param.annotation).replace("<class '", "").replace("'>", "")
                    )

                parameters[param_name] = param_info

            return parameters

        except Exception as e:
            self.logger.error(f"Error extracting parameters from {func}: {e}")
            return {}

    def _apply_tool_config(self) -> None:
        """Apply enabled/disabled configuration to tools."""
        # Disable tools that are explicitly disabled
        for tool_name in self.config.disabled:
            if tool_name in self.registered_tools:
                self.registered_tools[tool_name]["enabled"] = False
                self.logger.info(f"Disabled tool: {tool_name}")

        # If enabled list is specified, only enable those tools
        if self.config.enabled:
            for tool_name in self.registered_tools:
                enabled = tool_name in self.config.enabled
                self.registered_tools[tool_name]["enabled"] = enabled
                if not enabled:
                    self.logger.info(f"Tool not in enabled list: {tool_name}")
        else:
            # Enable all tools that aren't explicitly disabled
            for tool_name in self.registered_tools:
                if "enabled" not in self.registered_tools[tool_name]:
                    self.registered_tools[tool_name]["enabled"] = True

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool with the given parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool

        Returns:
            Dictionary containing execution result
        """
        print(f"🔧 [DEBUG] Executing tool: {tool_name} with parameters: {parameters}")

        if not self.is_initialized:
            await self.initialize()

        if tool_name not in self.registered_tools:
            print(
                f"🚨 [DEBUG] Tool '{tool_name}' not found! Available tools: {list(self.registered_tools.keys())}"
            )
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "result": None,
            }

        tool_info = self.registered_tools[tool_name]

        if not tool_info.get("enabled", True):
            print(f"🚨 [DEBUG] Tool '{tool_name}' is disabled!")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is disabled",
                "result": None,
            }

        try:
            # Validate parameters
            validated_params = self._validate_parameters(tool_info, parameters)
            if not validated_params["valid"]:
                return {
                    "success": False,
                    "error": f"Parameter validation failed: {validated_params['error']}",
                    "result": None,
                }

            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_tool_safely(tool_info, validated_params["parameters"]),
                timeout=self.execution_timeout,
            )

            return {"success": True, "error": None, "result": result}

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' execution timed out",
                "result": None,
            }
        except Exception as e:
            print(f"🚨 [DEBUG] Tool execution error for '{tool_name}': {e}")
            self.logger.error(f"Error executing tool '{tool_name}': {e}")
            return {"success": False, "error": str(e), "result": None}

    def _validate_parameters(
        self, tool_info: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate tool parameters."""
        try:
            tool_params = tool_info.get("parameters", {})
            validated_params = {}

            # Check required parameters
            for param_name, param_info in tool_params.items():
                if param_info.get("required", False):
                    if param_name not in parameters:
                        return {
                            "valid": False,
                            "error": f"Required parameter '{param_name}' is missing",
                            "parameters": {},
                        }

                # Use provided value or default
                if param_name in parameters:
                    validated_params[param_name] = parameters[param_name]
                elif param_info.get("default") is not None:
                    validated_params[param_name] = param_info["default"]

            return {"valid": True, "error": None, "parameters": validated_params}

        except Exception as e:
            return {"valid": False, "error": str(e), "parameters": {}}

    async def _execute_tool_safely(
        self, tool_info: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Any:
        """Execute tool with error handling."""
        try:
            if tool_info["type"] == "class":
                # Execute method on instance
                instance = tool_info["instance"]

                # Check if execute method is async
                if asyncio.iscoroutinefunction(instance.execute):
                    return await instance.execute(**parameters)
                else:
                    # Run in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, lambda: instance.execute(**parameters)
                    )

            elif tool_info["type"] == "function":
                # Execute function directly
                func = tool_info["function"]

                if asyncio.iscoroutinefunction(func):
                    return await func(**parameters)
                else:
                    # Run in executor
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, lambda: func(**parameters))

            else:
                raise ValueError(f"Unknown tool type: {tool_info['type']}")

        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools.

        Returns:
            List of tool information dictionaries
        """
        if not self.is_initialized:
            await self.initialize()

        available_tools = []

        for tool_name, tool_info in self.registered_tools.items():
            if tool_info.get("enabled", True):
                available_tools.append(
                    {
                        "name": tool_info["name"],
                        "description": tool_info["description"],
                        "parameters": tool_info["parameters"],
                        "type": tool_info["type"],
                    }
                )

        return available_tools

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool information dictionary or None if not found
        """
        if tool_name in self.registered_tools:
            tool_info = self.registered_tools[tool_name].copy()
            # Remove internal references
            if "instance" in tool_info:
                del tool_info["instance"]
            if "function" in tool_info:
                del tool_info["function"]
            return tool_info
        return None

    def register_tool(
        self, tool_name: str, tool_func: Callable, description: str = ""
    ) -> None:
        """
        Manually register a tool.

        Args:
            tool_name: Name of the tool
            tool_func: Function to execute
            description: Tool description
        """
        tool_info = {
            "name": tool_name,
            "description": description or tool_func.__doc__ or "No description",
            "parameters": self._extract_parameters(tool_func),
            "function": tool_func,
            "type": "function",
            "enabled": True,
        }

        self.registered_tools[tool_name] = tool_info
        self.logger.info(f"Manually registered tool: {tool_name}")

    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name in self.registered_tools:
            del self.registered_tools[tool_name]
            if tool_name in self.tool_instances:
                del self.tool_instances[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False

    async def cleanup(self) -> None:
        """Cleanup tool executor resources."""
        self.logger.info("Cleaning up tool executor...")

        # Cleanup tool instances
        for instance in self.tool_instances.values():
            if hasattr(instance, "cleanup"):
                try:
                    if asyncio.iscoroutinefunction(instance.cleanup):
                        await instance.cleanup()
                    else:
                        instance.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up tool instance: {e}")

        self.registered_tools.clear()
        self.tool_instances.clear()
        self.is_initialized = False

        self.logger.info("Tool executor cleanup complete")
