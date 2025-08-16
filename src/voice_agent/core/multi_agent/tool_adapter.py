"""
Tool adapter for LlamaIndex compatibility.

Bridges the existing voice agent tool system with LlamaIndex FunctionTool format,
enabling seamless tool integration across the multi-agent framework.
"""

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type

from ..tool_executor import ToolExecutor

# LlamaIndex imports with fallback
try:
    from llama_index.core.tools import FunctionTool

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    FunctionTool = None
    LLAMAINDEX_AVAILABLE = False


class ToolAdapter:
    """
    Adapter class that converts voice agent tools to LlamaIndex FunctionTool format.

    Provides bidirectional conversion between the existing tool system and
    LlamaIndex tools, enabling agents to use all available tools seamlessly.
    """

    def __init__(self, tool_executor: ToolExecutor):
        """
        Initialize the tool adapter.

        Args:
            tool_executor: Existing ToolExecutor instance
        """
        self.tool_executor = tool_executor
        self.logger = logging.getLogger(__name__)

        # Cached conversions
        self.converted_tools: Dict[str, FunctionTool] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.conversion_stats = {
            "tools_converted": 0,
            "conversions_cached": 0,
            "execution_count": 0,
            "error_count": 0,
        }

    async def initialize(self) -> None:
        """Initialize the tool adapter."""
        if not LLAMAINDEX_AVAILABLE:
            self.logger.warning(
                "LlamaIndex not available - tool adapter will be limited"
            )
            return

        # Ensure tool executor is initialized
        if not self.tool_executor.is_initialized:
            await self.tool_executor.initialize()

        self.logger.info("Tool adapter initialized")

    async def get_llamaindex_tools(
        self, agent_id: Optional[str] = None
    ) -> List[FunctionTool]:
        """
        Get all available tools converted to LlamaIndex FunctionTool format.

        Args:
            agent_id: Optional agent ID for filtering tools

        Returns:
            List of LlamaIndex FunctionTool objects
        """
        if not LLAMAINDEX_AVAILABLE:
            self.logger.warning("LlamaIndex not available - returning empty tool list")
            return []

        # Get available tools from tool executor
        available_tools = await self.tool_executor.get_available_tools()
        converted_tools = []

        for tool_info in available_tools:
            tool_name = tool_info["name"]

            # Check cache first
            if tool_name in self.converted_tools:
                converted_tools.append(self.converted_tools[tool_name])
                self.conversion_stats["conversions_cached"] += 1
                continue

            # Convert tool
            try:
                function_tool = await self._convert_to_function_tool(tool_info)
                if function_tool:
                    self.converted_tools[tool_name] = function_tool
                    self.tool_metadata[tool_name] = tool_info
                    converted_tools.append(function_tool)
                    self.conversion_stats["tools_converted"] += 1

            except Exception as e:
                self.logger.error(f"Failed to convert tool {tool_name}: {e}")
                self.conversion_stats["error_count"] += 1

        self.logger.info(
            f"Converted {len(converted_tools)} tools for agent {agent_id or 'unknown'}"
        )
        return converted_tools

    async def _convert_to_function_tool(
        self, tool_info: Dict[str, Any]
    ) -> Optional[FunctionTool]:
        """
        Convert a voice agent tool to LlamaIndex FunctionTool.

        Args:
            tool_info: Tool information from ToolExecutor

        Returns:
            Converted FunctionTool or None if conversion fails
        """
        if not FunctionTool:
            return None

        tool_name = tool_info["name"]
        tool_description = tool_info["description"]
        tool_parameters = tool_info.get("parameters", {})

        # Create wrapper function for the tool
        async def tool_wrapper(**kwargs) -> str:
            """Wrapper function for executing voice agent tools via LlamaIndex."""
            try:
                self.conversion_stats["execution_count"] += 1

                # Execute tool via tool executor
                result = await self.tool_executor.execute_tool(tool_name, kwargs)

                if result["success"]:
                    # Convert result to string format expected by LlamaIndex
                    tool_result = result["result"]
                    if isinstance(tool_result, dict):
                        return self._format_dict_result(tool_result)
                    elif isinstance(tool_result, (list, tuple)):
                        return self._format_list_result(tool_result)
                    else:
                        return str(tool_result)
                else:
                    error_msg = result.get("error", "Unknown error")
                    self.logger.error(f"Tool {tool_name} execution failed: {error_msg}")
                    return f"Error executing {tool_name}: {error_msg}"

            except Exception as e:
                self.logger.error(f"Tool wrapper error for {tool_name}: {e}")
                self.conversion_stats["error_count"] += 1
                return f"Error executing {tool_name}: {str(e)}"

        # Build parameter schema for LlamaIndex
        parameter_schema = self._build_parameter_schema(tool_parameters)

        try:
            # Create FunctionTool with proper async handling
            function_tool = FunctionTool.from_defaults(
                fn=tool_wrapper,
                name=tool_name,
                description=tool_description,
                async_fn=tool_wrapper,  # Specify async function
                tool_metadata={"source": "voice_agent", "original_info": tool_info},
            )

            self.logger.debug(f"Successfully converted tool: {tool_name}")
            return function_tool

        except Exception as e:
            self.logger.error(f"Failed to create FunctionTool for {tool_name}: {e}")
            return None

    def _build_parameter_schema(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build parameter schema compatible with LlamaIndex.

        Args:
            parameters: Parameter information from voice agent tool

        Returns:
            Parameter schema for LlamaIndex
        """
        schema = {"type": "object", "properties": {}, "required": []}

        for param_name, param_info in parameters.items():
            param_type = param_info.get("type", "string")
            param_required = param_info.get("required", False)
            param_default = param_info.get("default")

            # Map voice agent types to JSON schema types
            json_type = self._map_parameter_type(param_type)

            param_schema = {
                "type": json_type,
                "description": param_info.get("description", f"Parameter {param_name}"),
            }

            if param_default is not None:
                param_schema["default"] = param_default

            schema["properties"][param_name] = param_schema

            if param_required:
                schema["required"].append(param_name)

        return schema

    def _map_parameter_type(self, voice_agent_type: str) -> str:
        """Map voice agent parameter types to JSON schema types."""
        type_mapping = {
            "str": "string",
            "string": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "dict": "object",
            "any": "string",  # Default to string for unknown types
        }

        # Extract base type (handle cases like "typing.List[str]")
        base_type = voice_agent_type.lower().split("[")[0].split(".")[-1]
        return type_mapping.get(base_type, "string")

    def _format_dict_result(self, result: Dict[str, Any]) -> str:
        """Format dictionary result for LlamaIndex consumption."""
        try:
            formatted_parts = []
            for key, value in result.items():
                if isinstance(value, (dict, list)):
                    formatted_parts.append(f"{key}: {str(value)}")
                else:
                    formatted_parts.append(f"{key}: {value}")

            return "\n".join(formatted_parts)

        except Exception:
            return str(result)

    def _format_list_result(self, result: List[Any]) -> str:
        """Format list result for LlamaIndex consumption."""
        try:
            if all(isinstance(item, dict) for item in result):
                # List of dictionaries - format as structured data
                formatted_items = []
                for i, item in enumerate(result):
                    formatted_items.append(f"Item {i + 1}:")
                    for key, value in item.items():
                        formatted_items.append(f"  {key}: {value}")
                    formatted_items.append("")  # Blank line between items

                return "\n".join(formatted_items)
            else:
                # Simple list - join with newlines
                return "\n".join(str(item) for item in result)

        except Exception:
            return str(result)

    def get_tool_by_name(self, tool_name: str) -> Optional[FunctionTool]:
        """
        Get a specific converted tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            LlamaIndex FunctionTool or None if not found
        """
        return self.converted_tools.get(tool_name)

    def get_available_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.converted_tools.keys())

    async def refresh_tools(self) -> int:
        """
        Refresh tool conversions from the tool executor.

        Returns:
            Number of tools refreshed
        """
        self.logger.info("Refreshing tool conversions")

        # Clear cache
        old_count = len(self.converted_tools)
        self.converted_tools.clear()
        self.tool_metadata.clear()

        # Rebuild conversions
        await self.get_llamaindex_tools()
        new_count = len(self.converted_tools)

        self.logger.info(f"Refreshed tools: {old_count} -> {new_count}")
        return new_count

    def filter_tools_for_agent(
        self, tools: List[FunctionTool], agent_capabilities: List[str]
    ) -> List[FunctionTool]:
        """
        Filter tools based on agent capabilities.

        Args:
            tools: List of all available tools
            agent_capabilities: List of agent capability strings

        Returns:
            Filtered list of tools appropriate for the agent
        """
        if not agent_capabilities:
            return tools

        filtered_tools = []
        capability_keywords = {
            "tool_execution": ["calculate", "compute", "math", "system"],
            "file_operations": ["file", "directory", "path", "folder"],
            "web_search": ["search", "web", "internet", "url"],
            "weather_info": ["weather", "temperature", "forecast", "climate"],
            "calculations": ["calculate", "compute", "math", "equation"],
            "system_info": ["system", "info", "status", "monitor"],
        }

        for tool in tools:
            tool_name = tool.metadata.get("name", "").lower()
            tool_desc = tool.metadata.get("description", "").lower()
            tool_text = f"{tool_name} {tool_desc}"

            # Check if tool matches any agent capability
            tool_matches = False
            for capability in agent_capabilities:
                capability_key = capability.lower().replace("_", "_")
                keywords = capability_keywords.get(capability_key, [capability_key])

                if any(keyword in tool_text for keyword in keywords):
                    tool_matches = True
                    break

            if tool_matches:
                filtered_tools.append(tool)

        self.logger.debug(
            f"Filtered {len(tools)} tools to {len(filtered_tools)} for capabilities: {agent_capabilities}"
        )
        return filtered_tools

    async def execute_tool_directly(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool directly via the tool executor (bypass LlamaIndex).

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters

        Returns:
            Execution result
        """
        return await self.tool_executor.execute_tool(tool_name, parameters)

    def create_custom_function_tool(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[FunctionTool]:
        """
        Create a custom LlamaIndex FunctionTool from a function.

        Args:
            name: Tool name
            description: Tool description
            func: Function to wrap
            parameters: Optional parameter schema

        Returns:
            Created FunctionTool or None if LlamaIndex not available
        """
        if not FunctionTool:
            return None

        try:
            # Determine if function is async
            is_async = asyncio.iscoroutinefunction(func)

            # Create wrapper for consistent return format
            async def async_wrapper(**kwargs) -> str:
                try:
                    if is_async:
                        result = await func(**kwargs)
                    else:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: func(**kwargs)
                        )
                    return str(result)
                except Exception as e:
                    return f"Error executing {name}: {str(e)}"

            def sync_wrapper(**kwargs) -> str:
                try:
                    result = func(**kwargs)
                    return str(result)
                except Exception as e:
                    return f"Error executing {name}: {str(e)}"

            # Create FunctionTool
            if is_async:
                function_tool = FunctionTool.from_defaults(
                    fn=sync_wrapper,  # Sync version for compatibility
                    async_fn=async_wrapper,  # Async version
                    name=name,
                    description=description,
                    tool_metadata={"source": "custom", "is_async": True},
                )
            else:
                function_tool = FunctionTool.from_defaults(
                    fn=sync_wrapper,
                    name=name,
                    description=description,
                    tool_metadata={"source": "custom", "is_async": False},
                )

            # Cache the tool
            self.converted_tools[name] = function_tool
            self.tool_metadata[name] = {
                "name": name,
                "description": description,
                "type": "custom",
                "parameters": parameters or {},
            }

            self.logger.info(f"Created custom function tool: {name}")
            return function_tool

        except Exception as e:
            self.logger.error(f"Failed to create custom function tool {name}: {e}")
            return None

    def get_adapter_stats(self) -> Dict[str, Any]:
        """Get tool adapter statistics."""
        return {
            **self.conversion_stats,
            "cached_tools": len(self.converted_tools),
            "llamaindex_available": LLAMAINDEX_AVAILABLE,
            "tool_executor_initialized": self.tool_executor.is_initialized,
        }

    async def cleanup(self) -> None:
        """Cleanup tool adapter resources."""
        self.logger.info("Cleaning up tool adapter")

        self.converted_tools.clear()
        self.tool_metadata.clear()

        # Reset stats
        self.conversion_stats = {key: 0 for key in self.conversion_stats}

        self.logger.info("Tool adapter cleanup complete")
