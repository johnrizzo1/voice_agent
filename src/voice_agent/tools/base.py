"""
Base tool class for the voice agent tool framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel


class Tool(ABC):
    """
    Base class for all voice agent tools.

    Tools provide extensible capabilities for the voice agent,
    allowing it to perform actions beyond conversation.
    """

    # Tool metadata (to be overridden by subclasses)
    name: str = "base_tool"
    description: str = "Base tool class"
    version: str = "1.0.0"

    # Parameter schema class (to be overridden by subclasses)
    Parameters: Type[BaseModel] = BaseModel

    def __init__(self):
        """Initialize the tool."""
        self.is_initialized = False

    async def initialize(self) -> None:
        """
        Initialize the tool (optional override).
        Called when the tool is first loaded.
        """
        self.is_initialized = True

    @abstractmethod
    def execute(self, **parameters) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.

        Args:
            **parameters: Tool parameters

        Returns:
            Dictionary containing execution result
        """
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool parameters using the Parameters schema.

        Args:
            parameters: Raw parameters to validate

        Returns:
            Validated parameters dictionary
        """
        try:
            # Use Pydantic model for validation
            validated = self.Parameters(**parameters)
            return validated.model_dump()
        except Exception as e:
            raise ValueError(f"Parameter validation failed: {e}")

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool's parameter schema.

        Returns:
            JSON schema dictionary
        """
        if self.Parameters == BaseModel:
            return {"type": "object", "properties": {}}

        return self.Parameters.model_json_schema()

    def get_info(self) -> Dict[str, Any]:
        """
        Get tool information.

        Returns:
            Dictionary containing tool metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "schema": self.get_schema(),
            "initialized": self.is_initialized,
        }

    async def cleanup(self) -> None:
        """
        Cleanup tool resources (optional override).
        Called when the tool is being unloaded.
        """
        self.is_initialized = False


class SimpleTool(Tool):
    """
    Simplified tool class for function-based tools.

    This class allows creating tools from simple functions
    without needing to create a full class.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func,
        parameters_schema: Optional[Type[BaseModel]] = None,
    ):
        """
        Initialize a simple tool.

        Args:
            name: Tool name
            description: Tool description
            func: Function to execute
            parameters_schema: Pydantic model for parameter validation
        """
        super().__init__()

        self.name = name
        self.description = description
        self._func = func

        if parameters_schema:
            self.Parameters = parameters_schema

    def execute(self, **parameters) -> Dict[str, Any]:
        """Execute the wrapped function."""
        try:
            # Validate parameters
            validated_params = self.validate_parameters(parameters)

            # Execute function
            result = self._func(**validated_params)

            return {"success": True, "result": result, "error": None}

        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}


# Tool decorator for easy tool creation
def tool(
    name: str, description: str = "", parameters: Optional[Type[BaseModel]] = None
):
    """
    Decorator to create a tool from a function.

    Args:
        name: Tool name
        description: Tool description
        parameters: Pydantic model for parameter validation

    Returns:
        Decorated function with tool metadata
    """

    def decorator(func):
        # Mark function as a tool
        func._is_tool = True
        func._tool_name = name
        func._tool_description = description or func.__doc__ or "No description"
        func._tool_parameters = parameters

        return func

    return decorator
