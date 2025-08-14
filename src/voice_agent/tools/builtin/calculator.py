"""
Calculator tool for basic mathematical operations.
"""

import ast
import logging
import operator
from typing import Any, Dict, Union

from pydantic import BaseModel, Field

from ..base import Tool


class CalculatorParameters(BaseModel):
    """Parameters for the calculator tool."""

    expression: str = Field(description="Mathematical expression to evaluate")


class CalculatorTool(Tool):
    """
    Calculator tool for performing basic mathematical operations.

    Safely evaluates mathematical expressions using a restricted AST parser.
    Supports basic arithmetic operations: +, -, *, /, **, %, and parentheses.
    """

    name = "calculator"
    description = "Perform basic mathematical calculations"
    version = "1.0.0"

    Parameters = CalculatorParameters

    # Safe operators for mathematical expressions
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Safe functions
    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }

    def __init__(self):
        """Initialize the calculator tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def execute(self, expression: str) -> Dict[str, Any]:
        """
        Execute a mathematical calculation.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Dictionary containing the calculation result
        """
        try:
            # Clean and validate the expression
            expression = expression.strip()
            if not expression:
                return {
                    "success": False,
                    "result": None,
                    "error": "Empty expression provided",
                }

            # Parse and evaluate the expression safely
            result = self._safe_eval(expression)

            return {
                "success": True,
                "result": result,
                "error": None,
                "expression": expression,
            }

        except ZeroDivisionError:
            return {"success": False, "result": None, "error": "Division by zero"}
        except (ValueError, TypeError) as e:
            return {
                "success": False,
                "result": None,
                "error": f"Invalid expression: {str(e)}",
            }
        except SyntaxError:
            return {
                "success": False,
                "result": None,
                "error": "Invalid mathematical syntax",
            }
        except Exception as e:
            self.logger.error(f"Calculator error: {e}")
            return {
                "success": False,
                "result": None,
                "error": "Calculation error occurred",
            }

    def _safe_eval(self, expression: str) -> Union[int, float]:
        """
        Safely evaluate a mathematical expression using AST.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Numerical result of the expression

        Raises:
            ValueError: If expression contains unsafe operations
            SyntaxError: If expression has invalid syntax
        """
        try:
            # Parse the expression into an AST
            node = ast.parse(expression, mode="eval")

            # Evaluate the AST safely
            result = self._eval_node(node.body)

            # Ensure result is a number
            if not isinstance(result, (int, float)):
                raise ValueError("Expression must evaluate to a number")

            return result

        except SyntaxError as e:
            raise SyntaxError(f"Invalid syntax in expression: {expression}. Error {e}")

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """
        Recursively evaluate an AST node.

        Args:
            node: AST node to evaluate

        Returns:
            Numerical result

        Raises:
            ValueError: If node contains unsafe operations
        """
        if isinstance(node, ast.Constant):
            # Handle constants (numbers)
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Num):
            # Handle numbers (older Python versions)
            return node.n

        elif isinstance(node, ast.BinOp):
            # Handle binary operations (+, -, *, /, etc.)
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            if type(node.op) in self.SAFE_OPERATORS:
                op_func = self.SAFE_OPERATORS[type(node.op)]
                return op_func(left, right)
            else:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

        elif isinstance(node, ast.UnaryOp):
            # Handle unary operations (-, +)
            operand = self._eval_node(node.operand)

            if type(node.op) in self.SAFE_OPERATORS:
                op_func = self.SAFE_OPERATORS[type(node.op)]
                return op_func(operand)
            else:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )

        elif isinstance(node, ast.Call):
            # Handle function calls
            if isinstance(node.func, ast.Name):
                func_name = node.func.id

                if func_name in self.SAFE_FUNCTIONS:
                    # Evaluate arguments
                    args = [self._eval_node(arg) for arg in node.args]

                    # Call the function
                    func = self.SAFE_FUNCTIONS[func_name]
                    return func(*args)
                else:
                    raise ValueError(f"Unsupported function: {func_name}")
            else:
                raise ValueError("Complex function calls are not supported")

        else:
            raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    def get_help(self) -> Dict[str, Any]:
        """
        Get help information for the calculator tool.

        Returns:
            Dictionary containing help information
        """
        return {
            "name": self.name,
            "description": self.description,
            "supported_operations": [
                "Addition (+)",
                "Subtraction (-)",
                "Multiplication (*)",
                "Division (/)",
                "Floor Division (//)",
                "Modulo (%)",
                "Exponentiation (**)",
                "Parentheses for grouping",
            ],
            "supported_functions": list(self.SAFE_FUNCTIONS.keys()),
            "examples": [
                "2 + 3 * 4",
                "(10 + 5) / 3",
                "2 ** 8",
                "abs(-42)",
                "max(1, 2, 3, 4, 5)",
                "round(3.14159, 2)",
            ],
        }
