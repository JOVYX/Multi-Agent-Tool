"""Safe AST-based calculator tool."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Union

from multitool_agent.tools.base import BaseTool
from multitool_agent.utils.logging import get_logger

logger = get_logger("multitool_agent.tools.calculator")

# Safety limits for computation
MAX_EXPRESSION_LENGTH = 500
MAX_EXPONENT_BASE = 10_000
MAX_EXPONENT_VALUE = 1_000
MAX_RESULT_MAGNITUDE = 1e100


class CalculatorTool(BaseTool):
    """Tool for safe evaluation of arithmetic expressions without using unsafe eval()."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate an arithmetic expression and return the numerical result. "
            "Supports +, -, *, /, //, %, **, parentheses, decimals, and percentages. "
            "Example expressions: '2 + 2', '125 * 17.5 / 100', '15% of 2000', '2 ** 8'."
        )

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "The mathematical expression to evaluate, e.g. '100 / 4', "
                            "'125 * 17.5 / 100', '(50 + 25) * 2'."
                        ),
                    }
                },
                "required": ["expression"],
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute calculation safely using AST parser."""
        if not isinstance(arguments, dict) or "expression" not in arguments:
            return "Error: Missing required argument 'expression'."

        raw_expr = str(arguments["expression"]).strip()
        if not raw_expr:
            return "Error: Empty expression provided."

        if len(raw_expr) > MAX_EXPRESSION_LENGTH:
            return f"Error: Expression exceeds maximum length of {MAX_EXPRESSION_LENGTH} characters."

        try:
            cleaned_expr = self._preprocess_expression(raw_expr)
            result = self._evaluate_safely(cleaned_expr)
            return self._format_result(result)
        except ZeroDivisionError:
            return "Error: Division by zero is undefined."
        except OverflowError:
            return "Error: Arithmetic overflow. The result is too large to represent."
        except SyntaxError:
            return f"Error: Invalid mathematical syntax in expression '{raw_expr}'."
        except ValueError as ve:
            return f"Error: {str(ve)}"
        except Exception as e:
            logger.debug(f"Calculator unexpected error on '{raw_expr}': {e}")
            return f"Error evaluating expression: {type(e).__name__}: {str(e)}"

    def _preprocess_expression(self, expr: str) -> str:
        """Preprocess user expressions to support percentages and clean syntax."""
        # Replace 'X% of Y' with '(X / 100) * Y'
        expr = re.sub(
            r"(\d+(?:\.\d+)?)\s*%\s*of\s*",
            r"(\1 / 100) * ",
            expr,
            flags=re.IGNORECASE,
        )

        # Replace 'X%' when followed by an operator, bracket, or end of string with '(X / 100)'
        expr = re.sub(
            r"(\d+(?:\.\d+)?)\s*%(?=\s*($|[)+*/\-,]))",
            r"(\1 / 100)",
            expr,
        )

        # Replace caret '^' with power operator '**' if used as exponent
        # But ensure we do not replace already valid ** with ****
        if "^" in expr:
            expr = expr.replace("^", "**")

        return expr.strip()

    def _evaluate_safely(self, expr: str) -> Union[int, float]:
        """Parse and evaluate expression using AST traversal."""
        parsed = ast.parse(expr, mode="eval")
        return self._eval_node(parsed.body)

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """Recursively evaluate an AST node, strictly allowing only safe arithmetic operations."""
        if isinstance(node, ast.Constant):
            # Booleans are subclasses of int in Python, so explicitly reject them
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
            return node.value

        # For older python versions compatibility
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return node.n  # type: ignore[attr-defined]

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            op_map = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }

            op_type = type(node.op)
            if op_type not in op_map:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

            # Safety guards against denial of service with huge exponents
            if op_type is ast.Pow:
                if abs(right) > MAX_EXPONENT_VALUE:
                    raise ValueError(
                        f"Exponent {right} exceeds safety limit of {MAX_EXPONENT_VALUE}."
                    )
                if abs(left) > MAX_EXPONENT_BASE and right > 10:
                    raise ValueError(
                        f"Base {left} with exponent {right} exceeds computation limits."
                    )

            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division or modulo by zero.")

            result = op_map[op_type](left, right)

            # Check magnitude to avoid infinite or NaN results
            if isinstance(result, (int, float)) and abs(result) > MAX_RESULT_MAGNITUDE:
                raise OverflowError("Result exceeds maximum calculable magnitude.")

            return result

        # Explicitly reject all other AST node types (Call, Attribute, Name, Import, etc.)
        raise ValueError(
            f"Unsupported syntax or disallowed operation: {type(node).__name__}. "
            "Only numerical arithmetic expressions are permitted."
        )

    def _format_result(self, val: Union[int, float]) -> str:
        """Format number cleanly, trimming unnecessary decimal places."""
        if isinstance(val, float):
            # Check if float is actually a whole integer
            if val.is_integer() and abs(val) < 1e15:
                return str(int(val))
            # Round to 10 decimal places to eliminate floating point artifacts
            rounded = round(val, 10)
            return f"{rounded:g}"
        return str(val)
