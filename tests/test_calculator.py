"""Unit tests for the AST-based calculator tool."""

import pytest

from multitool_agent.tools.calculator import CalculatorTool


@pytest.fixture
def calc() -> CalculatorTool:
    return CalculatorTool()


def test_schema(calc: CalculatorTool) -> None:
    schema = calc.schema()
    assert schema["name"] == "calculator"
    assert "expression" in schema["input_schema"]["properties"]
    assert "expression" in schema["input_schema"]["required"]


def test_basic_arithmetic(calc: CalculatorTool) -> None:
    assert calc.execute({"expression": "2 + 2"}) == "4"
    assert calc.execute({"expression": "10 * 5"}) == "50"
    assert calc.execute({"expression": "100 / 4"}) == "25"
    assert calc.execute({"expression": "2 ** 8"}) == "256"
    assert calc.execute({"expression": "(5 + 3) * 2"}) == "16"
    assert calc.execute({"expression": "(50 + 25) * 2"}) == "150"


def test_decimals_and_unary(calc: CalculatorTool) -> None:
    assert calc.execute({"expression": "125 * 17.5 / 100"}) == "21.875"
    assert calc.execute({"expression": "5.5 + 4.5"}) == "10"
    assert calc.execute({"expression": "-5 + 10"}) == "5"
    assert calc.execute({"expression": "-(10 - 2)"}) == "-8"
    assert calc.execute({"expression": "+(4 * 2)"}) == "8"


def test_percentages(calc: CalculatorTool) -> None:
    # 15% of 2000 = 300
    assert calc.execute({"expression": "15% of 2000"}) == "300"
    # 17.5% * 850 = 148.75
    assert calc.execute({"expression": "17.5% * 850"}) == "148.75"
    # Standard modulo 10 % 3 = 1
    assert calc.execute({"expression": "10 % 3"}) == "1"


def test_division_by_zero(calc: CalculatorTool) -> None:
    res = calc.execute({"expression": "100 / 0"})
    assert "Error: Division by zero" in res

    res_mod = calc.execute({"expression": "10 % 0"})
    assert "Error: Division by zero" in res_mod or "zero" in res_mod.lower()


def test_missing_or_empty_arguments(calc: CalculatorTool) -> None:
    assert "Missing required argument" in calc.execute({})
    assert "Empty expression" in calc.execute({"expression": "   "})


def test_syntax_errors(calc: CalculatorTool) -> None:
    res = calc.execute({"expression": "2 + * 4"})
    assert "Invalid mathematical syntax" in res


def test_security_rejections(calc: CalculatorTool) -> None:
    # Function calls
    res = calc.execute({"expression": "__import__('os').system('ls')"})
    assert "Error:" in res and ("Unsupported" in res or "syntax" in res.lower())

    res = calc.execute({"expression": "open('/etc/passwd')"})
    assert "Error:" in res and ("Unsupported" in res or "syntax" in res.lower())

    res = calc.execute({"expression": "eval('1 + 1')"})
    assert "Error:" in res and ("Unsupported" in res or "syntax" in res.lower())

    # Attribute access
    res = calc.execute({"expression": "(1).__class__"})
    assert "Error:" in res and "Unsupported" in res

    # Variable lookup
    res = calc.execute({"expression": "x + 5"})
    assert "Error:" in res and "Unsupported" in res

    # String literals
    res = calc.execute({"expression": "'hello' + 'world'"})
    assert "Error:" in res and "Unsupported" in res


def test_denial_of_service_prevention(calc: CalculatorTool) -> None:
    # Exponent too large
    res = calc.execute({"expression": "2 ** 100000"})
    assert "Error:" in res and "limit" in res.lower()

    res = calc.execute({"expression": "99999 ** 20"})
    assert "Error:" in res and "limit" in res.lower()
