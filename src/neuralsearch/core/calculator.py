"""
Safe, sandboxed math expression evaluator and unit converter for instant search cards.
Zero eval() risk; uses AST parsing for mathematical formulas and fixed conversion tables.
"""

from __future__ import annotations
import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Optional, Tuple

# Supported operators for AST evaluation
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

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log10,
    "ln": math.log,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


@dataclass(slots=True)
class InstantCalculation:
    """Evaluated calculation or conversion result."""
    expression: str
    result: str
    formatted_display: str
    calculation_type: str  # "math" | "unit" | "currency"


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluates a safe AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.Name):
        lower_name = node.id.lower()
        if lower_name in SAFE_CONSTANTS:
            return float(SAFE_CONSTANTS[lower_name])
        raise ValueError(f"Unknown constant: {node.id}")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](_eval_node(node.operand))
        raise ValueError(f"Unsupported unary operator: {op_type}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            # Guard against massive exponentiation DDOS
            if op_type is ast.Pow and (abs(left) > 1000 or abs(right) > 1000):
                raise ValueError("Exponent too large")
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            if func_name in SAFE_FUNCTIONS:
                args = [_eval_node(arg) for arg in node.args]
                return float(SAFE_FUNCTIONS[func_name](*args))
        raise ValueError("Unsupported function call")

    raise ValueError(f"Unsupported AST node: {type(node)}")


def evaluate_math(query: str) -> Optional[InstantCalculation]:
    """
    Detects and safely evaluates arithmetic formulas (e.g. '45 * 12', 'sqrt(144) + 10', '2^16').
    """
    clean = query.strip().lower()

    # Handle percentage query like '15% of 250'
    pct_match = re.match(r'^(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)$', clean)
    if pct_match:
        pct = float(pct_match.group(1))
        val = float(pct_match.group(2))
        res = (pct / 100.0) * val
        formatted_res = f"{res:g}"
        return InstantCalculation(
            expression=f"{pct}% of {val}",
            result=formatted_res,
            formatted_display=f"{pct}% of {val:g} = {formatted_res}",
            calculation_type="math",
        )

    # Normalize power operator ^ to **
    expr = clean.replace("^", "**").replace("×", "*").replace("÷", "/")

    # Ensure query contains at least one math operator or safe function to avoid evaluating plain numbers
    has_math = any(op in expr for op in ["+", "-", "*", "/", "%", "**"]) or any(
        fn in expr for fn in ["sqrt", "sin", "cos", "tan", "log", "ln"]
    )
    if not has_math:
        return None

    # Disallow dangerous substrings
    if any(k in expr for k in ["import", "exec", "eval", "__", "lambda", "class", "def", ";"]):
        return None

    try:
        parsed = ast.parse(expr, mode="eval")
        res_val = _eval_node(parsed.body)
        formatted_res = f"{round(res_val, 8):g}"
        orig_display = clean.replace("**", "^")
        return InstantCalculation(
            expression=orig_display,
            result=formatted_res,
            formatted_display=f"{orig_display} = {formatted_res}",
            calculation_type="math",
        )
    except Exception:
        return None


# -----------------------------------------------------------------------------
# UNIT CONVERSION
# -----------------------------------------------------------------------------

# Conversion factors to standard base units (meter, gram, second, liter)
LENGTH_UNITS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
}

WEIGHT_UNITS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "mg": 0.001, "milligram": 0.001, "milligrams": 0.001,
    "lb": 453.59237, "pound": 453.59237, "pounds": 453.59237, "lbs": 453.59237,
    "oz": 28.349523, "ounce": 28.349523, "ounces": 28.349523,
}

DATA_UNITS = {
    "b": 1.0, "byte": 1.0, "bytes": 1.0,
    "kb": 1024.0, "kilobyte": 1024.0, "kilobytes": 1024.0,
    "mb": 1024.0**2, "megabyte": 1024.0**2, "megabytes": 1024.0**2,
    "gb": 1024.0**3, "gigabyte": 1024.0**3, "gigabytes": 1024.0**3,
    "tb": 1024.0**4, "terabyte": 1024.0**4, "terabytes": 1024.0**4,
}


def evaluate_unit_conversion(query: str) -> Optional[InstantCalculation]:
    """
    Detects and converts length, weight, data, and temperature units.
    e.g. '100 km to miles', '32 c to f', '5 kg in lbs', '1000 mb to gb'.
    """
    clean = query.strip().lower()
    pattern = r'^(\d+(?:\.\d+)?)\s*([a-zA-Z°]+)\s+(?:to|in)\s+([a-zA-Z°]+)$'
    match = re.match(pattern, clean)
    if not match:
        return None

    val = float(match.group(1))
    u_from = match.group(2).replace("°", "")
    u_to = match.group(3).replace("°", "")

    # 1. Temperature
    if u_from in ("c", "celsius") and u_to in ("f", "fahrenheit"):
        res = (val * 9.0 / 5.0) + 32.0
        return InstantCalculation(
            expression=f"{val}°C to °F",
            result=f"{res:.2f}°F",
            formatted_display=f"{val:g}° Celsius = {res:.2f}° Fahrenheit",
            calculation_type="unit",
        )
    if u_from in ("f", "fahrenheit") and u_to in ("c", "celsius"):
        res = (val - 32.0) * 5.0 / 9.0
        return InstantCalculation(
            expression=f"{val}°F to °C",
            result=f"{res:.2f}°C",
            formatted_display=f"{val:g}° Fahrenheit = {res:.2f}° Celsius",
            calculation_type="unit",
        )

    # 2. Length
    if u_from in LENGTH_UNITS and u_to in LENGTH_UNITS:
        meters = val * LENGTH_UNITS[u_from]
        res = meters / LENGTH_UNITS[u_to]
        return InstantCalculation(
            expression=f"{val} {u_from} to {u_to}",
            result=f"{res:g} {u_to}",
            formatted_display=f"{val:g} {u_from} = {res:.4g} {u_to}",
            calculation_type="unit",
        )

    # 3. Weight
    if u_from in WEIGHT_UNITS and u_to in WEIGHT_UNITS:
        grams = val * WEIGHT_UNITS[u_from]
        res = grams / WEIGHT_UNITS[u_to]
        return InstantCalculation(
            expression=f"{val} {u_from} to {u_to}",
            result=f"{res:g} {u_to}",
            formatted_display=f"{val:g} {u_from} = {res:.4g} {u_to}",
            calculation_type="unit",
        )

    # 4. Data
    if u_from in DATA_UNITS and u_to in DATA_UNITS:
        bytes_val = val * DATA_UNITS[u_from]
        res = bytes_val / DATA_UNITS[u_to]
        return InstantCalculation(
            expression=f"{val} {u_from} to {u_to}",
            result=f"{res:g} {u_to.upper()}",
            formatted_display=f"{val:g} {u_from.upper()} = {res:.4g} {u_to.upper()}",
            calculation_type="unit",
        )

    return None


def get_instant_tool(query: str) -> Optional[InstantCalculation]:
    """Unified detector for math and unit conversions."""
    # Check unit conversion first
    unit_res = evaluate_unit_conversion(query)
    if unit_res:
        return unit_res

    # Check math expression
    math_res = evaluate_math(query)
    if math_res:
        return math_res

    return None
