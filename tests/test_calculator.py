import pytest
from neuralsearch.core.calculator import evaluate_math, evaluate_unit_conversion, get_instant_tool


def test_basic_arithmetic():
    calc = evaluate_math("45 * 12")
    assert calc is not None
    assert calc.result == "540"
    assert calc.calculation_type == "math"

    calc = evaluate_math("100 / 4 + 15")
    assert calc is not None
    assert calc.result == "40"


def test_percentage():
    calc = evaluate_math("15% of 250")
    assert calc is not None
    assert calc.result == "37.5"
    assert "15% of 250" in calc.formatted_display


def test_power_and_sqrt():
    calc = evaluate_math("2^10")
    assert calc is not None
    assert calc.result == "1024"

    calc = evaluate_math("sqrt(144)")
    assert calc is not None
    assert calc.result == "12"


def test_invalid_or_non_math_queries():
    assert evaluate_math("hello world") is None
    assert evaluate_math("fastapi tutorial") is None
    assert evaluate_math("123") is None
    assert evaluate_math("__import__('os').system('ls')") is None


def test_length_conversion():
    conv = evaluate_unit_conversion("10 km to miles")
    assert conv is not None
    assert conv.calculation_type == "unit"
    assert "miles" in conv.result

    conv = evaluate_unit_conversion("100 m in cm")
    assert conv is not None
    assert conv.result == "10000 cm"


def test_temperature_conversion():
    conv = evaluate_unit_conversion("100 c to f")
    assert conv is not None
    assert "212.00°F" in conv.result

    conv = evaluate_unit_conversion("32 f to c")
    assert conv is not None
    assert "0.00°C" in conv.result


def test_data_conversion():
    conv = evaluate_unit_conversion("1024 mb to gb")
    assert conv is not None
    assert conv.result == "1 GB"


def test_get_instant_tool():
    tool = get_instant_tool("50 * 4")
    assert tool is not None
    assert tool.result == "200"

    tool = get_instant_tool("10 km to m")
    assert tool is not None
    assert tool.result == "10000 m"

    tool = get_instant_tool("random search text")
    assert tool is None
