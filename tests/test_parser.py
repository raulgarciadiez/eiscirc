import pytest
from eiscirc.circuit_parser import parse_circuit


def test_parse_simple_series():
    expr = 'R0-C1-L1'
    parsed = parse_circuit(expr)
    assert parsed[0] == 'series'


def test_parse_nested_parallel_series():
    expr = 'R0-(CPE1//R1)'
    parsed = parse_circuit(expr)
    assert parsed[0] == 'series'


def test_invalid_single_slash():
    with pytest.raises(ValueError):
        parse_circuit('R0-CPE1/L1')


def test_invalid_equal_operator():
    with pytest.raises(ValueError):
        parse_circuit('R0=(CPE1//L1)')


def test_unbalanced_parentheses():
    with pytest.raises(ValueError):
        parse_circuit('R0-(CPE1//L1')


def test_invalid_characters():
    with pytest.raises(ValueError):
        parse_circuit('R0-@C1')
