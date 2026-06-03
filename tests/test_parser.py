import pytest
import matplotlib

# Use a non-interactive backend for drawing tests
matplotlib.use('Agg')

from eiscirc.circuit_parser import parse_circuit, ImpedanceModel
import matplotlib.pyplot as plt




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


def _flatten_tokens(parsed):
    """Helper to recursively collect leaf tokens from a parsed structure"""
    if isinstance(parsed, str):
        return [parsed]
    elif isinstance(parsed, tuple):
        tokens = []
        for part in parsed[1:]:
            tokens.extend(_flatten_tokens(part))
        return tokens
    return []


def test_parse_complex_nested_circuit():
    # A deliberately complex nested expression mixing series, parallel and parentheses
    expr = 'R0-(CPE1//(R1-L1))//((Ws1//C1)-TLM1)//(G1-(C2//L2))-R2'
    parsed = parse_circuit(expr)
    # Top level should be 'series' because final -R2 occurs
    assert isinstance(parsed, tuple)
    # Ensure some expected substructure tokens are present
    tokens = _flatten_tokens(parsed)
    for tok in ['R0', 'CPE1', 'R1', 'L1', 'Ws1', 'C1', 'TLM1', 'G1', 'C2', 'L2', 'R2']:
        assert tok in tokens


def test_parse_contains_L_element():
    expr = 'R0-(L1//C1)-((R1//L2)-CPE1)//(R2-L3)'
    parsed = parse_circuit(expr)
    tokens = _flatten_tokens(parsed)
    assert 'L1' in tokens and 'L2' in tokens and 'L3' in tokens


def test_draw_complex_circuit_default_ax():
    expr = 'R0-(L1//C1)-((R1//L2)-CPE1)//(R2-L3)'
    model = ImpedanceModel(expr)
    ax = model.draw_circuit()  # should not raise
    # Axes should be returned
    assert hasattr(ax, 'add_line')


def test_draw_complex_circuit_with_provided_axis():
    expr = 'R0-(L1//C1)-(R1//(L2-CPE1))'
    model = ImpedanceModel(expr)
    fig, ax = plt.subplots(figsize=(6, 3))
    returned = model.draw_circuit(ax=ax, position=[0.4, 0.25])
    # When provided an axis, we should still get an Axes object back
    assert hasattr(returned, 'add_line')
