import numpy as np
from eiscirc.circuit_parser import ImpedanceModel


def test_resistor_impedance_shape_and_value():
    m = ImpedanceModel('R0')
    freqs = np.array([1.0, 10.0, 100.0])
    omega = 2 * np.pi * freqs
    Z = m.impedance(omega)
    # Z must be concatenated real and imag of same length as omega
    assert Z.shape[0] == 2 * omega.shape[0]
    # For pure resistor, imaginary part should be zero
    real = Z[: len(omega)]
    imag = Z[len(omega) :]
    assert np.allclose(imag, 0.0)
    assert np.allclose(real, m.params['R0'])


def test_cpe_basic_behavior():
    m = ImpedanceModel('R0-CPE1')
    freqs = np.array([1.0, 10.0, 100.0])
    omega = 2 * np.pi * freqs
    # Set a known CPE: alpha=1 => behaves as ideal capacitor.
    # Use a magnitude inside the default bounds (e.g. 1e-6 F)
    Cval = 1e-6
    # Do not set R0 to 0 because default bounds require R0 >= 1e-3
    m.set_params(CPE1=(Cval, 1.0))
    Z = m.impedance(omega)
    real = Z[: len(omega)]
    imag = Z[len(omega) :]
    # For ideal capacitor with C=Cval, Z_C = 1/(j*omega*Cval)
    # The model has R0 default (non-zero). So real part equals R0, imag equals -1/(omega*Cval)
    expected_real = np.ones_like(real) * m.params['R0']
    expected_imag = -1.0 / (omega * Cval)
    assert np.allclose(real, expected_real, atol=1e-12)
    assert np.allclose(imag, expected_imag, rtol=1e-6)
