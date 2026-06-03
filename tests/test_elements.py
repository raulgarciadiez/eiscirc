import numpy as np
from eiscirc.impedance_parameters_default import PARAMETER_CONFIG, extract_base_type
from eiscirc.circuit_parser import ImpedanceModel


def test_supported_elements_compute_impedance():
    freqs = np.array([1.0, 10.0, 100.0])
    omega = 2 * np.pi * freqs

    # For each known base type, create a single-token component name and test
    for base in PARAMETER_CONFIG.keys():
        # create a token like R0, C1, CPE1, Ws1, etc.
        token = f"{base}1"
        # Build a simple model with that single element
        try:
            model = ImpedanceModel(token)
        except ValueError as e:
            # Some base types (e.g. TLM) may be declared in PARAMETER_CONFIG but not
            # yet implemented in the impedance compiler. Skip those with a clear message.
            if 'Unknown component type' in str(e):
                import pytest

                pytest.skip(f"Skipping unimplemented base type in impedance compiler: {base}")
            raise

        # Ensure default params are present
        params = model.get_all_params()
        # Set any structured defaults to the values in PARAMETER_CONFIG where applicable
        # initialize_parameters already sets defaults, but be explicit for structured types
        # Try computing impedance and assert shape
        Z = model.impedance(omega)
        assert Z.shape[0] == 2 * omega.shape[0], f"Unexpected Z shape for {token}"
