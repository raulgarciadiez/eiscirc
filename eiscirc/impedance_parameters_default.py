# impedance_parameters_default.py

import numpy as np
import re

PARAMETER_CONFIG = {
    # Basic elements
    'R': {
        'default': 100,
        'bounds': (1e-3, 1e6),
        'description': 'Resistance (Ω)'
    },
    'C': {
        'default': 1e-6,
        'bounds': (1e-12, 1e-2), 
        'description': 'Capacitance (F)'
    },
    'L': {
        'default': 1e-3,
        'bounds': (1e-9, 1),
        'description': 'Inductance (H)'
    },

    # Distributed elements
    'W': {
        'default': 100,
        'bounds': (1e-3, 1e6),
        'description': 'Semi-infinite Warburg (Ω/s^0.5)'
    },

    # Constant Phase Element
    'CPE': {
        'value': {
            'default': 1e-6,
            'bounds': (1e-12, 1e-2),
            'description': 'CPE magnitude (S·s^α)'
        },
        'alpha': {
            'default': 0.8,
            'bounds': (0.001, 1.0),
            'description': 'CPE exponent (0=resistor, 1=capacitor)'
        }
    },

    # Finite-length Warburg elements
    'Ws': {
        'R': {
            'default': 100,
            'bounds': (1e-3, 1e6),
            'description': 'Finite Warburg resistance (Ω)'
        },
        'tau': {
            'default': 1,
            'bounds': (1e-6, 100),
            'description': 'Diffusion time constant (s)'
        }
    },
    'Wo': {
        'R': {
            'default': 100,
            'bounds': (1e-3, 1e6),
            'description': 'Open Warburg resistance (Ω)'
        },
        'tau': {
            'default': 1,
            'bounds': (1e-6, 100),
            'description': 'Diffusion time constant (s)'
        }
    },

    # Gerischer impedance
    'G': {
        'R': {
            'default': 100,
            'bounds': (1e-3, 1e6),
            'description': 'Gerischer resistance (Ω)'
        },
        'tau': {
            'default': 1,
            'bounds': (1e-6, 100),
            'description': 'Characteristic time (s)'
        }
    },

    # Havránek impedance
    'H': {
        'R': {
            'default': 100,
            'bounds': (1e-3, 1e6),
            'description': 'Havránek resistance (Ω)'
        },
        'tau': {
            'default': 1,
            'bounds': (1e-6, 100),
            'description': 'Time constant (s)'
        },
        'alpha': {
            'default': 0.75,
            'bounds': (0.001, 1.0),
            'description': 'Exponent (0-1)'
        }
    },

    # Transmission Line Model (if implemented)
    'TLM': {
        'R': {
            'default': 100,
            'bounds': (1e-3, 1e6),
            'description': 'Longitudinal resistance (Ω)'
        },
        'C': {
            'default': 1e-6,
            'bounds': (1e-12, 1e-2),
            'description': 'Transverse capacitance (F)'
        },
        'length': {
            'default': 1e-6,
            'bounds': (1e-9, 1e-3),
            'description': 'Characteristic length (m)'
        }
    }
}

#def extract_base_type(name: str) -> str:
#    """Extract component type from full parameter name (e.g., 'CPEa' -> 'CPE')"""
#    match = re.match(r'^([A-Za-z]+)', name)
#    return match.group(1) if match else name

def extract_base_type(name: str) -> str:
    for known_type in sorted(PARAMETER_CONFIG.keys(), key=len, reverse=True):
        if name.startswith(known_type):
            return known_type
    return name  # fallback

def initialize_parameters(param_names):
    params = {}
    for name in param_names:
        base_type = extract_base_type(name)
        config = PARAMETER_CONFIG.get(base_type, {})

        if 'default' in config:
            params[name] = config['default']
        else:
            params[name] = {k: v['default'] for k, v in config.items() if 'default' in v}
    return params

def initialize_bounds(param_names):
    bounds = {}
    for name in param_names:
        base_type = extract_base_type(name)
        config = PARAMETER_CONFIG.get(base_type, {})

        if 'bounds' in config:
            bounds[name] = config['bounds']
        else:
            for subkey, subconfig in config.items():
                bounds[f"{name}_{subkey}"] = subconfig['bounds']
    return bounds
