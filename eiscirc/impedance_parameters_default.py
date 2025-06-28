# impedance_parameters.py
import numpy as np

PARAMETER_CONFIG = {
    'R': {
        'default': 100,
        'bounds': (1e-3, 1e6),
        'description': 'Resistance (Ω)'
    },
    'C': {
        'default': 1e-6,
        'bounds': (1e-12, 1e-2), 
        'description': 'Capacitance (F)'
    },
    'L': {
        'default': 1e-3,
        'bounds': (1e-9, 1),  # H
        'description': 'Inductance'
    },
    'W': {
        'default': 100,
        'bounds': (1e-3, 1e6),
        'description': 'Warburg element'
    },
    'CPE': {
        'value': {
            'default': 1e-6,
            'bounds': (1e-12, 1e-2),
            'description': 'CPE magnitude'
        },
        'alpha': {
            'default': 0.8,
            'bounds': (0.001, 1.0),  # Expanded from 0.1-0.9 to 0.001-1.0
            'description': 'CPE exponent'
        }
    },
    'H': {
        'R': {
            'default': 100,
            'bounds': (1e-3, 1e6),
            'description': 'Havránek resistance'
        },
        'tau': {
            'default': 1,
            'bounds': (1e-6, 100),
            'description': 'Time constant'
        },
        'alpha': {
            'default': 0.75,
            'bounds': (0.001, 1.0),
            'description': 'Havránek exponent'
        }
    },
    # ... other components with same pattern
}

def initialize_parameters(param_names):
    """Initialize parameters with defaults"""
    params = {}
    for name in param_names:
        param_type = ''.join([c for c in name if not c.isdigit()])
        config = PARAMETER_CONFIG.get(param_type, {})
        
        if not config:
            raise ValueError(f"Unknown parameter type: {param_type}")
            
        if 'default' in config:  # Simple parameter
            params[name] = config['default']
        else:  # Complex parameter
            params[name] = {k: v['default'] for k, v in config.items()}
    
    return params

def initialize_bounds(param_names):
    """Initialize bounds dictionary"""
    bounds = {}
    for name in param_names:
        param_type = ''.join([c for c in name if not c.isdigit()])
        config = PARAMETER_CONFIG.get(param_type, {})
        
        if 'bounds' in config:  # Simple parameter
            bounds[name] = config['bounds']
        else:  # Complex parameter
            for subkey, subconfig in config.items():
                bounds[f"{name}_{subkey}"] = subconfig['bounds']
    
    return bounds