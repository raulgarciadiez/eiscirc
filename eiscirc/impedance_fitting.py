import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional, List

class ImpedanceFitter:
    def __init__(self, model, frequencies: np.ndarray, Z_data: np.ndarray):
        """
        Args:
            model: Initialized ImpedanceModel instance.
            frequencies: Array of frequencies (Hz).
            Z_data: Measured impedance (concatenated [Z_real, Z_imag]).
        """
        self.model = model
        self.frequencies = frequencies
        self.omega = 2 * np.pi * frequencies
        self.Z_data = Z_data
        
        # Internal state
        self.fixed_params: Dict[str, float] = {}
        self.bounds: Dict[str, Tuple[float, float]] = {}
        self.initial_guess: Dict[str, float] = {}
        self._fitted_params: Optional[Dict[str, float]] = None
        self._covariance: Optional[np.ndarray] = None

    def set_fixed_params(self, **fixed_params):
        """Fix parameters during fitting (e.g., R0=100)."""
        self.fixed_params.update(fixed_params)
        return self

    def set_bounds(self, **bounds):
        """Set parameter bounds (e.g., CPE1_alpha=(0.1, 0.9))."""
        self.bounds.update(bounds)
        return self

    def set_initial_guess(self, **initial_guess):
        """Override default initial guesses."""
        self.initial_guess.update(initial_guess)
        return self

    def _prepare_fitting(self):
        """Initialize missing parameters and prepare free/bounded params."""
        # Step 1: Auto-initialize missing parameters
        for name in self.model.param_names:
            if name not in self.model._params or self.model._params[name] is None:
                if isinstance(self.model._params.get(name, None), dict):
                    # Initialize CPE/Ws/G/H parameters
                    self.model._params[name] = {k: 1.0 for k in self.model._params[name].keys()}
                else:
                    # Initialize R, C, L, etc.
                    self.model._params[name] = 1.0

        # Step 2: Extract free parameters and bounds (same as before)
        all_param_names = []
        for name in self.model.param_names:
            if isinstance(self.model._params[name], dict):
                for subkey in self.model._params[name].keys():
                    all_param_names.append(f"{name}_{subkey}")
            else:
                all_param_names.append(name)

        # Rest of the method remains identical...
        self.free_param_names = [
            name for name in all_param_names
            if not any(name.startswith(p) for p in self.fixed_params)
        ]

        x0 = []
        for name in self.free_param_names:
            if name in self.initial_guess:
                x0.append(self.initial_guess[name])
            else:
                if '_' in name:
                    param, subkey = name.split('_', 1)
                    x0.append(self.model._params[param][subkey])
                else:
                    x0.append(self.model._params[name])

        lb = [-np.inf] * len(self.free_param_names)
        ub = [np.inf] * len(self.free_param_names)
        for name, (low, high) in self.bounds.items():
            if name in self.free_param_names:
                idx = self.free_param_names.index(name)
                lb[idx] = low
                ub[idx] = high

        return np.array(x0), (lb, ub)

    def get_full_parameters(self) -> dict:
        """Return a dict of all parameters (fitted + fixed)."""
        fitted_params = self._fitted_params or {}
        full_params = {}
        
        # Merge fitted and fixed parameters
        for name in self.model.param_names:
            if name in self.fixed_params:
                full_params[name] = self.fixed_params[name]
            elif isinstance(self.model._params[name], dict):
                # Handle multi-param elements (CPE, Ws, etc.)
                sub_params = {}
                for subkey in self.model._params[name].keys():
                    param_key = f"{name}_{subkey}"
                    if param_key in fitted_params:
                        sub_params[subkey] = fitted_params[param_key]
                    else:
                        sub_params[subkey] = self.model._params[name][subkey]
                full_params[name] = sub_params
            else:
                param_key = name
                if param_key in fitted_params:
                    full_params[name] = fitted_params[param_key]
                else:
                    full_params[name] = self.model._params[name]
        
        return full_params

    def _impedance_wrapper(self, omega, *free_params):
        """Wrapper for curve_fit to update model with free parameters."""
        params = {}
        for name, value in zip(self.free_param_names, free_params):
            if '_' in name:
                param, subkey = name.split('_', 1)
                if param not in params:
                    params[param] = {}
                params[param][subkey] = value
            else:
                params[name] = value

        # Update model and compute impedance
        self.model.set_params(**params)
        return self.model.impedance(omega)

    def fit(self, maxfev: int = 10000) -> Dict[str, float]:
        """Run fitting and return fitted parameters."""
        x0, (lb, ub) = self._prepare_fitting()
        
        popt, pcov = curve_fit(
            self._impedance_wrapper, 
            self.omega, 
            self.Z_data,
            p0=x0,
            bounds=(lb, ub),
            maxfev=maxfev
        )

        # Store results
        self._fitted_params = dict(zip(self.free_param_names, popt))
        self._covariance = pcov
        
        # Update model with fitted values
        fitted_params_full = {**self.fixed_params, **self._fitted_params}
        self.model.set_params(**fitted_params_full)
        
        return self._fitted_params

    def monte_carlo(self, n_runs: int = 100, noise_scale: float = 0.02):
        """Monte Carlo uncertainty analysis (placeholder for future implementation)."""
        pass

    def add_regularization(self, penalty_func):
        """Add regularization term (placeholder for future implementation)."""
        pass