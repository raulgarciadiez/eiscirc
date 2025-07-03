from .impedance_parameters_default import PARAMETER_CONFIG, initialize_parameters, initialize_bounds

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass  

@dataclass
class FitResult:
    parameters: Dict[str, float]       # Fitted parameters (e.g., {'R1': 500})
    errors: Dict[str, float]           # Standard deviations (e.g., {'R1': 10})
    rmse: float                        # Root-mean-square error
    covariance: np.ndarray             # Full covariance matrix
    Z_fit: np.ndarray                  # Fitted impedance (Z_real + Z_imag)

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
        
        self.fixed_params = {}
        #self.bounds = {}
        self.initial_guess = {}
        self._fitted_params = None
        self._covariance = None

        # Initialize all parameters with physical defaults
        #self._initialize_parameters()
        
        # Use shared initialization
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize using shared config"""
        # 1. Ensure model parameters are initialized
        if not hasattr(self.model, '_params') or not self.model._params:
            self.model._params = initialize_parameters(self.model.param_names)
        
        # 2. Initialize bounds
        self.bounds = initialize_bounds(self.model.param_names)
        
        # 3. Apply any fixed params or initial guesses
        self._apply_user_constraints()
        
        # 4. Calculate initial impedance if possible
        self._update_model_impedance()
        
    def _apply_user_constraints(self):
        """Apply user-provided fixed params and initial guesses"""
        # Apply fixed parameters
        for name, value in self.fixed_params.items():
            self._update_model_param(name, value)
            
        # Apply initial guesses (only to non-fixed parameters)
        for name, value in self.initial_guess.items():
            if not any(name.startswith(p.split('_')[0]) for p in self.fixed_params):
                self._update_model_param(name, value)
    
    def _update_model_param(self, name, value):
        """Update parameter with bounds checking"""
        # Extract base parameter type for bounds checking
        param_type = ''.join([c for c in name.split('_')[0] if not c.isdigit()])
        
        if '_' in name:  # Sub-parameter
            param, subkey = name.split('_', 1)
            bound = PARAMETER_CONFIG[param_type][subkey]['bounds']
            if not (bound[0] <= value <= bound[1]):
                raise ValueError(f"{name} must be in {bound}")
                
            if param not in self.model._params:
                self.model._params[param] = {}
            self.model._params[param][subkey] = value
            
        else:  # Simple parameter
            bound = PARAMETER_CONFIG[param_type]['bounds']
            if not (bound[0] <= value <= bound[1]):
                raise ValueError(f"{name} must be in {bound}")
            self.model._params[name] = value



    def _setup_bounds(self):
        """Configure bounds from shared config"""
        self.bounds = {}
        for name in self.model.param_names:
            base_type = ''.join([c for c in name if not c.isdigit()])
            
            if base_type in PARAMETER_CONFIG:
                config = PARAMETER_CONFIG[base_type]
                
                if 'bounds' in config:  # Simple parameter
                    self.bounds[name] = config['bounds']
                else:  # Complex parameter
                    for subkey, subconfig in config.items():
                        self.bounds[f"{name}_{subkey}"] = subconfig['bounds']



    def _initialize_parameters_OLD(self):
        """Initialize all parameters with physical defaults and bounds"""
        for name in self.model.param_names:
            # Determine parameter type (R, C, CPE, Ws, etc.)
            param_type = ''.join([c for c in name if not c.isdigit()])
            
            if param_type in self.PARAM_DEFAULTS:
                defaults = self.PARAM_DEFAULTS[param_type]
                
                if isinstance(defaults, dict):  # Complex parameter (CPE, Ws, etc.)
                    # Initialize the parameter dictionary if needed
                    if name not in self.model._params or not isinstance(self.model._params[name], dict):
                        self.model._params[name] = {}
                    
                    # Set each sub-parameter with its default if not already set
                    for subkey, (default_val, bounds) in defaults.items():
                        if subkey not in self.model._params[name] or self.model._params[name][subkey] is None:
                            self.model._params[name][subkey] = default_val
                        
                        # Set bounds if not already specified
                        bound_key = f"{name}_{subkey}"
                        if bound_key not in self.bounds:
                            self.bounds[bound_key] = bounds
                else:  # Simple parameter (R, C, L)
                    default_val, bounds = defaults
                    if name not in self.model._params or self.model._params[name] is None:
                        self.model._params[name] = default_val
                    if name not in self.bounds:
                        self.bounds[name] = bounds
        self._update_model_impedance()

    def set_fixed_params(self, **fixed_params):
        """Set fixed parameters and immediately update the model"""
        for name, value in fixed_params.items():
            self.fixed_params[name] = value
            self._update_model_param(name, value)
        self._update_model_impedance()
        return self

    def set_initial_guess(self, **initial_guess):
        """Set initial guesses and update model"""
        for name, value in initial_guess.items():
            self.initial_guess[name] = value
            if not any(name.startswith(p.split('_')[0]) for p in self.fixed_params):
                self._update_model_param(name, value)
        self._update_model_impedance()
        return self

    def _update_model_param(self, name, value):
        """Update a single parameter in the model"""
        if '_' in name:  # Handle sub-parameters (e.g., CPE1_alpha)
            param, subkey = name.split('_', 1)
            if param not in self.model._params or not isinstance(self.model._params[param], dict):
                self.model._params[param] = {}  # Ensure dict exists
            self.model._params[param][subkey] = value
        else:
            self.model._params[name] = value

    def _update_model_impedance(self):
        """Update model's impedance if all parameters are set"""
        try:
            # Check if all parameters are initialized
            all_params_set = True
            for name, value in self.model._params.items():
                if value is None:
                    all_params_set = False
                    break
                if isinstance(value, dict):
                    if any(v is None for v in value.values()):
                        all_params_set = False
                        break
            
            if all_params_set:
                # Prepare parameters for impedance calculation
                calc_params = {}
                for name, value in self.model._params.items():
                    if isinstance(value, dict):
                        if name.startswith('CPE'):
                            calc_params[name] = (value['value'], value['alpha'])
                        elif name.startswith(('Ws', 'Wo', 'G')):
                            calc_params[name] = (value['R'], value['tau'])
                        elif name.startswith('H'):
                            calc_params[name] = (value['R'], value['tau'], value['alpha'])
                    else:
                        calc_params[name] = value
                
                # Calculate impedance
                Z_total = self.model._impedance_func(self.omega, **calc_params)
                self.model.Z_real = np.real(Z_total).copy()
                self.model.Z_imag = np.imag(Z_total).copy()
        except Exception as e:
            print(f"Impedance update warning: {str(e)}")

    def set_fixed_params_OLD(self, **fixed_params):
        """Set fixed parameters and immediately update the model"""
        for name, value in fixed_params.items():
            self.fixed_params[name] = value
            # Update model immediately
            if '_' in name:  # Handle sub-parameters (e.g., CPE1_alpha)
                param, subkey = name.split('_', 1)
                if param not in self.model._params:
                    self.model._params[param] = {}
                self.model._params[param][subkey] = value
            else:
                self.model._params[name] = value
        # Recalculate impedance if all parameters are set
        self._update_model_impedance()
        return self



    def set_bounds(self, **bounds):
        """Set parameter bounds (e.g., CPE1_alpha=(0.1, 0.9))."""
        self.bounds.update(bounds)
        return self


    def _prepare_fitting(self):
        """Prepare for fitting while respecting fixed params and initial guesses."""
        # Step 1: Initialize ONLY non-fixed parameters
        for name in self.model.param_names:
            # Skip if parameter is fixed
            if any(name.startswith(p.split('_')[0]) for p in self.fixed_params):
                continue
                
            # Initialize only if not set and not in initial_guess
            if name not in self.model._params or self.model._params[name] is None:
                if name in self.initial_guess:
                    # Use user-provided initial guess
                    if '_' in name:  # Handle sub-parameters
                        param, subkey = name.split('_', 1)
                        if param not in self.model._params:
                            self.model._params[param] = {}
                        self.model._params[param][subkey] = self.initial_guess[name]
                    else:
                        self.model._params[name] = self.initial_guess[name]
                else:
                    # Default initialization (preserving your structure)
                    if isinstance(self.model._params.get(name, None), dict):
                        self.model._params[name] = {k: 1.0 for k in self.model._params[name].keys()}
                    else:
                        self.model._params[name] = 1.0

        # Step 2: Prepare bounds and free parameters
        all_param_names = []
        for name in self.model.param_names:
            if isinstance(self.model._params[name], dict):
                for subkey in self.model._params[name].keys():
                    all_param_names.append(f"{name}_{subkey}")
            else:
                all_param_names.append(name)

        # Exclude fixed parameters
        self.free_param_names = [
            name for name in all_param_names
            if not any(name.startswith(p.split('_')[0]) for p in self.fixed_params)
        ]

        # Prepare initial values and bounds
        x0 = []
        lb = []
        ub = []
        for name in self.free_param_names:
            # Get value
            if '_' in name:
                param, subkey = name.split('_', 1)
                value = self.model._params[param][subkey]
            else:
                value = self.model._params[name]
            x0.append(value)
            
            # Get bounds
            if name in self.bounds:
                lb.append(self.bounds[name][0])
                ub.append(self.bounds[name][1])
            else:
                lb.append(-np.inf)
                ub.append(np.inf)

        return np.array(x0), (lb, ub)


    def fit(self, maxfev: int = 10000) -> FitResult:
        """Run fitting and return a FitResult object"""
        x0, (lb, ub) = self._prepare_fitting()
        
        # Create wrapper that includes fixed parameters
        def wrapped_impedance(omega, *free_params):
            # Build complete parameter set
            params = {}
            
            # 1. Add fixed parameters
            for fixed in self.fixed_params:
                if '_' in fixed:
                    param, subkey = fixed.split('_', 1)
                    if param not in params:
                        params[param] = {}
                    params[param][subkey] = self.fixed_params[fixed]
                else:
                    params[fixed] = self.fixed_params[fixed]
            
            # 2. Add free parameters
            for name, value in zip(self.free_param_names, free_params):
                if '_' in name:
                    param, subkey = name.split('_', 1)
                    if param not in params:
                        params[param] = {}
                    params[param][subkey] = value
                else:
                    params[name] = value
            
            # Calculate impedance
            return self.model.impedance(omega, **params)
        
        # Run fitting
        popt, pcov = curve_fit(
            wrapped_impedance,
            self.omega,
            self.Z_data,
            p0=x0,
            bounds=(lb, ub),
            maxfev=maxfev
        )
        
        # Update model with fitted parameters
        fitted_params = {}
        for name, value in zip(self.free_param_names, popt):
            if '_' in name:
                param, subkey = name.split('_', 1)
                if param not in fitted_params:
                    fitted_params[param] = {}
                fitted_params[param][subkey] = value
            else:
                fitted_params[name] = value
        
        # Merge with fixed parameters
        for fixed in self.fixed_params:
            if '_' in fixed:
                param, subkey = fixed.split('_', 1)
                if param not in fitted_params:
                    fitted_params[param] = {}
                fitted_params[param][subkey] = self.fixed_params[fixed]
            else:
                fitted_params[fixed] = self.fixed_params[fixed]
        
        # Update model and calculate final impedance
        self.model.set_params(**fitted_params)
        Z_fit = self.model.impedance(self.omega)
        
        # Calculate errors and return
        residuals = self.Z_data - Z_fit
        rmse = np.sqrt(np.mean(residuals**2))
        perr = np.sqrt(np.diag(pcov))
        
        return FitResult(
            parameters=fitted_params,
            errors=dict(zip(self.free_param_names, perr)),
            rmse=rmse,
            covariance=pcov,
            Z_fit=Z_fit
        )