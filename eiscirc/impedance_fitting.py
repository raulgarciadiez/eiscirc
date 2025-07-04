from .impedance_parameters_default import PARAMETER_CONFIG, initialize_parameters, initialize_bounds, extract_base_type


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
        #self.initial_guess = {}
        self._fitted_params = None
        self._covariance = None

        self._initialize_parameters()
    
    @property
    def bounds(self):
        return self.model.bounds
    
    @property
    def initial_guess(self):
        class GuessProxy:
            def __init__(self, model):
                self._model = model

            def __getitem__(self, key):
                return self._model.get_all_params()[key]

            def __setitem__(self, key, value):
                self._model.set_params(**{key: value})

            def items(self):
                return self._model.get_all_params().items()

            def as_dict(self):
                return self._model.get_all_params()

            def __repr__(self):
                return repr(self._model.get_all_params())

        return GuessProxy(self.model)

        
    def _initialize_parameters(self):
        if not hasattr(self.model, '_params') or not self.model._params:
            self.model._params = initialize_parameters(self.model.param_names)
        self._apply_user_constraints()
        self._update_model_impedance()

        #self.bounds = initialize_bounds(self.model.param_names)

    #@property
    #def initial_guess(self):
    #    return self.model.get_all_params()

    def _apply_user_constraints(self):
        for name, value in self.fixed_params.items():
            self.model.set_params(**{name: value})
        for name, value in self.initial_guess.items():   #############################
            if not any(name.startswith(p.split('_')[0]) for p in self.fixed_params):
                self.model.set_params(**{name: value})
    
    def set_fixed_params(self, **fixed_params):
        for name, value in fixed_params.items():
            self.fixed_params[name] = value
            self.model.set_params(**{name: value})
        self._update_model_impedance()
        return self

    def set_initial_guess(self, **initial_guess):
        for name, value in initial_guess.items():    ##########################################
            self.initial_guess[name] = value
            if not any(name.startswith(p.split('_')[0]) for p in self.fixed_params):
                self.model.set_params(**{name: value})
        self._update_model_impedance()
        return self

    def _prepare_fitting(self):
        all_params = self.model.get_all_params()
        self.free_param_names = [k for k in all_params if not any(k.startswith(p.split('_')[0]) for p in self.fixed_params)]

        #x0 = [all_params[k] for k in self.free_param_names]
        #lb = [self.bounds.get(k, (-np.inf, np.inf))[0] for k in self.free_param_names]
        #ub = [self.bounds.get(k, (-np.inf, np.inf))[1] for k in self.free_param_names]

        x0 = [self.initial_guess[k] for k in self.free_param_names]
        lb = [self.bounds[k][0] for k in self.free_param_names]
        ub = [self.bounds[k][1] for k in self.free_param_names]


        return np.array(x0), (lb, ub)

    def _update_model_impedance(self):
        """Recalculate impedance using model.impedance if parameters are valid"""
        try:
            self.model.impedance(self.omega)
        except Exception as e:
            print(f"Impedance update warning: {str(e)}")


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