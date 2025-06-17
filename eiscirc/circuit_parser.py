import numpy as np
from scipy.optimize import curve_fit,minimize
import pandas as pd
from galvani import BioLogic
import os,re
import csv

# Adding a class to compute fit quality
class FitQuality:
    @staticmethod
    def compute_residuals(actual, predicted):
        """Compute the residuals between actual data and model predictions."""
        return actual - predicted
    
    @staticmethod
    def mean_squared_error(actual, predicted):
        """Calculate Mean Squared Error (MSE)."""
        residuals = FitQuality.compute_residuals(actual, predicted)
        mse = np.mean(residuals ** 2)
        return mse
    
    @staticmethod
    def root_mean_squared_error(actual, predicted):
        """Calculate Root Mean Squared Error (RMSE)."""
        mse = FitQuality.mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        return rmse
    
    @staticmethod
    def r_squared(actual, predicted):
        """Calculate the R-squared value."""
        residuals = FitQuality.compute_residuals(actual, predicted)
        ss_res = np.sum(residuals ** 2)  # Residual sum of squares
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot)
        return r2

    @staticmethod
    def adjusted_r_squared(actual, predicted):
        """Calculate the Adjusted R-squared value."""
        n = len(actual)  # Number of data points
        residuals = FitQuality.compute_residuals(actual, predicted)
        ss_res = np.sum(residuals ** 2)  # Residual sum of squares
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot)  # Standard R-squared  
        # Adjusted R-squared
        r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - 1)
        return r2_adj




# FitManager class to handle the fitting process
class FitManager:
    def __init__(self, data_handler, model_as_string = "R-C"):
        self.data_handler = data_handler
        self.previous_fitted_params = None  # Store the fitted parameters from the previous dataset

        self.Ewe = self.data_handler.Ewe
        self.sample_ID = self.data_handler.sample_ID
        self.root_folder=self.data_handler.root_folder

        self.model_as_string = model_as_string
        input_model = parse_circuit(model_as_string)
        self.circuit = ImpedanceModel(input_model)
    
    def _convert_list_to_dict(self, param_list, param_names, reference_dict):
        """
        Convert a list of parameters back into a dictionary format.
        Handles cases where parameters are tuples (e.g., CPE elements).
        """
        param_dict = {}
        index = 0
        for key in param_names:
            ref_value = reference_dict[key]
            if isinstance(ref_value, tuple):  # Example: "CPE1": (C_value, alpha)
                param_dict[key] = tuple(param_list[index:index+len(ref_value)])
                index += len(ref_value)
            else:
                param_dict[key] = param_list[index]
                index += 1
        return param_dict
    
    def fit_model_parser(self, fmin=None, fmax=None, initial_guess=None, bounds=None):
        """Fit the model to a data set."""
        filtered_df = self.data_handler.filter_frequencies(fmin, fmax)
        omega, Z_data = self.data_handler.prepare_data(filtered_df)

        param_names = list(initial_guess.keys())
        param_values = []
        bounds_list = []

        for key in param_names:
            value = initial_guess[key]
            bound = bounds[key]

            if isinstance(value, tuple):  # Example: "CPE1": (C_guess, alpha_guess)
                param_values.extend(value)
                bounds_list.extend(bound)  # Already in tuple (min, max) format
            else:
                param_values.append(value)
                bounds_list.append(bound)

        # Convert bounds to tuple of tuples format
        bounds_tuple = tuple(bounds_list)

        def objective_function(param_list):
            # Convert list back to dictionary format
            param_dict = self._convert_list_to_dict(param_list, param_names, initial_guess)

            # Compute simulated impedance
            Z_simulated = self.circuit.impedance(omega, **param_dict)

            # Compute least squares difference
            return np.sum(np.abs(Z_simulated - Z_data) ** 2)

        # Run the minimization with a dictionary-based optimizer
        result = minimize(
            objective_function,
            param_values,  # initial guess as a list
            bounds=bounds_tuple,
            method="L-BFGS-B"  # Robust optimization method
        )
        optimized_params = self._convert_list_to_dict(result.x, param_names, initial_guess)
        fit_quality = FitQuality.adjusted_r_squared(Z_data, self.circuit.impedance(omega, **optimized_params))

        write_fit_results_to_file(
            model=convert_circuit_to_filename(self.model_as_string),  # Use the model name dynamically
            root_folder=self.root_folder,
            sample_ID= self.sample_ID,
            Ewe=self.Ewe,
            fmin=fmin,
            fmax=fmax,
            initial_guess=initial_guess,
            params=optimized_params,
            bounds=bounds,
            residual=fit_quality
        )

        return optimized_params,fit_quality
    
# DataHandler class to manage data import, filtering, and transformation
class DataHandler:
    def __init__(self, dataframe,root_folder, sample_ID): # filepath
        # Load data from file
        self.root_folder = root_folder
        self.sample_ID = sample_ID
        self.df = dataframe
        self.Ewe = self.df["<Ewe>/V"].mean()

    # Method to filter frequencies based on a range
    def filter_frequencies(self, fmin=4, fmax=1e6):
        """Filter data based on a frequency range."""
        self.filtered_df = self.df.copy()
        self.filtered_df = self.filtered_df[(self.filtered_df['freq/Hz'] >= fmin) & (self.filtered_df['freq/Hz'] <= fmax)& (self.filtered_df['-Im(Z)/Ohm'] > 0)]
        return self.filtered_df

    # Prepare the data for fitting (frequencies and impedance)
    def prepare_data(self, filtered_df):
        frequencies = filtered_df['freq/Hz'].values
        omega = 2 * np.pi * frequencies
        real = filtered_df['Re(Z)/Ohm'].values
        imaginary = -filtered_df['-Im(Z)/Ohm'].values  # Correct sign
        Z_data = np.concatenate((real, imaginary))
        return omega, Z_data

class ImpedanceModel:
    def __init__(self, circuit_structure):
        self.circuit_structure = circuit_structure

    def impedance(self, omega, **params):
        """Compute the impedance of the entire circuit."""
        Z_dict = self._create_impedance_dict(params)
        print (Z_dict)
        Z_total = self._calculate_impedance(self.circuit_structure, Z_dict, omega)       
        # Extract real and imaginary parts from Z_total
        real_part = np.real(Z_total)
        imag_part = np.imag(Z_total)
        # Ensure the real and imaginary parts are the same length before concatenating
        if len(real_part) != len(imag_part):
            raise ValueError(f"Real part and imaginary part have different lengths: {len(real_part)} != {len(imag_part)}")
        # Concatenate the real and imaginary parts
        return np.concatenate((real_part, imag_part))

    def _create_impedance_dict(self, params):
        """Create a dictionary of impedances for each component."""
        Z_dict = {}
        for key, value in params.items():
            if key.startswith("R"):  # Resistor impedance is just the resistance value
                Z_dict[key] = value
            elif key.startswith("C") and not key.startswith("CPE"):  # Capacitor impedance: Z_C = 1 / (j * omega * C)
                Z_dict[key] = lambda omega, value=value: 1 / (1j * omega * value)
            elif key.startswith("CPE"):  # CPE impedance: Z_CPE = 1 / (j * omega * C) ^ alpha
                CPE_value, alpha = value  # (CPE_value, alpha)
                Z_dict[key] = lambda omega, CPE_value=CPE_value, alpha=alpha: 1 / (1j * omega * CPE_value)**alpha
            elif key.startswith("W"):  # Warburg impedance: Z_W = W / sqrt(j * omega)
                Z_dict[key] = lambda omega, value=value: value / np.sqrt(1j * omega)
            else:
                raise ValueError(f"Unknown component type: {key}")
        return Z_dict

    def _calculate_impedance(self, structure, Z_dict, omega):
        """Recursively evaluate the impedance of the circuit structure."""
        if isinstance(structure, str):  # A simple component
            if structure.startswith("R") or structure.startswith("C") or structure.startswith("CPE") or structure.startswith("W"):
                return Z_dict[structure](omega) if callable(Z_dict[structure]) else Z_dict[structure]
            else:
                raise ValueError(f"Unknown element: {structure}")
        elif isinstance(structure, tuple):  # An operation (series or parallel)
            operator = structure[0]
            if operator == "series":
                if len(structure) < 3:
                    raise ValueError(f"Series combination requires at least two components, got: {structure}")               
                # Process the first component in the series
                Z1 = self._calculate_impedance(structure[1], Z_dict, omega)               
                # Process the rest of the components in the series
                for i in range(2, len(structure)):
                    Z2 = self._calculate_impedance(structure[i], Z_dict, omega)
                    Z1 += Z2  # Sum the impedances for series connection                
                return Z1

            elif operator == "parallel":
                if len(structure) < 3:
                    raise ValueError(f"Parallel combination requires at least two components, got: {structure}")               
                # Process the first component in the parallel connection
                Z1 = self._calculate_impedance(structure[1], Z_dict, omega)            
                # Process the rest of the components in parallel
                for i in range(2, len(structure)):
                    Z2 = self._calculate_impedance(structure[i], Z_dict, omega)
                    Z1 = 1 / (1 / Z1 + 1 / Z2)  # Parallel connection formula           
                return Z1
            else:
                raise ValueError(f"Unknown operator: {operator}")
        else:
            raise ValueError("Unknown structure format")

    def display_structure(self):
        """Recursively display the structure of the circuit."""
        print(self._display_structure_recursive(self.circuit_structure))

    def _display_structure_recursive(self, structure):
        """Helper function to print the structure in a readable format."""
        if isinstance(structure, str):
            return structure
        elif isinstance(structure, tuple):
            operator = structure[0]
            
            # If 'series' or 'parallel' has more than two components, process all
            components = [self._display_structure_recursive(comp) for comp in structure[1:]]
            
            # Format output correctly
            return f"({f' {operator} '.join(components)})"
        else:
            return "Unknown structure"
        

    def generate_filename(self):
        """Generate a filename-safe string representation of the circuit."""
        filename = self._structure_to_filename_string(self.circuit_structure)

        # Remove any unnecessary spaces
        filename = filename.replace(" ", "")

        return filename

    def _structure_to_filename_string(self, structure):
        """Recursively convert tuple structure to a filename-safe format."""
        if isinstance(structure, str):
            return structure  # Directly return component names (e.g., R0, C1)
        elif isinstance(structure, tuple):
            operator = "_" if structure[0] == "series" else "||"  # "_" for series, "||" for parallel
            
            # Ensure all components are processed
            elements = [self._structure_to_filename_string(sub) for sub in structure[1:]]
            
            # Join elements with the correct operator
            return f"({operator.join(elements)})"
        else:
            raise ValueError("Unknown structure format")
        
    def _convert_list_to_dict(self, param_list, param_names, reference_dict):
        """
        Convert a list of parameters back into a dictionary format.
        Handles cases where parameters are tuples (e.g., CPE elements).
        """
        param_dict = {}
        index = 0
        for key in param_names:
            ref_value = reference_dict[key]
            if isinstance(ref_value, tuple):  # Example: "CPE1": (C_value, alpha)
                param_dict[key] = tuple(param_list[index:index+len(ref_value)])
                index += len(ref_value)
            else:
                param_dict[key] = param_list[index]
                index += 1
        return param_dict
        
def parse_circuit(expression):
    # Remove all whitespace from the expression
    expression = expression.replace(" ", "")

    # Define a recursive function to parse the expression
    def parse(expr):
        # Base case: if the expression is a single component (e.g., "R0", "C1")
        if re.match(r'^[A-Za-z]+\d+$', expr):
            return expr

        # Handle parentheses
        if expr.startswith('(') and expr.endswith(')'):
            return parse(expr[1:-1])  # Remove parentheses and parse the inner expression

        # Check for series connections (denoted by '-')
        # Split on '-' but not inside parentheses
        series_parts = []
        depth = 0
        start = 0
        for i, char in enumerate(expr):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            if char == '-' and depth == 0:
                series_parts.append(expr[start:i])
                start = i + 1
        if start != 0:
            series_parts.append(expr[start:])
            if len(series_parts) > 1:
                return ("series",) + tuple(parse(part) for part in series_parts)

        # Check for parallel connections (denoted by '//')
        # Split on '//' but not inside parentheses
        parallel_parts = []
        depth = 0
        start = 0
        for i, char in enumerate(expr):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            if expr[i:i+2] == '//' and depth == 0:
                parallel_parts.append(expr[start:i])
                start = i + 2
        if start != 0:
            parallel_parts.append(expr[start:])
            if len(parallel_parts) > 1:
                return ("parallel",) + tuple(parse(part) for part in parallel_parts)

        # If no operator is found, return the expression as is
        return expr

    return parse(expression)

def generate_random_initial_guess_from_bounds(bounds):
    min_bounds, max_bounds = bounds
    random_initial_guess = [np.random.uniform(low=lb, high=ub) for lb, ub in zip(min_bounds, max_bounds)]
    return random_initial_guess

def flatten_values_dict(param_dict):
    param_values = []
    for value in param_dict.values():
        if isinstance(value, tuple):  # If it's a tuple (e.g., CPE1: (C_value, alpha))
            param_values.extend(value)  # Add all tuple elements to the list
        else:
            param_values.append(value)  # Add single values directly
    return param_values

def flatten_names_dict(param_dict):
    param_names = []
    for key, value in param_dict.items():
        if isinstance(value, tuple):  # If it's a tuple (e.g., CPE1: (C_value, alpha))
            for i in range(len(value)):
                param_names.append(f"{key}_{chr(97 + i)}")  # Append like CPE1_a, CPE1_b
        else:
            param_names.append(key)  # Just append the name for non-tuple values
    return param_names

def flatten_bounds_dict(bounds):
    bounds_list = []

    for key, value in bounds.items():
        if isinstance(value[0], tuple) or isinstance(value[0], list):  # If it's a list of tuples (like CPE elements)
            for sub_value in value:
                bounds_list.append(sub_value[0])  # Min value
            for sub_value in value:
                bounds_list.append(sub_value[1])  # Max value
        else:  # If it's a simple tuple
            bounds_list.append(value[0])  # Min value
            bounds_list.append(value[1])  # Max value
    return bounds_list

def write_fit_results_to_file(model, root_folder, sample_ID, Ewe, fmin, fmax, initial_guess, params, bounds, residual):
    """Writes the fitting results to a CSV file."""
    log_filename = f"{(root_folder.rstrip("/"))}_{sample_ID}_{model}_fit_log.csv"
    filepath = os.path.join(os.getcwd(), log_filename)

    # Check if file exists; if not, write the header
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file doesn't exist
        if not file_exists:
            # Headers for initial guesses
            header = ['Ewe', 'fmin', 'fmax'] + [f"Initial_{param}" for param in flatten_names_dict(initial_guess)]
            
            # Headers for bounds (min and max for each parameter)
            for param in flatten_names_dict(initial_guess):
                header.append(f"{param}_bound_min")
            for param in flatten_names_dict(initial_guess):
                header.append(f"{param}_bound_max")

            # Headers for fitted parameters and residual
            header += [f"Fit_{param}" for param in flatten_names_dict(initial_guess)] + ['Residual']
            # Write the header row
            writer.writerow(header)
            
        # Write the row of fitting results
        row = [Ewe, fmin, fmax] + list(flatten_values_dict(initial_guess))  # Start with Ewe, fmin, fmax, and initial guess
        row += flatten_bounds_dict(bounds)  # Add bounds min and max
        row += list(flatten_values_dict(params))  + [residual]  # Add fitted parameters and residual

        # Write the data row
        writer.writerow(row)

def find_files(root_folder, extensions):
    file_list = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if all(substring in file for substring in extensions):
                file_list.append(os.path.join(root, file))
    if len(file_list) == 0:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith(extensions[-1]):
                    file_list.append(os.path.join(root, file))
    else:
        pass
    return file_list

def convert_files_to_pd_dataframes(mpr_files):
    groups = []
    for mpr_file_path in mpr_files:
        mpr_file = BioLogic.MPRfile(mpr_file_path)
        df = pd.DataFrame(mpr_file.data)

        std_threshold = 1e-4

        # Sort the DataFrame by column Z for grouping
        df = df.sort_values(by="<Ewe>/V").reset_index(drop=True)

        # Initialize variables
        current_group = [df.iloc[0]]  # Start with the first row

        # Iterate through the DataFrame
        for i in range(1, len(df)):
            temp_group = pd.DataFrame(current_group + [df.iloc[i]])
            if temp_group["<Ewe>/V"].std() > std_threshold:
                # Save the current group if the condition is met
                groups.append(pd.DataFrame(current_group))
                # Start a new group with the current row
                current_group = [df.iloc[i]]
            else:
                # Add the current row to the current group
                current_group.append(df.iloc[i])

        # Add the last group
        if current_group:
            groups.append(pd.DataFrame(current_group))
    return groups

def convert_circuit_to_filename(circuit_str):
    # Replace series connections '-' with '_s_'
    circuit_str = circuit_str.replace(' - ', '_s_')
    
    # Replace parallel connections '//' with '_p_'
    circuit_str = circuit_str.replace(' // ', '_p_')
    
    # Remove spaces
    circuit_str = circuit_str.replace(' ', '')
    
    # Replace '(' with 'l' and ')' with 'r' to represent hierarchy
    circuit_str = circuit_str.replace('(', 'l').replace(')', 'r')
    
    # Remove any trailing underscores
    circuit_str = circuit_str.strip('_')
    
    # Replace multiple underscores with a single underscore, except for hierarchy markers
    circuit_str = re.sub(r'(?<!_)_+(?!_)', '_', circuit_str)
    
    return circuit_str

