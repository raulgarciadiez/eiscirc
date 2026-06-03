import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from galvani import BioLogic
import os
import csv

def generate_random_initial_guess_from_bounds(bounds):
    """
    Generate a random initial guess where each parameter is randomly selected
    between its corresponding bound min and bound max.

    Parameters:
    - bounds: tuple of two lists, (min_bounds, max_bounds), where each list contains
      the minimum and maximum bounds for each parameter.

    Returns:
    - A list representing the randomly generated initial guess.
    """
    min_bounds, max_bounds = bounds

    # Generate a random initial guess by sampling between min_bounds and max_bounds
    random_initial_guess = [np.random.uniform(low=lb, high=ub) for lb, ub in zip(min_bounds, max_bounds)]

    return random_initial_guess

def generate_random_initial_guess(initial_guess, perturbation=0.1):
    """
    Generate random initial guesses based on the provided initial guess.
    
    Parameters:
    - initial_guess: A list or array of initial guesses.
    - perturbation: Fractional perturbation applied to the initial guess values.
    
    Returns:
    - new_initial_guess: A list of new initial guesses.
    """
    new_initial_guess = []

    for guess in initial_guess:
        # Calculate the perturbation range
        perturb_range = perturbation * abs(guess) if guess != 0 else perturbation

        # Generate a new guess within the perturbation range
        new_guess = guess + np.random.uniform(-perturb_range, perturb_range)

        # Ensure the new guess is non-negative
        if new_guess < 0:
            new_guess = 0.1  # Set a minimum value for non-negativity

        new_initial_guess.append(float(new_guess))  # Convert to standard float

    return new_initial_guess

def generate_random_bounds(bounds, perturbation=0.1, min_distance=0.1):
    """
    Generate random bounds for the parameters within the specified perturbation range.

    Parameters:
    - bounds: A tuple containing the lower and upper bounds as lists.
    - perturbation: A fraction of the range to perturb the bounds by.
    - min_distance: Minimum distance between new_lb and new_ub.

    Returns:
    - new_bounds: A tuple of the new lower and upper bounds as lists.
    """
    lower_bounds, upper_bounds = bounds
    new_lower_bounds = []
    new_upper_bounds = []

    for lb, ub in zip(lower_bounds, upper_bounds):
        valid_bounds = False
        
        while not valid_bounds:
            # If the parameter is in logarithmic scale, perturb accordingly
            if lb > 0 and ub > 0:
                log_lb = np.log10(lb)
                log_ub = np.log10(ub)
                log_range = log_ub - log_lb

                # Calculate maximum perturbation size in logarithmic scale
                max_perturbation = perturbation * log_range

                # Generate new lower and upper bounds in logarithmic scale
                new_log_lb = log_lb + np.random.uniform(-max_perturbation, max_perturbation)
                new_log_ub = log_ub + np.random.uniform(-max_perturbation, max_perturbation)

                # Convert back to linear scale
                new_lb = 10 ** new_log_lb
                new_ub = 10 ** new_log_ub
            else:
                # Non-logarithmic parameters
                range_value = ub - lb
                max_perturbation = perturbation * range_value
                new_lb = lb + np.random.uniform(-max_perturbation, max_perturbation)
                new_ub = ub + np.random.uniform(-max_perturbation, max_perturbation)

            # Ensure new bounds maintain a minimum distance
            if new_ub <= new_lb + min_distance:
                continue

            # Ensure new lower bounds are greater than a minimum threshold (e.g., 0.1)
            if new_lb < 0.1:
                new_lb = 0.1

            # Ensure new bounds are non-negative and valid
            if new_lb < new_ub:
                valid_bounds = True  # Valid bounds found

        # Append valid bounds as standard Python floats
        new_lower_bounds.append(float(new_lb))
        new_upper_bounds.append(float(new_ub))

    return (new_lower_bounds, new_upper_bounds)


def write_fit_results_to_file(model, root_folder, filename, Ewe, fmin, fmax, initial_guess, params, bounds, residual):
    """Writes the fitting results to a CSV file."""
    log_filename = f"{os.path.basename(filename)}_{model.__class__.__name__}_fit_log.csv"
    filepath = os.path.join(os.getcwd(), log_filename)

    # Check if file exists; if not, write the header
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file doesn't exist
        if not file_exists:
            # Headers for initial guesses
            header = ['Ewe', 'fmin', 'fmax'] + [f"Initial_{param}" for param in model.param_names]
            
            # Headers for bounds (min and max for each parameter)
            for param in model.param_names:
                header.append(f"{param}_bound_min")
            for param in model.param_names:
                header.append(f"{param}_bound_max")

            # Headers for fitted parameters and residual
            header += [f"Fit_{param}" for param in model.param_names] + ['Residual']

            # Write the header row
            writer.writerow(header)

        # Write the row of fitting results
        row = [Ewe, fmin, fmax] + list(initial_guess)  # Start with Ewe, fmin, fmax, and initial guess
        row += bounds[0] + bounds[1]  # Add bounds min and max
        row += list(params)  + [residual]  # Add fitted parameters and residual

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

# Base model class with flexible parameters
class BaseModel:
    def __init__(self, initial_guess=None, bounds=None):
        self.params = initial_guess
        self.bounds = bounds
    
    # This method must be implemented in each derived model
    def impedance(self, omega, *params):
        raise NotImplementedError("Each model must implement the impedance function!")
    
    # Method to return initial guess dynamically
    def get_initial_guess(self):
        return self.params
    
    # Method to return bounds dynamically
    def get_bounds(self):
        return self.bounds

# RRCRC Model class
class RRCRCModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'C1', 'R2', 'C2']

    def impedance(self, omega, *params):
        R0, R1, C1, R2, C2 = params
        
        jomega = 1j * omega
        Z_C1 = 1 / (jomega * C1)
        Z_R1C1 = R1 + Z_C1
        Z_C2 = 1 / (jomega * C2)
        Z_R2C2 = R2 + Z_C2
        Z_total = R0 + 1 / (1 / Z_R1C1 + 1 / Z_R2C2)
        return np.concatenate((np.real(Z_total), np.imag(Z_total)))
    
class RRCModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'C1']

    def impedance(self, omega, *params):
        R0, R1, C1 = params
        
        #jomega = 1j * omega
        #Z_C1 = 1 / (jomega * C1)
        #Z_R1C1 = R1 + Z_C1
        #Z_R1C1 = R1 / (1 + (1j * omega *C1))
 
        #Z_total = R0 + 1 / (1 / Z_R1C1)
       # Z_total = R0 + Z_R1C1

        Z_C1 = 1 / (1j * omega * C1)
        Z_parallel = (R1 * Z_C1) / (R1 + Z_C1)

        Z_total = R0 + Z_parallel

        return np.concatenate((np.real(Z_total), np.imag(Z_total)))
    
class RRCCPEModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'fs1', "n1"]

    def impedance(self,omega, *params):#R0, R1, fs1, n1):
        """
        R-RC model with CPE equation.
        """
        R0, R1, fs1, n1 = params
        #Z_R1CPE1 = R1 / (1 + (1j * omega / fs1)**n1)
        #Z = R0 + Z_R1CPE1

        Z_C1 = 1 / (1j * omega / fs1)**n1
        Z_parallel = (R1 * Z_C1) / (R1 + Z_C1)

        Z_total = R0 + Z_parallel
        return np.concatenate((np.real(Z_total), np.imag(Z_total)))
 
class RRCRCCPEModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'fs1', "n1", 'R2', 'fs2', "n2"]

    def impedance(self,omega, *params):#R0, R1, fs1, n1, R2, fs2, n2):
        """
        R-RC-RC model with CPE equation.
        """
        R0, R1, fs1, n1, R2, fs2, n2 = params
        Z_R1CPE1 = R1 / (1 + (1j * omega / fs1)**n1)
        Z_R2CPE2 = R2 / (1 + (1j * omega / fs2)**n2)
        Z = R0 + Z_R1CPE1 + Z_R2CPE2
        return np.concatenate((np.real(Z), np.imag(Z)))


    
class R3RCPEModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'fs1', "n1", 'R2', 'fs2', "n2", 'R3', 'fs3', "n3"]

    def impedance(self,omega, *params):
        """
        R-RC-RC model with CPE equation.
        """
        R0, R1, fs1, n1, R2, fs2, n2, R3, fs3, n3 = params
        Z_R1CPE1 = R1 / (1 + (1j * omega / fs1)**n1)
        Z_R2CPE2 = R2 / (1 + (1j * omega / fs2)**n2)
        Z_R3CPE3 = R3 / (1 + (1j * omega / fs3)**n3)

        Z = R0 + Z_R1CPE1 + Z_R2CPE2 + Z_R3CPE3

        return np.concatenate((np.real(Z), np.imag(Z)))
    

class RRCRCCPESeriesWarburgModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'fs1', "n1", 'R2', 'fs2', "n2", "W"]

    def impedance(self,omega, *params):
        """
        R-RC-RC-Warburg model with CPE equation.
        """
        R0, R1, fs1, n1, R2, fs2, n2,W = params
        Z_R1CPE1 = R1 / (1 + (1j * omega / fs1)**n1)
        Z_R2CPE2 = R2 / (1 + (1j * omega / fs2)**n2)
        Z_Warburg = W / np.sqrt(1j * omega)
        Z = R0 + Z_R1CPE1 + Z_R2CPE2 + Z_Warburg
        return np.concatenate((np.real(Z), np.imag(Z)))
    
class RRCRCCPESeriesWarburgFiniteModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'fs1', "n1", 'R2', 'fs2', "n2", "AW", "BW"]

    def impedance(self,omega, *params):
        """
        R-RC-RC-Warburg (finite) model with CPE equation.
        Aw (float): Warburg coefficient.
        Bw (float): Warburg finite length parameter.

        """
        R0, R1, fs1, n1, R2, fs2, n2,AW,BW = params
        Z_R1CPE1 = R1 / (1 + (1j * omega / fs1)**n1)
        Z_R2CPE2 = R2 / (1 + (1j * omega / fs2)**n2)
        Z_Warburg = AW / (np.sqrt(1j * omega)) * np.tanh(BW * np.sqrt(1j * omega))  # Finite Warburg
        Z = R0 + Z_R1CPE1 + Z_R2CPE2 + Z_Warburg
        return np.concatenate((np.real(Z), np.imag(Z)))

class RRCRRCCPEModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'fs1', "n1", 'R2', 'fs2', "n2", "R3"]

    def impedance(self,omega, *params):

        R0, R1, fs1, n1, R2, fs2, n2, R3 = params
        """
        Impedance of R-(CPE//R)-(CPE//(R-R)) circuit.
        
        Parameters:
            omega: Array of angular frequencies.
            R0: Resistance in series.
            R1: Resistance in the first parallel branch.
            R2, R3: Resistances in the second series branch.
            fs1, n1: Parameters for the first CPE.
            fs2, n2: Parameters for the second CPE.
        
        Returns:
            Array of complex impedances.
        """
        # CPE impedances
        Z_CPE1 = 1 / ((1j * omega / fs1) ** n1)
        Z_CPE2 = 1 / ((1j * omega / fs2) ** n2)
        
        # First parallel branch: CPE1 // R1
        Z_branch1 = 1 / (1 / Z_CPE1 + 1 / R1)
        
        # Second series branch: R2 + R3
        Z_R2R3 = R2 + R3
        
        # Second parallel branch: CPE2 // (R2 + R3)
        Z_branch2 = 1 / (1 / Z_CPE2 + 1 / Z_R2R3)
        
        # Total impedance
        Z_total = R0 + Z_branch1 + Z_branch2
        
        return np.concatenate((np.real(Z_total), np.imag(Z_total)))

class RNestedRCRCCPEModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'fs1', "n1", 'R2', 'fs2', "n2"]

    def impedance(self,omega, *params):

        R0, R1, fs1, n1, R2, fs2, n2 = params
        """
        Impedance of R-(CPE//(R-(CPE//R))) circuit.
        
        Parameters:
            omega: Array of angular frequencies.
            R0: Resistance in series.
            R1: Resistance in nested series.
            R2: Resistance in nested parallel.
            fs1, n1: Parameters for the first CPE.
            fs2, n2: Parameters for the second CPE.
        
        Returns:
            Array of complex impedances.
        """
        # CPE impedances
        Z_CPE1 = 1 / ((1j * omega / fs1) ** n1)
        Z_CPE2 = 1 / ((1j * omega / fs2) ** n2)
        
        # Nested parallel combination: (CPE2 // R2)
        Z_parallel_nested = 1 / (1 / Z_CPE2 + 1 / R2)
        
        # Nested circuit: R1 + (CPE2 // R2)
        Z_nested = R1 + Z_parallel_nested
        
        # Outer parallel combination: CPE1 // Z_nested
        Z_parallel_outer = 1 / (1 / Z_CPE1 + 1 / Z_nested)
        
        # Total impedance: R0 + (CPE1 // Z_nested)
        Z_total = R0 + Z_parallel_outer
        
        return np.concatenate((np.real(Z_total), np.imag(Z_total)))


class RRCRCPoreWarburgModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'fs1', "n1", 'R2', 'fs2', "n2", "Rpore", "W"]

    def impedance(self,omega, *params):
        """
        R-RC-RC|Rpore-W model with CPE equation.
        """
        R0, R1, fs1, n1, R2, fs2, n2, Rpore, W = params


        Z_R1CPE1 = R1 / (1 + (1j * omega / fs1)**n1)

        # Second R-CPE element
        Z_R2CPE2 = R2 / (1 + (1j * omega / fs2)**n2)

        # Combine second R-CPE element with pore resistance in parallel
        Z_R2CPE2_Rpore = (Z_R2CPE2 * Rpore) / (Z_R2CPE2 + Rpore)

        # Warburg impedance for diffusion effects
        Z_Warburg = W / np.sqrt(1j * omega)

        Z = R0 + Z_R1CPE1 + Z_R2CPE2_Rpore + Z_Warburg

        return np.concatenate((np.real(Z), np.imag(Z)))
    

class RRCPEWarburgModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        self.param_names = ['Rs', 'Rp', 'fs1', "n1", "AW", "BW"]

    def impedance(self,omega, *params):
        """
        Impedance model: Randles Circuit with Warburg and Two CPEs
        Z = Rs + (CPE1 || (Rp + Warburg)  )

        Rs (float): Solution resistance
        Rp (float): Faradaic resistance
        fs1 (float): Characteristic frequency of CPE1 (double layer capacitance)
        n1 (float): Exponent of CPE1 (0 < n1 < 1) (double layer capacitance)
        Aw (float): Warburg coefficient.
        Bw (float): Warburg finite length parameter.

        """
        Rs, Rp, fs1, n1, AW, BW = params

        Z_Rp = Rp  # Charge transfer resistance
        Z_CPE = 1 / (1j * omega / fs1)**n1  # CPE
        Z_W = AW / (np.sqrt(1j * omega)) * np.tanh(BW * np.sqrt(1j * omega))  # Finite Warburg
        
        Z_parallel = 1 / (1 / Z_CPE + 1 / (Z_Rp + Z_W))
        Z = Rs + Z_parallel

        return np.concatenate((np.real(Z), np.imag(Z)))

class RRCPEWarburgRCPEModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        self.param_names = ['R0', 'R1', 'fs1', "n1", "W",'R2', 'fs2', "n2", ]

    def impedance(self,omega, *params):
        """
        Impedance model: Randles Circuit with Warburg and Two CPEs
        Z = Rs + (CPE1 || (Rp + Warburg)  )

        R0 (float): Solution resistance
        R1 (float): Faradaic resistance
        fs1 (float): Characteristic frequency of CPE1 (double layer capacitance)
        n1 (float): Exponent of CPE1 (0 < n1 < 1) (double layer capacitance)
        W (float): Warburg coefficient
        R2 (float): Faradaic resistance
        fs2 (float): Characteristic frequency of CPE1 (double layer capacitance)
        n2 (float): Exponent of CPE1 (0 < n1 < 1) (double layer capacitance)

        """
        R0, R1, fs1, n1, W, R2, fs2, n2, = params

        Z_R1 = R1  # Charge transfer resistance
        Z_CPE = 1 / (1j * omega / fs1)**n1  # CPE
        Z_W = W / (np.sqrt(1j * omega))  # Warburg impedance

        # Modification: R1 and Warburg in series first, then parallel with CPE
        Z_Warburg_series = Z_R1 + Z_W
        Z_Warburg_parallel = 1 / (1 / Z_CPE + 1 / Z_Warburg_series)

        Z_R2 = R2  # Another charge transfer resistance
        Z_CPE2 = 1 / (1j * omega / fs2)**n2  # Another CPE

        Z_RCPE = 1 / (1 / Z_CPE2 + 1 / (Z_R2))  # R2 || CPE2

        Z = R0 + Z_Warburg_parallel + Z_RCPE  # Total impedance

        return np.concatenate((np.real(Z), np.imag(Z)))
    
class RRCWarburgRCPEModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        self.param_names = ['R0', 'R1', "C1", "W",'R2', 'fs2', "n2", ]

    def impedance(self,omega, *params):
        """
        Impedance model: Randles Circuit with Warburg and Two CPEs
        Z = Rs + (CPE1 || (Rp + Warburg)  )

        R0 (float): Solution resistance
        R1 (float): Faradaic resistance
        C1 (float): Characteristic frequency of CPE1 (double layer capacitance)
        W (float): Warburg coefficient
        R2 (float): Faradaic resistance
        fs2 (float): Characteristic frequency of CPE1 (double layer capacitance)
        n2 (float): Exponent of CPE1 (0 < n1 < 1) (double layer capacitance)

        """
        R0, R1, C1, W, R2, fs2, n2, = params

        Z_R1 = R1  # Charge transfer resistance
        Z_C1 = 1 / (1j * omega * C1)
        Z_W = W / (np.sqrt(1j * omega))  # Warburg impedance

        # Modification: R1 and Warburg in series first, then parallel with CPE
        Z_Warburg_series = Z_R1 + Z_W
        Z_Warburg_parallel = 1 / (1 / Z_C1 + 1 / Z_Warburg_series)

        Z_R2 = R2  # Another charge transfer resistance
        Z_CPE2 = 1 / (1j * omega / fs2)**n2  # Another CPE

        Z_RCPE = 1 / (1 / Z_CPE2 + 1 / (Z_R2))  # R2 || CPE2

        Z = R0 + Z_Warburg_parallel + Z_RCPE  # Total impedance

        return np.concatenate((np.real(Z), np.imag(Z)))

class RRCPECPEWarburgModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        self.param_names = ['Rs', 'Rct', 'fs1', "n1", 'fs2', "n2", "W"]

    def impedance(self,omega, *params):
        """
        Impedance model: Randles Circuit with Warburg and Two CPEs
        Z = Rs + (Rct || CPE1 || Warburg) + CPE2

        Rs (float): Solution resistance
        Rct (float): Charge transfer resistance
        fs1 (float): Characteristic frequency of CPE1
        n1 (float): Exponent of CPE1 (0 < n1 < 1)
        fs2 (float): Characteristic frequency of CPE2
        n2 (float): Exponent of CPE2 (0 < n2 < 1)
        W (float): Warburg coefficient for diffusion

        """
        Rs, Rct, fs1, n1, fs2, n2, W = params
        # First CPE (non-ideal capacitance)
        Z_CPE1 = 1 / (1j * omega / fs1)**n1

        # Second CPE (non-ideal capacitance)
        Z_CPE2 = 1 / (1j * omega / fs2)**n2

        # Warburg Impedance for diffusion
        Z_Warburg = W / np.sqrt(1j * omega)

        # Parallel combination: Rct || CPE1 || Warburg
        Z_parallel = 1 / (1 / Rct + 1 / Z_CPE1 + 1 / Z_Warburg)

        # Total impedance
        Z = Rs + Z_parallel + Z_CPE2

        return np.concatenate((np.real(Z), np.imag(Z)))


class RCRCModel(BaseModel):
    def impedance(self, omega, R0, R1, C1, R2):
        jomega = 1j * omega
        Z_C1 = 1 / (jomega * C1)
        Z_R1C1 = R1 + Z_C1
        Z_R2 = R2
        Z_total = R0 + 1 / (1 / Z_R1C1 + 1 / Z_R2)
        return np.concatenate((np.real(Z_total), np.imag(Z_total)))

import numpy as np
from scipy.optimize import curve_fit

# BaseModel, FitManager, DataHandler remain the same

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
        """
        Calculate the Adjusted R-squared value.

        :param actual: The actual data points (observations).
        :param predicted: The predicted data points from the model.
        :param n_params: The number of fitted parameters in the model.
        :return: Adjusted R-squared value.
        """
        n = len(actual)  # Number of data points
        residuals = FitQuality.compute_residuals(actual, predicted)
        ss_res = np.sum(residuals ** 2)  # Residual sum of squares
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
        
        r2 = 1 - (ss_res / ss_tot)  # Standard R-squared
        
        # Adjusted R-squared
        r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - 1)
        
        return r2_adj
    
    @staticmethod
    def evaluate_fit(actual, predicted):
        """Return a dictionary of fit quality metrics."""
        return {
            "MSE": FitQuality.mean_squared_error(actual, predicted),
            "RMSE": FitQuality.root_mean_squared_error(actual, predicted),
            "R-squared": FitQuality.adjusted_r_squared(actual, predicted),
        }
    

    @staticmethod
    def check_boundaries_hit(params, bounds, tolerance=0.01):
        """
        Check if any parameters are near the boundaries.
        - `params`: The fitted parameters.
        - `bounds`: A tuple of (lower_bounds, upper_bounds).
        - `tolerance`: Percentage (e.g., 0.01 = 1%) within which a parameter is considered close to the boundary.
        Returns a list of booleans indicating whether each parameter is near the boundary.
        """
        lower_bounds, upper_bounds = bounds
        hit_status = []
        for param, lower, upper in zip(params, lower_bounds, upper_bounds):
            if abs(param - lower) / (upper - lower) < tolerance:
                hit_status.append(True)
            elif abs(param - upper) / (upper - lower) < tolerance:
                hit_status.append(True)
            else:
                hit_status.append(False)
        return hit_status


# FitManager class to handle the fitting process
class FitManager:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.previous_fitted_params = None  # Store the fitted parameters from the previous dataset

        self.Ewe = self.data_handler.Ewe
        self.suffix = self.data_handler.suffix
        self.root_folder=self.data_handler.root_folder

    def fit_model(self, model, fmin=None, fmax=None, initial_guess=None, bounds=None):
        """Fit the model to a data set."""
        filtered_df = self.data_handler.filter_frequencies(fmin, fmax)
        omega, Z_data = self.data_handler.prepare_data(filtered_df)

        def model_wrapper(omega, *params):
            return model.impedance(omega, *params)

        if initial_guess is None:
            if self.previous_fitted_params is not None:
                initial_guess = self.previous_fitted_params  # Use previous fit parameters
            else:
                initial_guess = model.get_initial_guess()

        if bounds is None:
            bounds = model.get_bounds()

        popt, pcov = curve_fit(model_wrapper, 
                               omega, 
                               Z_data,
                               p0=initial_guess, 
                               bounds=bounds, 
                               maxfev=1e6)
        model.params = popt
        self.previous_fitted_params = popt  # Store for next iteration

        # Calculate fit quality using R-squared
        fitted_Z_data = model_wrapper(omega, *popt)
        fit_quality = FitQuality.adjusted_r_squared(Z_data, fitted_Z_data)
        
        # Call the function to write the results to the file
        write_fit_results_to_file(
            model=model,  # Use the model name dynamically
            root_folder=self.root_folder,
            filename = self.suffix,
            Ewe=self.Ewe,
            fmin=fmin,
            fmax=fmax,
            initial_guess=initial_guess,
            params=model.params,
            bounds=bounds,
            residual=fit_quality
        )
        #hit_boundaries = FitQuality.check_boundaries_hit(popt, bounds)
        #if any(hit_boundaries):
        #    print("Warning: Some parameters are near the bounds!")
        #    print("Parameters hitting bounds:", np.array(popt)[hit_boundaries])

        return model, pcov, fit_quality

    def fit_model_parser(self, circuit, fmin=None, fmax=None, initial_guess=None, bounds=None):
        """Fit the model to a data set."""
        filtered_df = self.data_handler.filter_frequencies(fmin, fmax)
        omega, Z_data = self.data_handler.prepare_data(filtered_df)

        self.initial_guess = initial_guess
        initial_guess_values = []
        for key, value in self.initial_guess.items():
            if isinstance(value, tuple):  # If parameter is a tuple (like CPE), unpack it
                initial_guess_values.extend(value)
            else:
                initial_guess_values.append(value)
    
        def model_wrapper(omega, *param_values):
            #param_names = list(self.initial_guess.keys())  # Ensure parameter names are in order
            #param_dict = dict(zip(param_names, param_values))  # Convert to dictionary

            param_dict = {}
            i = 0
            for key in initial_guess.keys():
                if isinstance(self.initial_guess[key], tuple):  # Handle CPE elements properly
                    param_dict[key] = (param_values[i], param_values[i + 1])
                    i += 2
                else:
                    param_dict[key] = param_values[i]
                    i += 1
            return circuit.impedance(omega, **param_dict)
        
        popt, pcov = curve_fit(model_wrapper, 
                               omega, 
                               Z_data,
                               p0=initial_guess_values, 
                               bounds=bounds, 
                               maxfev=1e6)
        return popt
    
    def fit_multiple_files(self, model, data_handlers, fmin, fmax):
        """Fit multiple .mpr files, using the previous file's fit parameters for the next one."""
        for i, data_handler in enumerate(data_handlers):
            self.data_handler = data_handler
            print(f"Fitting file {i+1}/{len(data_handlers)}...")
            model2, pcov, fit_quality = self.fit_model(model, fmin, fmax)
            print(f"Fit quality (R²) for file {i+1}: {fit_quality['R_squared']}")
        return model2
    
    def reset_previous_parameters(self):
        """Reset the stored fitted parameters (useful if you want to start fresh)."""
        self.previous_fitted_params = None
        #print("Previous fitted parameters have been reset.")

# DataHandler class to manage data import, filtering, and transformation
class DataHandler:
    def __init__(self, dataframe,root_folder, suffix): # filepath
        # Load data from file
        #self.filepath = filepath
        #self.root_folder = self.extract_root_folder()
        self.root_folder = root_folder
        self.suffix = suffix
        #mpr_file = BioLogic.MPRfile(self.filepath)
        #self.df = pd.DataFrame(mpr_file.data)
        self.df = dataframe
        self.Ewe = self.df["<Ewe>/V"].mean()

    #def extract_root_folder(self):
    #    # Get the directory of the file
    #    directory = os.path.dirname(self.filepath)
    #    # Extract the first subfolder from the full path
    #    root_folder = os.path.split(directory)[0]  # This extracts the top-level directory
    #    return root_folder

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
                #Z_dict[key] = lambda omega, CPE_value=CPE_value, alpha=alpha: (print(f"Computing CPE impedance for omega={omega}, CPE_value={CPE_value}, alpha={alpha}"),1 / (1j * omega * CPE_value)**alpha)[-1]
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


