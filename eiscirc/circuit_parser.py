import re
import numpy as np
from math import pi
from functools import lru_cache
from types import SimpleNamespace

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class ImpedanceModel:
    def __init__(self, circuit_structure):
        """
        Initialize with either a circuit structure tuple (from parse_circuit) or a string expression.
        """
        if isinstance(circuit_structure, str):
            self.circuit_string = circuit_structure
            self.circuit_structure = parse_circuit(circuit_structure)
        else:
            self.circuit_string = self._structure_to_string(circuit_structure)
            self.circuit_structure = circuit_structure

        # Store parameter names
        self.param_names = self._get_parameter_names()

        # Pre-compile the impedance calculation
        self._compile_impedance_function()
        
        self._params = {}
        for name in self.param_names:
            if name.startswith('CPE'):
                self._params[name] = {'value': None, 'alpha': None}  # CPE: value + alpha
            elif name.startswith(('Ws', 'Wo', 'G')):
                self._params[name] = {'R': None, 'tau': None}  # Ws/Wo/G: R + tau
            elif name.startswith('H'):
                self._params[name] = {'R': None, 'tau': None, 'alpha': None}  # Havránek: R + tau + alpha
            else:
                self._params[name] = None  # Standard components (R, C, L, W)

    class ParamAccessor:
        def __init__(self, parent, mode='set'):
            self._parent = parent
            self._mode = mode  # 'set' or 'get'
            
        def __getattr__(self, name):
            # Handle underscore notation (e.g., "CPE1_alpha")
            if '_' in name:
                param_name, subkey = name.split('_', 1)
                if param_name in self._parent._params and isinstance(self._parent._params[param_name], dict):
                    if subkey in self._parent._params[param_name]:
                        return self._parent._params[param_name][subkey]
                    raise AttributeError(f"Invalid sub-parameter '{subkey}' for {param_name}")
                raise AttributeError(f"Unknown parameter: {param_name}")
            
            # Standard parameter access
            if name in self._parent._params:
                param = self._parent._params[name]
                if isinstance(param, dict):
                    # Return a dynamic object for dot notation
                    return type('ParamProxy', (), {
                        **param,
                        '__setattr__': lambda obj, attr, val: param.update({attr: val})
                    })()
                return param
            raise AttributeError(f"No parameter '{name}'")
            
        def __setattr__(self, name, value):
            if name in ['_parent', '_mode']:
                super().__setattr__(name, value)
                return
                
            # Handle underscore notation (e.g., "CPE1_alpha=0.9")
            if '_' in name:
                param_name, subkey = name.split('_', 1)
                if param_name in self._parent._params and isinstance(self._parent._params[param_name], dict):
                    if subkey in self._parent._params[param_name]:
                        self._parent._params[param_name][subkey] = value
                        return
                    raise AttributeError(f"Invalid sub-parameter '{subkey}' for {param_name}")
                raise AttributeError(f"Unknown parameter: {param_name}")
            
            # Standard parameter setting
            if name in self._parent._params:
                param = self._parent._params[name]
                if isinstance(param, dict):
                    if isinstance(value, (tuple, list)):  # Tuple input
                        if name.startswith('CPE') and len(value) == 2:
                            param.update({'value': value[0], 'alpha': value[1]})
                        elif name.startswith(('Ws', 'Wo', 'G')) and len(value) == 2:
                            param.update({'R': value[0], 'tau': value[1]})
                        elif name.startswith('H') and len(value) == 3:
                            param.update({'R': value[0], 'tau': value[1], 'alpha': value[2]})
                    else:  # Assume object with attributes
                        param.update(value.__dict__ if hasattr(value, '__dict__') else {})
                else:
                    self._parent._params[name] = value
            else:
                raise AttributeError(f"Cannot set unknown parameter '{name}'")


    def _get_parameter_names(self):
        """Extract complete component names in original circuit order"""
        if not hasattr(self, 'circuit_string'):
            return []
        
        # This regex matches:
        # - Standard components: R, L, C, W followed by numbers/letters (R1, C2b, L10)
        # - CPE components: CPE followed by letters/numbers (CPE1, CPEox)
        pattern = r'\b([RLW]\w+|C(?!PE)\w+|CPE\w+|Ws\w+|Wo\w+|G\w+|H\w+)\b'
        components = re.findall(pattern, self.circuit_string)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_components = []
        for comp in components:
            if comp not in seen:
                seen.add(comp)
                unique_components.append(comp)
        
        return unique_components

    def set_params(self, **kwargs):
        """Handle all parameter setting styles:
        - model.set_params(Ws1=(100, 0.5))  # tuple
        - model.set_params(Ws1={'R':100, 'tau':0.5})  # dict
        - model.set_params(Ws1_R=100, Ws1_tau=0.5)  # underscore
        """
        for key, value in kwargs.items():
            # Handle underscore notation (Ws1_R=100)
            if '_' in key and not key.startswith(('CPE_', 'Ws_', 'Wo_', 'G_', 'H_')):
                param_name, subkey = key.split('_', 1)
                if param_name in self._params and isinstance(self._params[param_name], dict):
                    if subkey in self._params[param_name]:
                        self._params[param_name][subkey] = value
                        continue
                    else:
                        raise ValueError(f"Invalid sub-parameter '{subkey}' for {param_name}")

            # Handle direct parameter setting
            if key in self._params:
                if isinstance(self._params[key], dict):
                    if isinstance(value, (tuple, list)):
                        if key.startswith('CPE') and len(value) == 2:
                            self._params[key].update({'value': value[0], 'alpha': value[1]})
                        elif key.startswith(('Ws', 'Wo', 'G')) and len(value) == 2:
                            self._params[key].update({'R': value[0], 'tau': value[1]})
                        elif key.startswith('H') and len(value) == 3:
                            self._params[key].update({'R': value[0], 'tau': value[1], 'alpha': value[2]})
                    elif isinstance(value, dict):
                        self._params[key].update(value)
                    else:
                        raise ValueError(f"Invalid input format for {key}")
                else:
                    self._params[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def _get_expected_format(self, name):
        """Helper to generate error messages for multi-param elements"""
        if name.startswith('CPE'):
            return "(value, alpha)"
        elif name.startswith(('Ws', 'Wo', 'G')):
            return "(R, tau)"
        elif name.startswith('H'):
            return "(R, tau, alpha)"
        return "single value"
    
    @property
    def params(self):
        """Attribute-style parameter access (params.R0 = 10)"""
        return self.ParamAccessor(self, mode='set')
    
    @property
    def param_values(self):
        """Read-only parameter access (param_values.R0)"""
        params = {}
        for name, value in self._params.items():
            if isinstance(value, dict):  # CPE case
                params[name] = SimpleNamespace(**value)
            else:
                params[name] = value
        return SimpleNamespace(**params)

    
    def _compile_impedance_function(self):
        """Pre-compile the impedance calculation for better performance"""
        # Create a template function that we'll optimize
        self._impedance_func = self._create_compiled_impedance(self.circuit_structure)
    
    def _create_compiled_impedance(self, structure):
        """
        Recursively build a compiled impedance function for the given structure.
        Returns a function that takes (omega, **params) and returns impedance.
        """
        if isinstance(structure, str):
            # Handle individual components
            if structure.startswith("R"):
                return lambda omega, **params: params[structure]
            elif structure.startswith("C") and not structure.startswith("CPE"):
                return lambda omega, **params: 1 / (1j * omega * params[structure])
            elif structure.startswith("CPE"):  ### value and alpha (0-1)
                return lambda omega, **params: 1 / (1j * omega * params[structure][0])**params[structure][1]
            elif structure.startswith("L"):
                return lambda omega, **params: 1j * omega * params[structure]
            elif structure.startswith("Ws"):  # Finite Warburg (Resistance and Tau (time))
                return lambda omega, **params: params[structure][0] * np.tanh( params[structure][1] * np.sqrt(1j * omega) ) / np.sqrt(1j * omega)
            elif structure.startswith("Wo"):  # Shorted Warburg
                return lambda omega, **params: params[structure][0] * np.coth( params[structure][1] * np.sqrt(1j * omega) ) / np.sqrt(1j * omega)
            elif structure.startswith("W") and not structure.startswith("Ws")  and not structure.startswith("Wo"):
                return lambda omega, **params: params[structure] / np.sqrt(1j * omega)
            elif structure.startswith("G"):   # Gerischer (Resistance and Tau (time))
                return lambda omega, **params: params[structure][0] / np.sqrt(1 + 1j * omega * params[structure][1])
            elif structure.startswith("H"):   # Havránek (Resistance, Tau (time) and alpha (0-1))
                return lambda omega, **params: params[structure][0] / (1 + (1j * omega * params[structure][1])**params[structure][2])
            #elif structure.startswith("TLM"): # Transmission Line
            #    return lambda omega, **params: (params[structure] * np.coth(np.sqrt(params[f"Z1_{structure}"] * params[f"Z2_{structure}"])) / np.sqrt(params[f"Z1_{structure}"] / params[f"Z2_{structure}"]))
            else:
                raise ValueError(f"Unknown component type: {structure}")
        elif isinstance(structure, tuple):
            operator = structure[0]
            sub_functions = [self._create_compiled_impedance(sub) for sub in structure[1:]]
            
            if operator == "series":
                def series_func(omega, **params):
                    Z_total = 0
                    for func in sub_functions:
                        Z_total += func(omega, **params)
                    return Z_total
                return series_func
            elif operator == "parallel":
                def parallel_func(omega, **params):
                    Y_total = 0
                    for func in sub_functions:
                        Z = func(omega, **params)
                        Y_total += 1/Z
                    return 1/Y_total
                return parallel_func
            else:
                raise ValueError(f"Unknown operator: {operator}")
        else:
            raise ValueError("Unknown structure format")

    def impedance(self, omega, *args, **kwargs):
        """
        Unified impedance calculation that handles:
        1. Direct parameter passing (R0=10, CPE1=(1e-6,0.8))
        2. Attribute-style parameters (model.set_params.X = Y)
        3. Mixed usage
        """
        # Process input parameters if any were provided
        if args or kwargs:
            # Case 1: Array input
            if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
                if len(args[0]) != len(self.param_names):
                    raise ValueError(f"Expected {len(self.param_names)} parameters, got {len(args[0])}")
                params = dict(zip(self.param_names, args[0]))
                self.set_params(**params)
            
            # Case 2: Dictionary input
            elif len(args) == 1 and isinstance(args[0], dict):
                self.set_params(**args[0])
            
            # Case 3: Keyword arguments
            else:
                self.set_params(**kwargs)
        
        # Prepare parameters for calculation (convert CPE dicts to tuples)
        calc_params = {}
        for name, value in self._params.items():
            if isinstance(value, dict):
                # Check for missing parameters
                missing = [k for k, v in value.items() if v is None]
                if missing:
                    raise ValueError(f"Missing parameters for {name}: {missing}")
                
                # Convert to tuple in the order expected by _create_compiled_impedance
                if name.startswith('CPE'):
                    calc_params[name] = (value['value'], value['alpha'])
                elif name.startswith(('Ws', 'Wo', 'G')):
                    calc_params[name] = (value['R'], value['tau'])
                elif name.startswith('H'):
                    calc_params[name] = (value['R'], value['tau'], value['alpha'])
            else:
                if value is None:
                    raise ValueError(f"Missing parameter: {name}")
                calc_params[name] = value
        
        # Calculate impedance using stored parameters
        try:
            Z_total = self._impedance_func(omega, **calc_params)
        except KeyError as e:
            missing = set(self.param_names) - set(calc_params.keys())
            raise ValueError(f"Missing parameters: {missing}") from e
        
        # Store and return results
        self.Z_real = np.real(Z_total)
        self.Z_imag = np.imag(Z_total)
        
        if len(self.Z_real) != len(self.Z_imag):
            raise ValueError(f"Real/imaginary length mismatch: {len(self.Z_real)} vs {len(self.Z_imag)}")
        
        return np.concatenate((self.Z_real, self.Z_imag))    
    def _structure_to_string(self, structure):
        """Convert structure tuple back to string for parameter ordering"""
        if isinstance(structure, str):
            return structure
        elif isinstance(structure, tuple):
            operator = structure[0]
            elements = [self._structure_to_string(s) for s in structure[1:]]
            if operator == 'series':
                return '-'.join(elements)
            elif operator == 'parallel':
                return '//'.join(elements)
        return ''
    
######################################################
################# NOT really used ####################    
######################################################


    def __call__(self, omega, **params):
        """Alias for impedance()"""
        return self.impedance(omega, **params)

    # Keep all the display and utility methods from your original implementation
    def display_structure(self):
        """Recursively display the structure of the circuit."""
        print(self._display_structure_recursive(self.circuit_structure))

    def _display_structure_recursive(self, structure):
        """Helper function to print the structure in a readable format."""
        if isinstance(structure, str):
            return structure
        elif isinstance(structure, tuple):
            operator = structure[0]
            components = [self._display_structure_recursive(comp) for comp in structure[1:]]
            return f"({f' {operator} '.join(components)})"
        else:
            return "Unknown structure"

    def generate_filename(self):
        """Generate a filename-safe string representation of the circuit."""
        filename = self._structure_to_filename_string(self.circuit_structure)
        return filename.replace(" ", "")
    
    def _structure_to_filename_string(self, structure):
        """Recursively convert tuple structure to a filename-safe format."""
        if isinstance(structure, str):
            return structure
        elif isinstance(structure, tuple):
            operator = "_" if structure[0] == "series" else "||"
            elements = [self._structure_to_filename_string(sub) for sub in structure[1:]]
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
            if isinstance(ref_value, tuple):
                param_dict[key] = tuple(param_list[index:index+len(ref_value)])
                index += len(ref_value)
            else:
                param_dict[key] = param_list[index]
                index += 1
        return param_dict


######################################################
################# UNTIL HERE ####################    
######################################################


    def draw_circuit(self, ax=None, position=None, size=(6, 3), scale=1.0, 
                    loc='upper right', borderpad=1):
        """
        Draw the circuit diagram as matplotlib figure or inset.
        """
        # Set drawing parameters based on context
        if ax is None:
            fig, ax = plt.subplots(figsize=size)
            ax.set_aspect('equal')
            ax.axis('off')
            is_inset = False
            element_scale = 1.5 * scale
            fontsize = 10 * scale
        else:
            if position is None:
                position = [0.3, 0.3]
            ax = inset_axes(ax, width=position[0], height=position[1],
                        loc=loc, borderpad=borderpad)
            ax.set_aspect('equal')
            ax.axis('off')
            is_inset = True
            element_scale = 0.8 * scale
            fontsize = 8 * scale
        
        # Store drawing parameters and ax as instance attributes
        self.ax = ax
        self.element_width = 1.0 * element_scale
        self.element_height = 0.6 * element_scale
        self.wire_length = 0.4 * element_scale
        self.vertical_spacing = 1.2 * element_scale
        self.fontsize = fontsize

        # Start drawing
        start_x = 0.5 * element_scale
        start_y = 0
        final_x = self._draw_structure(self.circuit_structure, start_x, start_y)
        
        # Add input/output wires
        ax.add_line(Line2D([0, start_x], [start_y, start_y], 
                        color='black', linewidth=1))
        ax.add_line(Line2D([final_x - self.wire_length, final_x + 0.5*element_scale], 
                        [start_y, start_y], color='black', linewidth=1))
        
        # Set axis limits
        x_padding = 0.5 * element_scale
        y_padding = 1.0 * element_scale
        ax.set_xlim(0 - x_padding, final_x + x_padding)
        ax.set_ylim(-2*element_scale - y_padding, 2*element_scale + y_padding)
        
        return ax



    def _draw_component(self, component, x, y):
        """Draw individual component and return its right edge"""
        if component.startswith('R'):
            self._draw_resistor(ax=self.ax, x=x, y=y, 
                            width=self.element_width, 
                            height=self.element_height,
                            label=component,
                            fontsize=self.fontsize)
        elif component.startswith('CPE'):
            self._draw_cpe(ax=self.ax, x=x, y=y,
                                width=self.element_width,
                                height=self.element_height,
                                label=component,
                                fontsize=self.fontsize)
        elif component.startswith('C'):
            self._draw_capacitor(ax=self.ax, x=x, y=y,
                                width=self.element_width,
                                height=self.element_height,
                                label=component,
                                fontsize=self.fontsize)
        elif component.startswith('L'):
            self._draw_inductor(ax=self.ax, x=x, y=y,
                                width=self.element_width,
                                height=self.element_height,
                                label=component,
                                fontsize=self.fontsize)
        elif component.startswith('W'):
            self._draw_warburg(ax=self.ax, x=x, y=y,
                                width=self.element_width,
                                height=self.element_height,
                                label=component,
                                fontsize=self.fontsize)

        else:
            self._draw_generic(ax=self.ax, x=x, y=y,
                                width=self.element_width,
                                height=self.element_height,
                                label=component,
                                fontsize=self.fontsize)
        return x + self.element_width

    def _draw_parallel(self, structure, x, y, level):
        """Handle parallel circuit drawing"""
        num_branches = len(structure) - 1
        total_height = (num_branches - 1) * self.vertical_spacing
        
        # 1. Entry connection
        entry_wire_end = x + self.wire_length
        self.ax.add_line(Line2D([x, entry_wire_end], [y, y],
                            color='black', linewidth=1))
        
        # Vertical divider
        self.ax.add_line(Line2D([entry_wire_end, entry_wire_end],
                            [y-total_height/2, y+total_height/2],
                            color='black', linewidth=1))
        
        # 2. Draw branches
        branch_exits = []
        for i, substructure in enumerate(structure[1:]):
            branch_y = y + (i - (num_branches-1)/2) * self.vertical_spacing
            
            # Horizontal to branch
            branch_start = entry_wire_end + self.wire_length
            self.ax.add_line(Line2D([entry_wire_end, branch_start],
                            [branch_y, branch_y],
                            color='black', linewidth=1))
            
            # Draw branch component
            component_end = self._draw_structure(substructure, branch_start, branch_y)
            
            # Horizontal to merge point
            merge_point = component_end + self.wire_length
            self.ax.add_line(Line2D([component_end, merge_point],
                                [branch_y, branch_y],
                                color='black', linewidth=1))
            branch_exits.append(merge_point)
        
        # 3. Exit connection
        exit_wire_start = max(branch_exits)
        self.ax.add_line(Line2D([exit_wire_start, exit_wire_start],
                            [y-total_height/2, y+total_height/2],
                            color='black', linewidth=1))
        self.ax.add_line(Line2D([exit_wire_start, exit_wire_start + self.wire_length],
                            [y, y],
                            color='black', linewidth=1))
        
        return exit_wire_start + self.wire_length

    def _draw_structure(self, structure, x, y, level=0):
        """Recursively draw circuit components and connections"""
        if isinstance(structure, str):
            return self._draw_component(structure, x, y)
        
        elif isinstance(structure, tuple):
            if structure[0] == 'series':
                current_x = x
                for substructure in structure[1:]:
                    current_x = self._draw_structure(substructure, current_x, y, level)
                return current_x
            
            elif structure[0] == 'parallel':
                return self._draw_parallel(structure, x, y, level)

    @staticmethod
    def _draw_resistor(ax, x, y, width, height, label, fontsize=10):
        """Draw resistor with proper angled finish"""
        peaks = 4  # Number of complete peaks
        segment_length = width / (peaks * 2 + 1)  # +1 for the final segment
        peak_height = height * 0.2
        
        # Calculate centered starting position
        total_pattern_width = (peaks * 2 + 1) * segment_length
        start_x = x + (width - total_pattern_width)/2
        
        # Start wire
        ax.add_line(Line2D([x, start_x], [y, y], color='black', linewidth=1))
        
        # Create points - start at centerline
        points = [(start_x, y)]
        direction = 1  # Start going up
        
        for i in range(peaks * 2):
            x_pos = start_x + (i + 1) * segment_length
            y_pos = y + (peak_height * direction)
            points.append((x_pos, y_pos))
            direction *= -1  # Reverse direction
        
        # Final point back to centerline (angled, not vertical)
        points.append((start_x + (peaks*2 + 1)*segment_length, y))
        
        # Draw the continuous line
        ax.add_line(Line2D([p[0] for p in points], [p[1] for p in points],
                    color='black', linewidth=1))
        
        # End wire
        ax.add_line(Line2D([points[-1][0], x + width], [y, y],
                        color='black', linewidth=1))
        
        # Label
        ax.text(x + width/2, y + height/2 + 0.1, label,
                ha='center', va='bottom', fontsize=fontsize)

    @staticmethod
    def _draw_capacitor(ax, x, y, width, height, label, fontsize=10):
        """Draw capacitor with vertical plates || and proper horizontal connections"""
        plate_sep = width * 0.2  # Space between plates
        plate_height = height * 0.6  # Height of plates
        plate_thickness = 1.5  # Line thickness
        
        # Calculate plate positions (centered vertically and horizontally)
        left_plate_x = x + (width - plate_sep)/2
        right_plate_x = x + (width + plate_sep)/2
        
        # Left vertical plate
        ax.add_line(Line2D([left_plate_x, left_plate_x],
                        [y - plate_height/2, y + plate_height/2],
                        color='black', linewidth=plate_thickness))
        
        # Right vertical plate
        ax.add_line(Line2D([right_plate_x, right_plate_x],
                        [y - plate_height/2, y + plate_height/2],
                        color='black', linewidth=plate_thickness))
        
        # Horizontal connection wires
        ax.add_line(Line2D([x, left_plate_x], [y, y], 
                        color='black', linewidth=1))
        ax.add_line(Line2D([right_plate_x, x + width], [y, y],
                        color='black', linewidth=1))
        
        # Label centered above plates
        ax.text(x + width/2, y + plate_height/2 + 0.2, label,
                ha='center', va='bottom', fontsize=fontsize)

    @staticmethod
    def _draw_cpe(ax, x, y, width, height, label, fontsize=10):
        """Draw CPE as properly centered >> symbol"""
        # Dimensions for the angles
        angle_width = width * 0.2  # Width of each angle
        angle_height = height * 0.4  # Height of each angle
        gap = width * 0.1  # Space between angles
        
        # Calculate center position
        total_width = 2*angle_width + gap
        start_x = x + (width - total_width)/2
        
        # First angle (>)
        ax.add_line(Line2D([start_x, start_x + angle_width],
                        [y - angle_height/2, y],
                        color='black', linewidth=1))
        ax.add_line(Line2D([start_x + angle_width, start_x],
                        [y, y + angle_height/2],
                        color='black', linewidth=1))
        
        # Second angle (>) - offset to the right
        second_x = start_x + angle_width + gap
        ax.add_line(Line2D([second_x, second_x + angle_width],
                        [y - angle_height/2, y],
                        color='black', linewidth=1))
        ax.add_line(Line2D([second_x + angle_width, second_x],
                        [y, y + angle_height/2],
                        color='black', linewidth=1))
        
        # Wires - connect to component edges
        ax.add_line(Line2D([x, start_x], [y, y], color='black', linewidth=1))
        ax.add_line(Line2D([second_x + angle_width, x + width], 
                        [y, y], color='black', linewidth=1))
        
        # Label above the component
        ax.text(x + width/2, y + height/2 + 0.1, label,
                ha='center', va='bottom', fontsize=fontsize)

    @staticmethod
    def _draw_warburg(ax, x, y, width, height, label, fontsize=10):
        """Draw Warburg as large W"""
        # Draw a prominent W
        w_height = height * 0.5
        w_width = width * 0.3
        
        # Points for W shape
        points = [
            (x + width/2 - w_width/2, y + w_height/2),
            (x + width/2 - w_width/4, y - w_height/2),
            (x + width/2, y),
            (x + width/2 + w_width/4, y - w_height/2),
            (x + width/2 + w_width/2, y + w_height/2)
        ]
        
        # Draw W
        ax.add_line(Line2D([p[0] for p in points], [p[1] for p in points],
                color='black', linewidth=1))
        
        # Wires
        ax.add_line(Line2D([x, x + width/2 - w_width/2], [y, y], color='black'))
        ax.add_line(Line2D([x + width/2 + w_width/2, x + width], [y, y], color='black'))
        
        # Label
        ax.text(x + width/2, y + w_height/2 + 0.2, label,
                ha='center', va='bottom', fontsize=fontsize)

    @staticmethod
    def _draw_inductor(ax, x, y, width, height, label, fontsize=10):
        """Draw inductor as ~~~~"""
        coils = 4
        coil_width = width / coils
        coil_height = height * 0.6
        
        # Start wire
        ax.add_line(Line2D([x, x + coil_width/4], [y, y], color='black'))
        
        # Draw coils as sine waves
        x_vals = np.linspace(x + coil_width/4, x + width - coil_width/4, 50)
        y_vals = y + coil_height/2 * np.sin(2*np.pi*(x_vals-x)/(coil_width))
        ax.add_line(Line2D(x_vals, y_vals, color='black'))
        
        # End wire
        ax.add_line(Line2D([x + width - coil_width/4, x + width], [y, y], color='black'))
        
        # Label
        ax.text(x + width/2, y + coil_height/2 + 0.2, label,
                ha='center', va='bottom', fontsize=fontsize)
        

    @staticmethod
    def _draw_generic(ax, x, y, width, height, label, fontsize = 10):
        """Draw a generic component symbol"""
        ax.add_patch(patches.Rectangle((x, y - height/2), width, height, 
                                       fill=False, edgecolor='black'))
        ax.text(x + width/2, y, label, ha='center', va='center', fontsize=fontsize)

def parse_circuit(expression):
    """Parse a circuit expression string into a structured tuple format."""
    expression = expression.replace(" ", "")
    
    def parse(expr):
        # Base case: single component
        if re.match(r'^[A-Za-z]+\d+$', expr):
            return expr
        
        # Handle parentheses
        if expr.startswith('(') and expr.endswith(')'):
            return parse(expr[1:-1])
        
        # Check for series connections (denoted by '-')
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
        
        return expr
    
    return parse(expression)