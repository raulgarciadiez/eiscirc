import re
import numpy as np
from math import pi
from functools import lru_cache

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


class ImpedanceModel:
    def __init__(self, circuit_structure):
        """
        Initialize with either a circuit structure tuple (from parse_circuit) or a string expression.
        """
        if isinstance(circuit_structure, str):
            self.circuit_structure = parse_circuit(circuit_structure)
        else:
            self.circuit_structure = circuit_structure
        
        # Pre-compile the impedance calculation
        self._compile_impedance_function()
    
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
            elif structure.startswith("CPE"):
                return lambda omega, **params: 1 / (1j * omega * params[structure][0])**params[structure][1]
            elif structure.startswith("W"):
                return lambda omega, **params: params[structure] / np.sqrt(1j * omega)
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

    def impedance(self, omega, **params):
        """
        Compute the impedance of the entire circuit.
        Returns concatenated real and imaginary parts for fitting.
        """
        Z_total = self._impedance_func(omega, **params)
        real_part = np.real(Z_total)
        imag_part = np.imag(Z_total)
        
        if len(real_part) != len(imag_part):
            raise ValueError(f"Real and imaginary parts have different lengths: {len(real_part)} != {len(imag_part)}")
        
        return np.concatenate((real_part, imag_part))
    
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
    
    def draw_circuit(self, figsize=(8, 4), dpi=100):
        """
        Draw the circuit diagram using matplotlib.
        
        Parameters:
        - figsize: tuple (width, height) in inches
        - dpi: resolution in dots per inch
        
        Returns:
        - matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Set drawing parameters
        element_width = 1.5
        element_height = 0.8
        wire_length = 0.5
        vertical_spacing = 1.5
        
        # Start drawing from left to right
        start_x = 1
        start_y = 0
        
        # Recursive drawing function
        def draw_structure(structure, x, y, level=0):
            if isinstance(structure, str):
                # Draw single component
                if structure.startswith('R'):
                    self._draw_resistor(ax, x, y, element_width, element_height, label=structure)
                elif structure.startswith('C'):
                    self._draw_capacitor(ax, x, y, element_width, element_height, label=structure)
                elif structure.startswith('L'):
                    self._draw_inductor(ax, x, y, element_width, element_height, label=structure)
                elif structure.startswith('W'):
                    self._draw_warburg(ax, x, y, element_width, element_height, label=structure)
                elif structure.startswith('CPE'):
                    self._draw_cpe(ax, x, y, element_width, element_height, label=structure)
                else:
                    self._draw_generic(ax, x, y, element_width, element_height, label=structure)
                return x + element_width + wire_length
            
            elif isinstance(structure, tuple):
                operator = structure[0]
                if operator == 'series':
                    current_x = x
                    for substructure in structure[1:]:
                        current_x = draw_structure(substructure, current_x, y, level)
                    return current_x
                
                elif operator == 'parallel':
                    # Calculate how many parallel branches we have
                    num_branches = len(structure) - 1
                    total_height = (num_branches - 1) * vertical_spacing
                    
                    # Draw vertical wires
                    ax.add_line(Line2D([x, x], [y - total_height/2, y + total_height/2], 
                                    color='black', linewidth=1))
                    
                    # Draw each branch
                    branch_x = []
                    for i, substructure in enumerate(structure[1:]):
                        branch_y = y + (i - (num_branches-1)/2) * vertical_spacing
                        end_x = draw_structure(substructure, x + wire_length, branch_y, level+1)
                        branch_x.append(end_x)
                        
                        # Draw vertical connection lines
                        if i > 0:
                            ax.add_line(Line2D([x, x + wire_length], [branch_y, branch_y], 
                                          color='black', linewidth=1))
                    
                    # Draw closing vertical wire
                    max_x = max(branch_x)
                    ax.add_line(Line2D([max_x, max_x], 
                                      [y - total_height/2, y + total_height/2], 
                                      color='black', linewidth=1))
                    
                    # Draw horizontal connection lines
                    for bx in branch_x:
                        ax.add_line(Line2D([bx, max_x], [y + (i - (num_branches-1)/2) * vertical_spacing, 
                                      y + (i - (num_branches-1)/2) * vertical_spacing], 
                                      color='black', linewidth=1))
                    
                    return max_x + wire_length
        
        # Start drawing from left side
        final_x = draw_structure(self.circuit_structure, start_x, start_y)
        
        # Add input/output wires
        ax.add_line(Line2D([0, start_x], [start_y, start_y], color='black', linewidth=1))
        ax.add_line(Line2D([final_x - wire_length, final_x + 1], [start_y, start_y], color='black', linewidth=1))
        
        # Set axis limits
        ax.set_xlim(0, final_x + 1)
        ax.set_ylim(-3, 3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _draw_resistor(ax, x, y, width, height, label):
        """Draw a resistor symbol"""
        # Resistor body
        ax.add_patch(patches.Rectangle((x, y - height/2), width, height, 
                                       fill=False, edgecolor='black'))
        # Label
        ax.text(x + width/2, y + height/2 + 0.1, label, 
                ha='center', va='bottom', fontsize=10)
    
    @staticmethod
    def _draw_capacitor(ax, x, y, width, height, label):
        """Draw a capacitor symbol"""
        plate_sep = height * 0.4
        plate_width = width * 0.3
        # Plates
        ax.add_line(Line2D([x + width/2 - plate_width/2, x + width/2 + plate_width/2], 
                          [y + plate_sep/2, y + plate_sep/2], color='black', linewidth=2))
        ax.add_line(Line2D([x + width/2 - plate_width/2, x + width/2 + plate_width/2], 
                          [y - plate_sep/2, y - plate_sep/2], color='black', linewidth=2))
        # Wires
        ax.add_line(Line2D([x, x + width/2 - plate_width/2], [y, y + plate_sep/2], color='black'))
        ax.add_line(Line2D([x + width/2 + plate_width/2, x + width], [y + plate_sep/2, y], color='black'))
        # Label
        ax.text(x + width/2, y + plate_sep/2 + 0.2, label, ha='center', va='bottom', fontsize=10)
    
    @staticmethod
    def _draw_inductor(ax, x, y, width, height, label):
        """Draw an inductor symbol"""
        coils = 3
        coil_width = width / (coils * 2)
        coil_height = height * 0.6
        
        # Start wire
        ax.add_line(Line2D([x, x + coil_width/2], [y, y], color='black'))
        
        # Coils
        for i in range(coils):
            cx = x + coil_width/2 + i * 2 * coil_width
            ax.add_artist(patches.Arc((cx + coil_width, y), coil_width*2, coil_height, 
                          angle=0, theta1=180, theta2=0, color='black'))
            ax.add_artist(patches.Arc((cx + coil_width*2, y), coil_width*2, coil_height, 
                          angle=0, theta1=180, theta2=0, color='black'))
        
        # End wire
        ax.add_line(Line2D([x + width - coil_width/2, x + width], [y, y], color='black'))
        # Label
        ax.text(x + width/2, y + coil_height/2 + 0.2, label, ha='center', va='bottom', fontsize=10)
    
    @staticmethod
    def _draw_warburg(ax, x, y, width, height, label):
        """Draw a Warburg element symbol"""
        # Similar to resistor but with special marking
        ax.add_patch(patches.Rectangle((x, y - height/2), width, height, 
                                       fill=False, edgecolor='black', linestyle='--'))
        ax.text(x + width/2, y + height/2 + 0.1, label, 
                ha='center', va='bottom', fontsize=10)
        # Add W symbol inside
        ax.text(x + width/2, y, 'W', ha='center', va='center', fontsize=8)
    
    @staticmethod
    def _draw_cpe(ax, x, y, width, height, label):
        """Draw a CPE symbol"""
        # Similar to capacitor but with special marking
        plate_sep = height * 0.4
        plate_width = width * 0.3
        # Plates
        ax.add_line(Line2D([x + width/2 - plate_width/2, x + width/2 + plate_width/2], 
                          [y + plate_sep/2, y + plate_sep/2], color='black', linewidth=2))
        ax.add_line(Line2D([x + width/2 - plate_width/2, x + width/2 + plate_width/2], 
                          [y - plate_sep/2, y - plate_sep/2], color='black', linewidth=2))
        # Wires
        ax.add_line(Line2D([x, x + width/2 - plate_width/2], [y, y + plate_sep/2], color='black'))
        ax.add_line(Line2D([x + width/2 + plate_width/2, x + width], [y + plate_sep/2, y], color='black'))
        # Label and Q symbol
        ax.text(x + width/2, y + plate_sep/2 + 0.2, label, ha='center', va='bottom', fontsize=10)
        ax.text(x + width/2, y, 'Q', ha='center', va='center', fontsize=8)
    
    @staticmethod
    def _draw_generic(ax, x, y, width, height, label):
        """Draw a generic component symbol"""
        ax.add_patch(patches.Rectangle((x, y - height/2), width, height, 
                                       fill=False, edgecolor='black'))
        ax.text(x + width/2, y, label, ha='center', va='center', fontsize=10)

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