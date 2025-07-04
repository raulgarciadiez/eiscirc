from .impedance_parameters_default import PARAMETER_CONFIG, initialize_parameters, initialize_bounds, extract_base_type

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
    DEFAULT_FREQS = np.logspace(5, -2, 50)
    def __init__(self, circuit_structure):
        if isinstance(circuit_structure, str):
            self.circuit_string = circuit_structure
            self.circuit_structure = parse_circuit(circuit_structure)
        else:
            self.circuit_string = self._structure_to_string(circuit_structure)
            self.circuit_structure = circuit_structure

        self._compile_impedance_function()
        self.Z_real = None
        self.Z_imag = None

        self.param_names = self._get_parameter_names()
        self._params = initialize_parameters(self.param_names)
        self._local_bounds = initialize_bounds(self.param_names)

        self._default_omega = 2 * np.pi * self.DEFAULT_FREQS
        self._last_omega = self._default_omega
        self.impedance(self._default_omega) 
  
    
    ####### if user wants to change default
    def set_omega(self, frequencies: np.ndarray):
        self._default_omega = 2 * np.pi * frequencies
        self._last_omega = self._default_omega
        return self.impedance(self._default_omega)

    def _get_parameter_names(self):
        pattern = r'\b([RLW]\w+|C(?!PE)\w+|CPE\w+|Ws\w+|Wo\w+|G\w+|H\w+)\b'
        components = re.findall(pattern, self.circuit_string)
        seen = set()
        return [x for x in components if not (x in seen or seen.add(x))]

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            # Normalize keys like "CPE1.alpha" and "CPE1_alpha"
            if '.' in key:
                param, subkey = key.split('.', 1)
            elif '_' in key and key not in self._params:
                param, subkey = key.split('_', 1)
            else:
                param, subkey = key, None

            if param in self._params:
                current_val = self._params[param]
                if isinstance(current_val, dict):
                    if subkey:
                        self._check_bounds(f"{param}_{subkey}", value)
                        self._params[param][subkey] = value
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            self._check_bounds(f"{param}_{k}", v)
                            self._params[param][k] = v
                    elif isinstance(value, (tuple, list)):
                        base_type = extract_base_type(param)
                        keys = list(PARAMETER_CONFIG[base_type].keys())
                        if len(value) != len(keys):
                            raise ValueError(f"{param} expects {len(keys)} values")
                        for k, v in zip(keys, value):
                            self._check_bounds(f"{param}_{k}", v)
                            self._params[param][k] = v
                    else:
                        raise ValueError(f"Invalid input for structured param '{param}'")
                else:
                    self._check_bounds(param, value)
                    self._params[param] = value
            else:
                raise ValueError(f"Invalid parameter: {param}")
        return self


    def set_bounds(self, **bounds):
        for param, bound in bounds.items():
            if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                raise ValueError(f"Bounds for {param} must be a (min, max) tuple")
            self._local_bounds[param] = (float(bound[0]), float(bound[1]))
        return self

    def get_bounds(self, param_name):
        if param_name in self._local_bounds:
            return self._local_bounds[param_name]

        base_type = extract_base_type(param_name.split('_')[0])
        subkey = param_name.split('_')[1] if '_' in param_name else None

        config = PARAMETER_CONFIG.get(base_type, {})
        if subkey:
            return config.get(subkey, {}).get('bounds')
        return config.get('bounds')


    @property
    def bounds(self):
        return self.BoundsAccessor(self)
    
    class BoundsAccessor:
        def __init__(self, model):
            self._model = model

        def __getitem__(self, key):
            return self._model.get_bounds(key)

        def __getattr__(self, name):
            if '_' in name:
                return self._model.get_bounds(name)

            # Try base param
            if name in self._model._params:
                param_val = self._model._params[name]
                if isinstance(param_val, dict):
                    return self._create_proxy(name)
                return self._model.get_bounds(name)

            raise AttributeError(f"No such parameter or bounds: {name}")

        def _create_proxy(self, param_name):
            model = self._model
            base_type = extract_base_type(param_name)
            subkeys = list(PARAMETER_CONFIG[base_type].keys())

            class BoundProxy:
                def __getattr__(self_inner, attr):
                    full_key = f"{param_name}_{attr}"
                    if full_key in model._local_bounds or attr in subkeys:
                        return model.get_bounds(full_key)
                    raise AttributeError(f"No bound for '{full_key}'")

                def __repr__(self_inner):
                    return str({k: model.get_bounds(f"{param_name}_{k}") for k in subkeys})
                
                def __setattr__(self_inner, attr, value):
                    full_key = f"{param_name}_{attr}"
                    if not isinstance(value, (tuple, list)) or len(value) != 2:
                        raise ValueError("Bounds must be a (min, max) tuple")
                    model._local_bounds[full_key] = tuple(float(v) for v in value)

            return BoundProxy()

        def __repr__(self):
            return str(dict(self.items()))

        def items(self):
            for name in self._model.param_names:
                val = self._model._params[name]
                if isinstance(val, dict):
                    for subkey in val:
                        yield f"{name}_{subkey}", self._model.get_bounds(f"{name}_{subkey}")
                else:
                    yield name, self._model.get_bounds(name)
        def __setitem__(self, key, value):
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("Bounds must be a (min, max) tuple")
            self._model._local_bounds[key] = tuple(float(v) for v in value)


    @property
    def params(self):
        """Attribute-style parameter access (params.R0 = 10)"""
        return self.ParamAccessor(self, mode='set')


    class ParamAccessor:
        def __init__(self, parent, mode='set'):
            self._parent = parent
            self._mode = mode

        def __getattr__(self, name):
            if '.' in name:
                name = name.replace('.', '_')

            if '_' in name:
                param, subkey = name.split('_', 1)
                if param in self._parent._params:
                    val = self._parent._params[param]
                    if isinstance(val, dict) and subkey in val:
                        return val[subkey]
                raise AttributeError(f"Invalid sub-parameter '{name}'")

            if name in self._parent._params:
                val = self._parent._params[name]
                return self._create_param_proxy(name, val) if isinstance(val, dict) else val

            raise AttributeError(f"Unknown parameter '{name}'")

        def __setattr__(self, name, value):
            if name in ['_parent', '_mode']:
                return super().__setattr__(name, value)

            if '.' in name:
                name = name.replace('.', '_')

            if '_' in name:
                param, subkey = name.split('_', 1)
                if param in self._parent._params:
                    val = self._parent._params[param]
                    if isinstance(val, dict):
                        self._parent._check_bounds(f"{param}_{subkey}", value)
                        val[subkey] = value
                        return
                raise AttributeError(f"Invalid sub-parameter assignment: '{name}'")

            if name in self._parent._params:
                val = self._parent._params[name]
                if isinstance(val, dict):
                    self._parent.set_params(**{name: value})
                else:
                    self._parent._check_bounds(name, value)
                    self._parent._params[name] = value
            else:
                raise AttributeError(f"Unknown parameter '{name}'")
            
            if self._parent._last_omega is not None:
                try:
                    self._parent.impedance(self._parent._last_omega)
                except Exception as e:
                    print(f"Warning: impedance update failed after param change: {e}")


        def _create_param_proxy(self, param_name, param_dict):
            parent = self._parent
            expected_order = list(PARAMETER_CONFIG[extract_base_type(param_name)].keys())

            class ParamProxy:
                def __getattr__(self_inner, attr):
                    if attr in param_dict:
                        return param_dict[attr]
                    raise AttributeError(f"No such sub-parameter '{attr}'")

                def __setattr__(self_inner, attr, value):
                    if attr in param_dict:
                        full_name = f"{param_name}_{attr}"
                        parent._check_bounds(full_name, value)
                        param_dict[attr] = value

                        # ✅ Trigger impedance update if omega is known
                        if parent._last_omega is not None:
                            try:
                                parent.impedance(parent._last_omega)
                            except Exception as e:
                                print(f"Warning: impedance update failed after param change: {e}")
                    else:
                        raise AttributeError(f"No such sub-parameter '{attr}'")


                def __iter__(self_inner):
                    return iter(self_inner.__tuple__())

                def __getitem__(self_inner, index):
                    return self_inner.__tuple__()[index]

                def __tuple__(self_inner):
                    return tuple(param_dict[k] for k in expected_order if k in param_dict)

                def __repr__(self_inner):
                    return repr(self_inner.__tuple__())

                def __str__(self_inner):
                    return str(self_inner.__tuple__())

            return ParamProxy()

        # ✅ Dict-like access
        def __getitem__(self, key):
            if '_' in key:
                param, subkey = key.split('_', 1)
                return self._parent._params[param][subkey]
            return self._parent._params[key]

        def __iter__(self):
            return iter(self.items())

        def items(self):
            """Flat key-value pairs like 'CPE1_alpha': 0.8"""
            for name, val in self._parent._params.items():
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        yield f"{name}_{subkey}", subval
                else:
                    yield name, val

        def as_dict(self):
            return dict(self.items())



    def _check_bounds(self, name, value):
        # Use local bounds first
        if name in self._local_bounds:
            bounds = self._local_bounds[name]
        else:
            base_type = extract_base_type(name.split('_')[0])
            subkey = name.split('_')[1] if '_' in name else None

            if subkey:
                bounds = PARAMETER_CONFIG.get(base_type, {}).get(subkey, {}).get('bounds')
            else:
                bounds = PARAMETER_CONFIG.get(base_type, {}).get('bounds')

        if bounds and not (bounds[0] <= value <= bounds[1]):
            raise ValueError(f"{name} must be in [{bounds[0]}, {bounds[1]}], got {value}")


    def get_all_params(self):
        result = {}
        for name, val in self._params.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    result[f"{name}_{subkey}"] = subval
            else:
                result[name] = val
        return result

    def impedance(self, omega, *args, **kwargs):
        if omega is None:
                omega = self._last_omega or self._default_omega

        self._last_omega = omega
        if args or kwargs:
            if len(args) == 1:
                arg = args[0]
                if isinstance(arg, (list, tuple, np.ndarray)):
                    if len(arg) != len(self.param_names):
                        raise ValueError(f"Expected {len(self.param_names)} parameters")
                    self.set_params(**dict(zip(self.param_names, arg)))
                elif isinstance(arg, dict):
                    self.set_params(**arg)
            else:
                self.set_params(**kwargs)

        calc_params = {}
        for name, value in self._params.items():
            if isinstance(value, dict):
                if name.startswith('CPE'):
                    calc_params[name] = (value['value'], value['alpha'])
                elif name.startswith(('Ws', 'Wo', 'G')):
                    calc_params[name] = (value['R'], value['tau'])
                elif name.startswith('H'):
                    calc_params[name] = (value['R'], value['tau'], value['alpha'])
            else:
                calc_params[name] = value

        Z_total = self._impedance_func(omega, **calc_params)
        self.Z_real = np.real(Z_total)
        self.Z_imag = np.imag(Z_total)
        return np.concatenate([self.Z_real, self.Z_imag])

    def _compile_impedance_function(self):
        self._impedance_func = self._create_compiled_impedance(self.circuit_structure)

    def _create_compiled_impedance(self, structure):
        if isinstance(structure, str):
            if structure.startswith("R"):
                return lambda omega, **params: params[structure]
            elif structure.startswith("C") and not structure.startswith("CPE"):
                return lambda omega, **params: 1 / (1j * omega * params[structure])
            elif structure.startswith("CPE"):
                return lambda omega, **params: 1 / (1j * omega * params[structure][0])**params[structure][1]
            elif structure.startswith("L"):
                return lambda omega, **params: 1j * omega * params[structure]
            elif structure.startswith("Ws"):
                return lambda omega, **params: params[structure][0] * np.tanh(params[structure][1] * np.sqrt(1j * omega)) / np.sqrt(1j * omega)
            elif structure.startswith("Wo"):
                return lambda omega, **params: params[structure][0] * np.coth(params[structure][1] * np.sqrt(1j * omega)) / np.sqrt(1j * omega)
            elif structure.startswith("W") and not structure.startswith("Ws") and not structure.startswith("Wo"):
                return lambda omega, **params: params[structure] / np.sqrt(1j * omega)
            elif structure.startswith("G"):
                return lambda omega, **params: params[structure][0] / np.sqrt(1 + 1j * omega * params[structure][1])
            elif structure.startswith("H"):
                return lambda omega, **params: params[structure][0] / (1 + (1j * omega * params[structure][1])**params[structure][2])
            else:
                raise ValueError(f"Unknown component type: {structure}")
        elif isinstance(structure, tuple):
            operator = structure[0]
            sub_functions = [self._create_compiled_impedance(sub) for sub in structure[1:]]

            if operator == "series":
                return lambda omega, **params: sum(f(omega, **params) for f in sub_functions)
            elif operator == "parallel":
                return lambda omega, **params: 1 / sum(1 / f(omega, **params) for f in sub_functions)
        else:
            raise ValueError("Unknown structure format")

    def _structure_to_string(self, structure):
        if isinstance(structure, str):
            return structure
        elif isinstance(structure, tuple):
            operator = structure[0]
            elements = [self._structure_to_string(s) for s in structure[1:]]
            return '-'.join(elements) if operator == 'series' else '//'.join(elements)
        return ''


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

class ModelBoundsAccessor:
    """Helper class for dictionary-style bounds access"""
    def __init__(self, model):
        self._model = model
        
    def __getitem__(self, key):
        return self._model.get_bounds(key)
        
    def __setitem__(self, key, value):
        self._model.set_bounds(**{key: value})