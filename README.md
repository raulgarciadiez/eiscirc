# ImpedanceModel

A flexible, extensible Python framework for defining, simulating, and fitting equivalent circuits for impedance spectroscopy (EIS).

Designed for electrochemical systems, battery analysis, materials characterization, and more.

---

## Features

- **Circuit parsing** from strings or tuples
- **Flexible parameter access**: `.CPE1.alpha`, `CPE1_alpha`, or `CPE1={"alpha": ...}`
- **Live impedance updates** on param changes
- **Automatic default frequency support**
- **Fully synchronized ImpedanceFitter**
- **Ready for model comparison, confidence intervals, and batch fitting**

---

## Installation

```bash
pip install numpy scipy matplotlib
```
## Examples

### Quick Start

See [`examples/quick_start.py`](examples/quick_start.py)

#### Define a model

```python
from circuit_parser import ImpedanceModel

# Provide an equivalent circuit by writing an string where - is for series and // for parallel circuits
model = ImpedanceModel("R0-CPE1//R1")

# Model uses the default values in PARAMETER_CONFIG dictionary in `eiscirc/impedance_parameters_default.py` if not updated
print (dict(model.params)) 

# Manual update of parameters
model.params.R0 = 100
model.params.CPE1.alpha = 0.9
print (dict(model.params))
print (model.params.R0, model.params.CPE1)
```

#### Plot Nyquist

```python
import matplotlib.pyplot as plt

# A default frequency array is used for the calculation of the impedance Z
plt.scatter(model.Z_real, -model.Z_imag, label = "default frequencies")
plt.xlabel("Re(Z)")
plt.ylabel("-Im(Z)")

# Customize frequencies array can also be provided for the impedance calculation
frequencies = np.logspace(7, -3, 1000)
omega = 2 * np.pi * frequencies
Z = model.impedance(omega)
plt.scatter(model.Z_real, -model.Z_imag)

```
