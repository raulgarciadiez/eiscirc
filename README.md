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

See [`examples/quick_start.py`](examples/quick_start.ipynb)

#### Define a model

```python
from eiscirc import ImpedanceModel

# Provide an equivalent circuit by writing an string where - is for series and // for parallel circuits
model = ImpedanceModel("R0-CPE1//R1")

# Model uses the default values in PARAMETER_CONFIG dictionary in `eiscirc/impedance_parameters_default.py` if not updated
print (dict(model.params)) 

# Manual update of parameters
model.params.R0 = 50
model.params.CPE1.alpha = 0.9
print (dict(model.params))
print (model.params.R0, model.params.CPE1)
```

#### Plot Nyquist

```python
import matplotlib.pyplot as plt
import numpy as np
from eiscirc import ImpedanceModel

model = ImpedanceModel("R0-CPE1//R1")

# A default frequency array is used for the calculation of the impedance Z
plt.scatter(model.Z_real, -model.Z_imag, label = "default frequencies")

# Customize frequencies array can also be provided for the impedance calculation
frequencies = np.logspace(7, -3, 1000)
omega = 2 * np.pi * frequencies
Z = model.impedance(omega, R1 = 50) # change parameter R1 for impedance calculation
plt.scatter(model.Z_real, -model.Z_imag, label = "customized freq. + different R1")

plt.legend()
plt.xlabel("Re(Z)")
plt.ylabel("-Im(Z)")
```

#### Fitting Experimental Data

```python
import matplotlib.pyplot as plt
import numpy as np
from eiscirc import ImpedanceModel

model = ImpedanceModel("R0-CPE1//R1")
frequencies_experimental = np.logspace(5, -2, 60)
Z_simulated = model.impedance(2 * np.pi * frequencies_experimental) + 0.01 * np.random.randn(120)

plt.scatter(model.Z_real, -model.Z_imag, label='Simulated data')

# User another model to fit simulated data
model_fit = ImpedanceModel("R0-(Rextra-CPE1)//R1")
print ("Default values of model_fit", dict(model_fit.params))
fitter = ImpedanceFitter(model_fit, frequencies_experimental, Z_simulated)

fitter.set_initial_guess(CPE1_alpha=0.3)       # Optional
fitter.set_fixed_params(R0=40)        # Optional

plt.plot(model_fit.Z_real, -model_fit.Z_imag, label='Before fit')

result = fitter.fit()
print("Fitted:", result.parameters)

plt.plot(model_fit.Z_real, -model_fit.Z_imag, label='After fit')

plt.legend()
plt.xlabel("Re(Z)")
plt.ylabel("-Im(Z)")
```