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

See [`examples/quick_start.ipynb`](examples/quick_start.ipynb)

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

## Running tests

To run the test suite locally (same steps used by CI), install the project and test dependencies and run pytest. Example (PowerShell):

```powershell
# (optional) create/activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# upgrade pip and install runtime deps if you keep a requirements.txt
python -m pip install --upgrade pip
if (Test-Path requirements.txt) { python -m pip install -r requirements.txt }

# install pytest and the package in editable mode so tests import the local package
python -m pip install pytest
python -m pip install -e .

# run tests
pytest -q
```

This matches the CI step which runs `python -m pip install -e .` before executing the tests.

## Circuit grammar and supported elements

The library accepts compact circuit expressions using the following rules:

- Series connection: use `-` between elements (e.g. `R0-C1-L1`).
- Parallel connection: use `//` between elements (e.g. `R0-(CPE1//R1)`).
- Parentheses may be used to group sub-expressions (e.g. `R0-(R1-CPE1)//R2`).
- Component tokens must be letters followed by an index number, for example `R0`, `C1`, `CPE1`, `Ws1`.

Examples:

- `R0-CPE1//R1`  — series of R0 and (CPE1 in parallel with R1).
- `R0-(CPE1//R1)` — same as above but explicit grouping.

Supported component base types are read from `eiscirc/impedance_parameters_default.py` and typically include:

- `R`  — resistor
- `C`  — capacitor
- `L`  — inductor
- `CPE` — constant phase element
- `W`  — Warburg (semi-infinite)
- `Ws`, `Wo` — finite/open Warburg variants
- `G` — Gerischer
- `H` — Havránek-style element
- `TLM` — transmission-line model (if used)

If you pass an invalid expression the parser will raise a ValueError with a helpful message (examples: unbalanced parentheses, single `/` instead of `//`, or usage of the `=` operator).

For advanced usage, consult `eiscirc/impedance_parameters_default.py` to see available sub-parameters for structured elements (for example `CPE` contains `value` and `alpha`).

### Examples: composed expressions and expected parse trees

Here are some representative expressions and what the parser returns (the structured tuple format):

- Simple series: `R0-C1-L1`

	parse_circuit("R0-C1-L1") -> ("series", "R0", "C1", "L1")

- Parallel inside series (common pattern): `R0-(CPE1//R1)`

	parse_circuit("R0-(CPE1//R1)") -> ("series", "R0", ("parallel", "CPE1", "R1"))

- Double-parallel (parallel of two parallels): `R0-(R1//C1)//(R2//C2)`

	parse_circuit("R0-(R1//C1)//(R2//C2)") -> ("parallel", ("series", "R0", ("parallel", "R1", "C1")), ("parallel", "R2", "C2"))

- Nested groups: `R0-((R1-C1)//(R2-(CPE1//L1)))`

	parse_circuit("R0-((R1-C1)//(R2-(CPE1//L1)))") ->
	("series",
		 "R0",
		 ("parallel",
			 ("series", "R1", "C1"),
			 ("series", "R2", ("parallel", "CPE1", "L1"))
		 )
	)

Tip: use the interactive helper to see the parse tree in Python:

```python
from eiscirc.circuit_parser import parse_circuit

examples = [
		"R0-C1-L1",
		"R0-(CPE1//R1)",
		"R0-(R1//C1)//(R2//C2)",
		"R0-((R1-C1)//(R2-(CPE1//L1)))",
]

for expr in examples:
		print(expr, '->', parse_circuit(expr))
```

This will print the nested tuple that the model compiler expects.

## How to be sure all tests pass

Locally:

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies and the package (editable mode) so tests import the local package:

```powershell
python -m pip install --upgrade pip
if (Test-Path requirements.txt) { python -m pip install -r requirements.txt }
python -m pip install pytest
python -m pip install -e .
```

3. Run the full test suite:

```powershell
pytest -q
```

To run only the new element tests:

```powershell
pytest tests/test_elements.py -q
```

On GitHub (CI):

- The project has a GitHub Actions workflow that runs pytest on pushes and PRs. After pushing your branch or opening a PR, go to the repository's "Actions" tab or the PR page to see the workflow run and check the job logs.
- If a test fails in CI but passes locally, check the job log for environment differences (Python version, missing system packages, or differences in installed package versions). The CI matrix runs Python 3.10 and 3.11.

What to look for in test output:

- `.` or `PASSED` lines indicate passing tests.
- `F` or `FAILED` indicates failing tests; pytest shows traceback and assertion details.
- `E` or `ERROR` indicates import/runtime errors during collection; check the stack trace for the failing import or expression.

If you'd like, I can add a small GitHub status badge to the top of this README once the CI run for your branch is green.