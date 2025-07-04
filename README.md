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


## Quick Start

See [`examples/quick_start.py`](examples/quick_start.py)

Or try in Python:

```python
from circuit_parser import ImpedanceModel
model = ImpedanceModel("R0-CPE1")
model.params.CPE1.alpha = 0.9
