# ANN Model for Hybrid Nanofluid Boundary Layer Flow

A traditional Artificial Neural Network trained with Levenberg-Marquardt algorithm to predict flow and thermal characteristics of hybrid nanofluids.

## Quick Start

```bash
# Generate training data
python src/main.py generate

# Train the model
python src/main.py train

# Validate the model
python src/validate_model.py

# Or run everything
python src/main.py all
```

## Documentation

- **[README](docs/README.md)** - Complete project overview, installation, and validation guide
- **[ARCHITECTURE](docs/ARCHITECTURE.md)** - Technical details, ANN structure, optimization
- **[USAGE](docs/USAGE.md)** - Detailed usage guide, configuration, troubleshooting

## Project Structure

```
ANN-net/
├── docs/              # All documentation
├── src/               # Source code
│   ├── models/        # ANN architecture
│   ├── solver/        # ODE solver
│   └── *.py           # Core modules
├── data/              # Generated datasets
├── outputs/           # Models and plots
│   ├── models/        # Trained models
│   └── plots/         # Generated figures
└── requirements.txt   # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, NumPy, SciPy, Matplotlib, Pandas

## Features

- ✅ 9 hidden layers × 30 neurons with tanh activation
- ✅ Levenberg-Marquardt optimization
- ✅ Numerical ODE solver (scipy.integrate.solve_bvp)
- ✅ Comprehensive validation with 8+ error metrics
- ✅ Manuscript-style visualization

## License

Academic and research purposes.

---

**Version:** 2.0.0 | **Last Updated:** 2025-11-23
