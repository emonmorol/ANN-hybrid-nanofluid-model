# ANN for Hybrid Nanofluid Flow 🌊

A deep learning approach to modeling hybrid nanofluid flow over a stretching sheet. This project uses a 9-layer neural network trained with advanced optimization algorithms to predict fluid dynamics and heat transfer with scientific accuracy.

---

## 🚀 Quick Start

### 1. Install Dependencies
First, make sure you have all the necessary tools installed:

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Workflow
The easiest way to get started is to run everything at once:

```bash
python src/main.py all
```

This command will:
- Clean up any previous runs
- Generate training data from physics equations (~32,400 samples)
- Train the neural network with real-time visualization
- Save the trained model and scalers

*Grab a coffee! ☕ This takes about 8-22 minutes depending on your hardware.*

### 3. Visualize the Results
After training, create publication-quality plots:

```bash
python src/visualizer.py
```

This generates detailed prediction plots comparing the model against test cases. Check `outputs/plots/model_predictions.png`!

### 4. Validate the Model
Run rigorous validation against numerical solutions:

```bash
python src/validate_model.py
```

This computes comprehensive error metrics (MSE, RMSE, MAE, R²) and creates validation plots.

---

## 📋 Available Commands

The `main.py` script is your control center. Here are all the available commands:

```bash
# Generate training and test datasets from physics equations
python src/main.py generate

# Train the neural network (requires data to be generated first)
python src/main.py train

# Regenerate training history plots from a saved checkpoint
# Useful if you want to recreate plots with different styling
python src/main.py regenerate

# Clean up all generated files (data, models, scalers, plots)
# Gives you a fresh start
python src/main.py clean

# Run the complete pipeline (clean → generate → train)
# The "do everything" button
python src/main.py all
```

### Standalone Scripts

You can also run these scripts independently:

```bash
# Create prediction visualizations (requires trained model)
python src/visualizer.py

# Validate model accuracy (requires trained model and test data)
python src/validate_model.py
```


## 📂 Project Structure

Here's a complete overview of how the project is organized:

```
ANN-net/
├── src/                          # Source code directory
│   ├── models/                   # Neural network models
│   │   ├── __init__.py          # Model package initialization
│   │   ├── ann.py               # ANN architecture definition
│   │   ├── lm_optimizer.py      # Levenberg-Marquardt optimizer
│   │   └── checkpoints/         # Model checkpoints (generated)
│   ├── solver/                   # Numerical solvers
│   │   ├── __init__.py          # Solver package initialization
│   │   └── ode_solver.py        # BVP solver for validation
│   ├── config.py                # Configuration & hyperparameters
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── generate_data.py         # Dataset generation from physics
│   ├── main.py                  # Main pipeline orchestrator
│   ├── trainer.py               # Training loop & optimization
│   ├── validate_model.py        # Model validation & error analysis
│   └── visualizer.py            # Plotting & visualization
├── data/                         # Generated datasets (created on first run)
│   ├── training_data.csv        # Training dataset
│   └── test_data.csv            # Test dataset
├── outputs/                      # Generated outputs
│   ├── models/                  # Trained models & scalers
│   └── plots/                   # Generated visualizations
├── docs/                         # Documentation
│   ├── README.md                # Detailed project documentation
│   ├── USAGE.md                 # Usage guide & examples
│   ├── ARCHITECTURE.md          # Technical architecture details
│   └── MANUSCRIPT.docx          # Research manuscript
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Key Components

*   **`src/config.py`**: Central configuration hub for physics parameters, model architecture, and training settings
*   **`src/main.py`**: Command-line interface for the entire pipeline (generate, train, clean, regenerate)
*   **`src/models/ann.py`**: Custom ANN architecture with Xavier initialization
*   **`src/trainer.py`**: Training loop with support for Adam and Levenberg-Marquardt optimizers
*   **`src/generate_data.py`**: Physics-based dataset generator using numerical ODE solutions
*   **`src/data_loader.py`**: Data preprocessing, normalization, and train/val/test splitting
*   **`src/visualizer.py`**: Publication-quality plotting with Seaborn styling
*   **`src/validate_model.py`**: Comprehensive validation against numerical solutions
*   **`src/solver/ode_solver.py`**: Boundary value problem solver for ground truth generation


Happy coding! 🚀
