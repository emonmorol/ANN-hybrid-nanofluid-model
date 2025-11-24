# How to Use This Repo 🛠️

Welcome to the comprehensive user guide! This document will walk you through everything from generating your first dataset to training, visualizing, and validating your model.

## ⚡ Quick Reference

If you just want the commands, here they are:

```bash
# Complete workflow (The "Easy Button")
python src/main.py all

# Individual pipeline steps
python src/main.py generate    # 1. Generate training data from physics
python src/main.py train       # 2. Train the neural network
python src/main.py regenerate  # 3. Recreate training plots from checkpoint
python src/main.py clean       # 4. Clean up all generated files

# Standalone analysis scripts (require trained model)
python src/visualizer.py       # Create prediction comparison plots
python src/validate_model.py   # Perform rigorous validation
```

---

## Detailed Usage

### 1. Data Generation

**Command:**
```bash
python src/main.py generate
# or
python src/generate_data.py
```

**What it does:**
- Generates parameter combinations from `config.PARAM_RANGES`
- Solves ODEs for each combination
- Creates `data/training_data.csv` (~32,400 samples)
- Creates `data/test_data.csv` (10 random cases)

**Expected output:**
```
Generating dataset with 81 parameter combinations...
Total data points: 32400
Solving ODE cases: 100%|████████████| 81/81
✓ Dataset saved to: data/training_data.csv
  Total cases: 81
  Successful: 81
  Total rows: 32400
```

**Time:** ~2-5 minutes (depends on CPU)

---

### 2. Model Training

**Command:**
```bash
python src/main.py train
```

**What it does:**
- Loads training data
- Normalizes inputs/outputs
- Initializes ANN model
- Trains using L-BFGS or LM or Adam optimizer
- Saves best model to `models/checkpoints/best_model.pth`
- Generates training history plot

**Expected output:**
```
[2/3] Training Model...
Loading dataset...
  Total samples: 32400
  Training samples: 25920
  Validation samples: 3240
  Test samples: 3240

Initializing model...
==========================================================
ANN Architecture Summary
==========================================================
Input dimension:        1
Hidden layers:          9
Neurons per layer:      30
Total parameters:       8,732
==========================================================

Training with batch size: 25920
Epoch 1/100 | Train Loss: 1.234e-03 | Val Loss: 1.456e-03
Epoch 2/100 | Train Loss: 5.678e-04 | Val Loss: 6.789e-04
...
Epoch 45/100 | Train Loss: 2.345e-08 | Val Loss: 3.456e-08

Early stopping triggered at epoch 45

Test Set Metrics:
  MSE (overall): 2.567e-08
  MAE (overall): 1.234e-04
  Max Error: 5.678e-04

✓ Model saved to: outputs/models/best_model.pth
```

**Time:** ~5-15 minutes (depends on optimizer and data size)

---

### 3. Model Validation

**Command:**
```bash
python src/validate_model.py
```

**What it does:**
- Loads trained model
- Solves ODEs for test cases
- Compares ANN predictions with numerical solutions
- Computes 8+ error metrics
- Generates validation plots
- Creates summary CSV

**Expected output:**
```
======================================================================
VALIDATION RESULTS
======================================================================

Parameters: {'M': 1.0, 'Nr': 0.5, 'Nh': 0.5, ...}

--- f(η) Metrics ---
  MSE                      : 2.345e-08
  RMSE                     : 1.531e-04
  MAE                      : 1.123e-04
  R_squared                : 0.999998

--- θ(η) Metrics ---
  MSE                      : 1.234e-09
  RMSE                     : 3.512e-05
  MAE                      : 2.345e-05
  R_squared                : 0.999999
======================================================================

✓ Validation plots saved to: plots/validation_single_case.png
✓ Summary table saved to: plots/validation_summary.csv
```

**Generated files:**
- `outputs/plots/validation_single_case.png`: Detailed validation plots
- `outputs/plots/validation_summary.csv`: Metrics table (manuscript-ready)

---

### 4. Standalone Visualization

**Command:**
```bash
python src/visualizer.py
```

**What it does:**
- Loads the trained model and scalers
- Loads test data cases
- Generates predictions for each test case
- Creates comprehensive comparison plots
- Shows model predictions vs. numerical solutions

**Expected output:**
```
Loading model from: outputs/models/best_model.pth
Loading test data from: data/test_data.csv
Found 10 test cases

Processing test case 1/10...
Processing test case 2/10...
...
Processing test case 10/10...

✓ Visualization saved to: outputs/plots/model_predictions.png

Overall Metrics:
  MSE: 2.345e-08
  MAE: 1.234e-04
  R²: 0.999998
```

**Generated files:**
- `outputs/plots/model_predictions.png`: Multi-panel comparison plots

**When to use:**
- After training to visualize model performance
- To create publication-quality figures
- To compare predictions across multiple test cases
- When you need detailed visual analysis

---

### 5. Regenerate Training History

**Command:**
```bash
python src/main.py regenerate
```

**What it does:**
- Loads the saved model checkpoint (which includes training history)
- Extracts the training and validation loss curves
- Regenerates the training history plot with current styling
- Saves the updated plot

**Expected output:**
```
Regenerating training history plot...

✓ Training history plot regenerated successfully!
  Location: outputs/plots/training_history.png
```

**When to use:**
- You want to recreate plots with different styling or formatting
- The original training plot was lost or corrupted
- You need to adjust plot aesthetics for publication
- You want to regenerate plots after updating visualization code

**Note:** This command does NOT retrain the model—it only recreates the plot from saved data.

---

## Configuration

### Editing src/config.py

**Change parameter ranges:**
```python
PARAM_RANGES = {
    'M': [0.5, 1.0, 2.0],      # Fewer values = faster generation
    'Nr': [0.2, 0.5, 1.0],
    # ... add or remove parameters
}
```

**Change model architecture:**
```python
MODEL_PARAMS = {
    'hidden_dim': 50,           # More neurons = more capacity
    'num_hidden_layers': 12,    # More layers = deeper network
}
```

**Change training settings:**
```python
TRAIN_PARAMS = {
    'epochs': 200,              # More epochs = longer training
    'optimizer': 'lbfgs',       # 'lbfgs', 'adam', 'lm_custom'
    'batch_size': 1000,         # Smaller = less memory
}
```

---

## Advanced Usage

### Custom Data Generation

```python
import sys
from pathlib import Path
sys.path.insert(0, 'e:/University/FYDP/ANN-net')

from src.generate_data import DatasetGenerator
from src.solver.ode_solver import HybridNanofluidSolver

# Generate single case
params = {
    'M': 1.5, 'Nr': 0.8, 'Nh': 0.6, 'lam': 1.2,
    'beta': 0.1, 'Pr': 6.2, 'n': 1.0, 'Tr': 1.5, 'As': 1.0,
    'eta_max': 10.0, 'n_points': 400
}

solver = HybridNanofluidSolver(params)
eta, solution = solver.solve(verbose=True)
results = solver.compute_derivatives(eta, solution)
eng_quantities = solver.compute_engineering_quantities(solution)

print(f"Cf = {eng_quantities['Cf']:.6f}")
print(f"Nu = {eng_quantities['Nu']:.6f}")
```

### Custom Training

```python
import sys
from pathlib import Path
sys.path.insert(0, 'e:/University/FYDP/ANN-net')

from src.trainer import Trainer
from src.models.ann import HybridNanofluidANN
from src.data_loader import DataLoader

# Load data
loader = DataLoader("data/training_data.csv")
data = loader.load_data(normalize=True)

# Create model
model = HybridNanofluidANN(
    input_dim=1,
    hidden_dim=40,  # Custom size
    num_hidden_layers=12,
    output_dim=2
)

# Train
trainer = Trainer(model, optimizer_type='lbfgs', visualize=True)
history = trainer.train(
    data['X_train'], data['y_train'],
    data['X_val'], data['y_val'],
    epochs=150,
    batch_size='full'
)

# Save
trainer.save_model("my_custom_model.pth")
```

### Custom Validation

```python
import sys
from pathlib import Path
sys.path.insert(0, 'e:/University/FYDP/ANN-net')

from src.validate_model import ModelValidator

validator = ModelValidator(
    model_path="outputs/models/best_model.pth",
    scaler_dir="outputs/models"
)

# Single case
result = validator.validate_single_case(params, verbose=True)
validator.plot_validation_results(result, save_path="my_plot.png")

# Multiple cases
test_cases = [
    {**params, 'M': 0.5},
    {**params, 'M': 1.0},
    {**params, 'M': 2.0},
]
summary = validator.validate_multiple_cases(test_cases)
summary.to_csv("my_summary.csv")
```

---

## Troubleshooting

### Issue: "Training data not found"
**Solution:**
```bash
python src/main.py generate
```

### Issue: "ODE solver failed"
**Cause:** Extreme parameter values  
**Solution:** Adjust parameters in `src/config.py` or increase `max_nodes` in `src/solver/ode_solver.py`

### Issue: "Out of memory"
**Solution:** Reduce `batch_size` in `src/config.py`:
```python
TRAIN_PARAMS = {
    'batch_size': 500,  # Instead of 'full'
}
```

### Issue: "Training loss not decreasing"
**Solution:**
1. Check data normalization is enabled
2. Try different optimizer: `'optimizer': 'adam'`
3. Increase epochs: `'epochs': 200`
4. Regenerate data if corrupted

### Issue: "High validation errors"
**Cause:** Model not trained well or test case outside training range  
**Solution:**
1. Retrain with more epochs
2. Check test parameters are within `PARAM_RANGES`
3. Increase training data diversity

---

## Tips & Best Practices

### For Best Results:

1. **Data Generation:**
   - Use diverse parameter ranges
   - Ensure ODE solver converges for all cases
   - Check for NaN values in generated data

2. **Training:**
   - Use full batch for LM/L-BFGS optimizers
   - Enable early stopping to prevent overfitting
   - Monitor validation loss

3. **Validation:**
   - Test on unseen parameter combinations
   - Check multiple error metrics, not just MSE
   - Visualize predictions vs. ground truth

4. **Performance:**
   - Use L-BFGS for faster convergence
   - Enable visualization only for debugging
   - Save models frequently

---

## Output Files

### Generated During Workflow:

```
data/
├── training_data.csv          # Training dataset (~32,400 samples)
└── test_data.csv              # Test dataset (10 random cases)

outputs/models/
├── best_model.pth             # Trained model with checkpoint data
├── scaler_eta.pkl             # Input scaler (η normalization)
├── scaler_f.pkl               # Output scaler (f normalization)
└── scaler_theta.pkl           # Output scaler (θ normalization)

outputs/plots/
├── training_history.png       # Training/validation loss curves
├── model_predictions.png      # Prediction comparison plots (from visualizer.py)
├── validation_single_case.png # Single case validation plots
└── validation_summary.csv     # Comprehensive metrics table
```

**File Descriptions:**

- **`training_data.csv`**: Contains ~32,400 rows with η, f, θ, and all physics parameters
- **`test_data.csv`**: 10 random test cases for validation
- **`best_model.pth`**: PyTorch checkpoint including model weights, optimizer state, and training history
- **`scaler_*.pkl`**: Scikit-learn scalers for input/output normalization (required for predictions)
- **`training_history.png`**: Loss curves showing model convergence during training
- **`model_predictions.png`**: Multi-panel plots comparing ANN predictions with numerical solutions
- **`validation_single_case.png`**: Detailed validation plots for a single parameter set
- **`validation_summary.csv`**: Table of error metrics (MSE, RMSE, MAE, R², etc.) for all test cases

---

## Command Summary

| Command | Description | Time |
|---------|-------------|------|
| `python src/main.py generate` | Generate training data from physics | 2-5 min |
| `python src/main.py train` | Train ANN model | 5-15 min |
| `python src/main.py regenerate` | Recreate training plots from checkpoint | <5 sec |
| `python src/visualizer.py` | Create prediction comparison plots | 10-30 sec |
| `python src/validate_model.py` | Validate model with error analysis | 1-2 min |
| `python src/main.py all` | Run complete pipeline (clean → generate → train) | 8-22 min |
| `python src/main.py clean` | Remove all generated files | <1 sec |

---

**Last Updated:** 2025-11-24  
**Version:** 2.1.0
