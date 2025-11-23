# Usage Guide

## Quick Reference

```bash
# Complete workflow
python src/main.py all

# Individual steps
python src/main.py generate    # Generate training data
python src/main.py train       # Train model
python src/validate_model.py   # Validate model
python src/main.py clean       # Clean generated files
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
- Trains using L-BFGS or LM optimizer
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
├── training_data.csv          # Training dataset
└── test_data.csv              # Test dataset

outputs/models/
├── best_model.pth             # Trained model
├── scaler_eta.pkl             # Input scaler
├── scaler_f.pkl               # Output scaler (f)
└── scaler_theta.pkl           # Output scaler (θ)

outputs/plots/
├── training_history.png       # Loss curves
├── validation_single_case.png # Validation plots
└── validation_summary.csv     # Metrics table
```

---

## Command Summary

| Command | Description | Time |
|---------|-------------|------|
| `python src/main.py generate` | Generate training data | 2-5 min |
| `python src/main.py train` | Train ANN model | 5-15 min |
| `python src/validate_model.py` | Validate model | 1-2 min |
| `python src/main.py all` | Run complete pipeline | 8-22 min |
| `python src/main.py clean` | Remove generated files | <1 sec |

---

**Last Updated:** 2025-11-23  
**Version:** 2.0.0
