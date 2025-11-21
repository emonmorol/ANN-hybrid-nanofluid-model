# Getting Started Guide

## ANN Model for Hybrid Nanofluid Boundary Layer Flow

Welcome! This guide will help you get started with the project in 5 minutes.

---

## Prerequisites

✅ **Python 3.10+** installed
✅ **pip** package manager
✅ **10 GB** free disk space
✅ **Windows/Linux/Mac** OS

---

## Installation (2 minutes)

### Step 1: Navigate to Project Directory
```bash
cd "e:/University/FYDP/ANN net"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**What gets installed:**
- PyTorch (deep learning)
- NumPy (numerical computing)
- SciPy (ODE solver)
- Matplotlib (plotting)
- Pandas (data handling)
- scikit-learn (preprocessing)
- tqdm (progress bars)

---

## Verify Installation (1 minute)

Run the component test suite:

```bash
python test_components.py
```

**Expected output:**
```
======================================================================
COMPONENT TEST SUITE
======================================================================

[1/5] Testing ODE Solver...
  ✓ ODE Solver working

[2/5] Testing ANN Model...
  ✓ ANN Model working

[3/5] Testing LM Optimizer...
  ✓ LM Optimizer working

[4/5] Testing Data Structures...
  ✓ Data structures working

[5/5] Testing Dependencies...
  ✓ All dependencies installed

======================================================================
✓ ALL TESTS PASSED
======================================================================
```

If all tests pass, you're ready to go! 🎉

---

## Quick Start (2 minutes to start)

### Option A: Automated Pipeline (Recommended)

Run everything automatically:

```bash
python run_pipeline.py
```

**What it does:**
1. Generates 32,400 training samples from numerical ODE solutions
2. Trains 9-layer ANN with Levenberg-Marquardt optimizer
3. Generates 6 manuscript-style plots

**Time required:** 15-45 minutes (mostly automated)

---

### Option B: Step-by-Step (For Learning)

#### Step 1: Generate Training Data (5-10 min)
```bash
python generate_data.py
```

**Output:**
- `data/training_data.csv` (32,400 rows)
- `data/test_data.csv` (4,000 rows)

**What happens:**
- Solves ODE system for 81 parameter combinations
- Each case: 400 points from η=0 to η=10
- Computes f, f', f'', θ, θ' for each point

---

#### Step 2: Train ANN Model (10-30 min)
```bash
python train_ann.py
```

**Output:**
- `models/checkpoints/best_model.pth`
- `models/checkpoints/scaler_*.pkl`
- `plots/training_history.png`

**What happens:**
- Loads and normalizes data
- Splits: 80% train, 10% val, 10% test
- Trains 9-layer ANN (7,562 parameters)
- Uses Levenberg-Marquardt optimization
- Saves best model based on validation loss

**Training progress:**
```
Epoch   1/100 | Train Loss: 0.123456 | Val Loss: 0.234567 | Time: 2.34s
Epoch  10/100 | Train Loss: 0.012345 | Val Loss: 0.023456 | Time: 2.12s
...
✓ Training complete!
  Best validation loss: 0.000123
```

---

#### Step 3: Generate Plots (2-5 min)
```bash
python plot_results.py
```

**Output:**
- `plots/velocity_profile_M.png`
- `plots/temperature_profile_Nr.png`
- `plots/ann_vs_numerical.png`
- `plots/cf_vs_M.png`
- `plots/nu_vs_Nr.png`

**What happens:**
- Loads trained model
- Compares ANN predictions with numerical solutions
- Generates manuscript-style figures
- Shows velocity and temperature profiles
- Plots engineering quantities (Cf, Nu)

---

## Understanding the Output

### 1. Training Data (`data/training_data.csv`)

**Columns:**
- `eta`: Similarity variable (0 to 10)
- `f`: Stream function
- `fp`: Velocity (f')
- `fpp`: Velocity gradient (f'')
- `theta`: Temperature
- `thetap`: Temperature gradient (θ')
- `M`, `Nr`, `Nh`, `lam`, etc.: Parameters
- `Cf`: Skin friction coefficient
- `Nu`: Nusselt number

**Sample:**
```
eta,    f,      fp,     fpp,    theta,  thetap, M,   Nr,  ...
0.0,    0.000,  1.000,  0.123,  0.172,  -0.414, 1.0, 0.5, ...
0.025,  0.025,  0.999,  0.121,  0.165,  -0.398, 1.0, 0.5, ...
...
```

---

### 2. Trained Model (`models/checkpoints/best_model.pth`)

**Contains:**
- Model weights (7,562 parameters)
- Training history
- Best validation loss

**To load:**
```python
import torch
from models.ann import HybridNanofluidANN

model = HybridNanofluidANN()
checkpoint = torch.load('models/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

### 3. Plots (`plots/*.png`)

**velocity_profile_M.png**
- Shows f'(η) for different magnetic parameters
- Demonstrates effect of M on velocity

**temperature_profile_Nr.png**
- Shows θ(η) for different radiation parameters
- Demonstrates effect of Nr on temperature

**ann_vs_numerical.png**
- Compares ANN predictions with numerical solutions
- Shows prediction errors

**cf_vs_M.png**
- Skin friction coefficient vs magnetic parameter

**nu_vs_Nr.png**
- Nusselt number vs radiation parameter

---

## Common Questions

### Q: How long does it take?
**A:** 
- Data generation: 5-10 minutes
- Training: 10-30 minutes (depends on convergence)
- Plotting: 2-5 minutes
- **Total: 15-45 minutes**

### Q: Can I run on CPU?
**A:** Yes! The code is designed for CPU. GPU is not required.

### Q: How much memory needed?
**A:** ~2-4 GB RAM for training, ~1 GB for data.

### Q: Can I modify parameters?
**A:** Yes! Edit `generate_data.py`:
```python
M_values = [0.5, 1.0, 2.0]  # Change these
Nr_values = [0.2, 0.5, 1.0]
# etc.
```

### Q: What if training fails?
**A:** 
1. Check data exists: `data/training_data.csv`
2. Verify all tests pass: `python test_components.py`
3. Try reducing batch size in `train_ann.py`

### Q: How to use the trained model?
**A:** 
```python
import torch
import numpy as np
from models.ann import HybridNanofluidANN

# Load model
model = HybridNanofluidANN()
checkpoint = torch.load('models/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
eta = np.linspace(0, 10, 100).reshape(-1, 1)
predictions = model.predict(eta)
f = predictions[:, 0]
theta = predictions[:, 1]
```

---

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution:**
```bash
pip install torch numpy scipy matplotlib pandas scikit-learn tqdm
```

### Issue: "ODE solver failed"
**Solution:** Some parameter combinations may not converge. This is expected. The code handles this gracefully.

### Issue: "Training loss not decreasing"
**Solution:** 
- Increase `max_nfev` in `train_ann.py` (line ~200)
- Check data normalization
- Verify data quality

### Issue: "Memory error"
**Solution:** Reduce `batch_size` in `train_ann.py`:
```python
batch_size=1000  # Try 500 or 250
```

---

## Next Steps

After running the pipeline:

1. **Review Results**
   - Check plots in `plots/` directory
   - Examine training metrics
   - Verify boundary conditions

2. **Experiment**
   - Modify parameters in `generate_data.py`
   - Try different ANN architectures in `models/ann.py`
   - Adjust training settings in `train_ann.py`

3. **Extend**
   - Add new physical parameters
   - Implement sensitivity analysis
   - Compare with other optimizers

4. **Document**
   - Save your results
   - Note parameter combinations
   - Record observations

---

## File Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `test_components.py` | Verify installation | Before starting |
| `run_pipeline.py` | Run everything | Quick start |
| `generate_data.py` | Create dataset | Custom parameters |
| `train_ann.py` | Train model | After data generation |
| `plot_results.py` | Visualize | After training |
| `README.md` | Full documentation | Detailed reference |
| `SUMMARY.md` | Quick overview | Project summary |

---

## Support

For help:
1. Read `README.md` for detailed documentation
2. Run `python test_components.py` to verify setup
3. Check manuscript for physical model details
4. Review code comments for implementation details

---

## Success Criteria

You'll know everything is working when:

✅ All component tests pass
✅ Data generation completes without errors
✅ Training converges (validation loss decreases)
✅ Plots are generated successfully
✅ Boundary conditions are satisfied
✅ ANN predictions match numerical solutions

---

**Ready to start?**

```bash
python run_pipeline.py
```

**Good luck! 🚀**

---

*Last Updated: 2025-11-21*
*Version: 1.0.0*
