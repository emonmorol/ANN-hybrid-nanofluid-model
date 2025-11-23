# Architecture Documentation

## Overview

This document provides technical details about the ANN model architecture, optimization algorithm, and numerical solver implementation.

---

## ANN Architecture

### Network Structure

```
Input: η (similarity variable)
  ↓
Layer 1: Linear(1 → 30) + Tanh
Layer 2: Linear(30 → 30) + Tanh
Layer 3: Linear(30 → 30) + Tanh
Layer 4: Linear(30 → 30) + Tanh
Layer 5: Linear(30 → 30) + Tanh
Layer 6: Linear(30 → 30) + Tanh
Layer 7: Linear(30 → 30) + Tanh
Layer 8: Linear(30 → 30) + Tanh
Layer 9: Linear(30 → 30) + Tanh
Output: Linear(30 → 2) → [f(η), θ(η)]
```

### Parameters

- **Total trainable parameters:** 8,732
- **Weight initialization:** Xavier uniform
- **Activation function:** Hyperbolic tangent (tanh)
- **Framework:** PyTorch

### Design Rationale

- **9 hidden layers:** Sufficient depth for complex nonlinear mappings
- **30 neurons per layer:** Balances expressiveness and computational efficiency
- **Tanh activation:** Smooth, differentiable, bounded output [-1, 1]
- **No dropout/batch norm:** Not needed for this regression task with sufficient data

---

## Levenberg-Marquardt Optimization

### Algorithm Overview

The LM algorithm is a hybrid optimization method combining:
- **Gradient Descent:** Robust when far from optimum
- **Gauss-Newton:** Fast convergence near optimum

### Mathematical Formulation

**Update Rule:**
```
(J^T·J + λ·I)·Δp = -J^T·r

where:
  J = Jacobian matrix (∂r/∂p)
  r = residuals (predictions - targets)
  λ = damping parameter
  I = identity matrix
  Δp = parameter update
```

**Adaptive Damping:**
```
if loss_new < loss_old:
    λ = λ × 0.1  # Move toward Gauss-Newton
    accept update
else:
    λ = λ × 10   # Move toward gradient descent
    reject update
```

### Implementation

Two implementations available:

#### 1. scipy.optimize.least_squares (Recommended)
```python
from scipy.optimize import least_squares

result = least_squares(
    residual_function,
    initial_params,
    method='lm',
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    max_nfev=1000
)
```

**Advantages:**
- Highly optimized C implementation
- Automatic Jacobian computation
- Robust convergence

#### 2. Custom PyTorch Implementation
```python
from src.models.lm_optimizer import LevenbergMarquardtOptimizer

optimizer = LevenbergMarquardtOptimizer(
    model.parameters(),
    lambda_init=1e-3,
    lambda_min=1e-7,
    lambda_max=1e7
)
```

**Advantages:**
- Full control over optimization
- GPU acceleration support
- Research flexibility

### Convergence Criteria

Training stops when:
- **Function tolerance:** `|Δloss| < ftol = 1e-8`
- **Parameter tolerance:** `|Δp| < xtol = 1e-8`
- **Gradient tolerance:** `|∇loss| < gtol = 1e-8`
- **Max iterations:** Reached max_nfev

---

## Numerical ODE Solver

### Problem Formulation

**Boundary Value Problem (BVP):**

Variables: `y = [f, f', f'', θ, θ']`

**ODEs:**
```python
dy/dη = [
    f',                    # df/dη
    f'',                   # df'/dη
    f''',                  # df''/dη (from momentum eq.)
    θ',                    # dθ/dη
    θ''                    # dθ'/dη (from energy eq.)
]
```

**Boundary Conditions:**
```python
At η = 0:
    y[0] = 0              # f(0) = 0
    y[1] = 1 + β·y[2]     # f'(0) = 1 + β·f''(0)
    y[4] = -Nh·(1-y[3])   # θ'(0) = -Nh·(1-θ(0))

At η = η_max:
    y[1] = 1              # f'(∞) → 1
    y[3] = 0              # θ(∞) → 0
```

### Solver Configuration

```python
from scipy.integrate import solve_bvp

solution = solve_bvp(
    fun=ode_system,
    bc=boundary_conditions,
    x=eta_grid,
    y=initial_guess,
    tol=1e-6,
    max_nodes=5000
)
```

**Parameters:**
- `eta_max = 10.0`: Far-field boundary
- `n_points = 400`: Grid resolution
- `tol = 1e-6`: Solver tolerance
- `max_nodes = 5000`: Maximum adaptive nodes

### Initial Guess Strategy

```python
# Polynomial initial guess
f_guess = eta - (1/6) * eta^3
theta_guess = exp(-eta)
```

This provides a reasonable starting point for most parameter combinations.

---

## Data Pipeline

### 1. Data Generation

```
Parameter Grid → ODE Solver → Numerical Solutions → CSV
```

**Process:**
1. Generate all parameter combinations from `src/config.PARAM_RANGES`
2. Solve BVP for each combination
3. Extract f(η), θ(η) at 400 points
4. Save to `data/training_data.csv`

**Output format:**
```
case_id, eta, f, fp, fpp, theta, thetap, M, Nr, Nh, lam, beta, Pr, n, Tr, As, Cf, Nu
```

### 2. Data Loading

```
CSV → Normalization → Train/Val/Test Split → PyTorch Tensors
```

**Normalization:**
- η: MinMaxScaler to [0, 1]
- f: MinMaxScaler to [0, 1]
- θ: MinMaxScaler to [0, 1]

**Split:**
- Train: 80%
- Validation: 10%
- Test: 10%

### 3. Training

```
Batches → Forward Pass → Loss → LM Optimizer → Update Weights
```

**Loss function:** Mean Squared Error (MSE)
```python
loss = (1/N) * Σ[(f_pred - f_true)^2 + (θ_pred - θ_true)^2]
```

---

## File Organization

### Core Modules

- **`src/config.py`**: Central configuration (parameters, paths, hyperparameters)
- **`src/main.py`**: Orchestration layer (CLI interface)
- **`src/generate_data.py`**: Dataset generation pipeline
- **`src/data_loader.py`**: Data loading and preprocessing
- **`src/trainer.py`**: Training loop and optimization
- **`src/visualizer.py`**: Real-time and static plotting
- **`src/validate_model.py`**: Model validation and metrics

### Model Components

- **`src/models/ann.py`**: ANN architecture definition
- **`src/models/lm_optimizer.py`**: Custom LM optimizer
- **`src/solver/ode_solver.py`**: Numerical BVP solver

### Data Flow

```
src/config.py → src/generate_data.py → data/training_data.csv
                                              ↓
                                      src/data_loader.py
                                              ↓
                                      src/trainer.py + src/models/ann.py
                                              ↓
                                      outputs/models/best_model.pth
                                              ↓
                                      src/validate_model.py
                                              ↓
                                      outputs/plots/validation_*.png
```

---

## Performance Considerations

### Memory Usage

- **Training data:** ~32,400 samples × 2 outputs = ~64,800 values
- **Model parameters:** 8,732 floats ≈ 35 KB
- **Batch processing:** Full batch uses ~2 MB RAM

### Computational Complexity

- **Forward pass:** O(L × N × M) where L=layers, N=neurons, M=batch_size
- **LM update:** O(P^2) where P=parameters (8,732)
- **Data generation:** O(C × I) where C=cases (81), I=iterations per solve

### Optimization Tips

1. **Use full batch:** LM works best with full batch
2. **GPU acceleration:** Move to CUDA for larger models
3. **Parallel data generation:** Use multiprocessing for ODE solving
4. **Early stopping:** Prevents overfitting and saves time

---

## References

- **Levenberg-Marquardt:** Marquardt, D. W. (1963). SIAM Journal on Applied Mathematics
- **solve_bvp:** SciPy documentation
- **Xavier initialization:** Glorot & Bengio (2010). AISTATS

---

**Last Updated:** 2025-11-23  
**Version:** 2.0.0
