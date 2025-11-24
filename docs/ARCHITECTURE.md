# Architecture Documentation

## Overview

This document provides technical details about the ANN model architecture, optimization algorithms, and numerical solver implementation used in this project.

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

- **Input dimension:** 1 (η - similarity variable)
- **Hidden layers:** 9
- **Neurons per layer:** 30
- **Output dimension:** 2 (f and θ)
- **Total trainable parameters:** 8,732
- **Weight initialization:** Xavier uniform
- **Activation function:** Hyperbolic tangent (tanh)
- **Framework:** PyTorch 2.x

### Design Rationale

- **9 hidden layers:** Provides sufficient depth to capture the highly nonlinear behavior of hybrid nanofluid flow and heat transfer
- **30 neurons per layer:** Balances model expressiveness with computational efficiency and prevents overfitting
- **Tanh activation:** Smooth, differentiable, bounded output [-1, 1], ideal for physics-informed problems
- **Xavier initialization:** Maintains variance of activations across layers, preventing vanishing/exploding gradients
- **No dropout/batch norm:** Not required for this regression task with sufficient training data and proper normalization

---

## Optimization Algorithms

This project supports multiple optimization algorithms, each with specific advantages for training neural networks on physics problems.

---

## Adam Optimizer

### Algorithm Overview

**Adam (Adaptive Moment Estimation)** is the primary optimizer used in this project. It's a first-order gradient-based optimization algorithm that combines the benefits of two other extensions of stochastic gradient descent:
- **AdaGrad:** Adapts learning rates based on parameter importance
- **RMSProp:** Uses moving averages of squared gradients

### Mathematical Formulation

**Update Rule:**
```
m_t = β₁ · m_{t-1} + (1 - β₁) · ∇θ        # First moment (mean)
v_t = β₂ · v_{t-1} + (1 - β₂) · (∇θ)²    # Second moment (variance)

m̂_t = m_t / (1 - β₁ᵗ)                    # Bias correction
v̂_t = v_t / (1 - β₂ᵗ)                    # Bias correction

θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)    # Parameter update

where:
  α = learning rate (default: 1e-3)
  β₁ = exponential decay rate for first moment (default: 0.9)
  β₂ = exponential decay rate for second moment (default: 0.999)
  ε = small constant for numerical stability (default: 1e-8)
```

### Why We Use Adam

**Primary Advantages:**
1. **Adaptive Learning Rates:** Automatically adjusts learning rates for each parameter, ideal for problems with sparse gradients
2. **Computational Efficiency:** Low memory requirements, computationally efficient
3. **Robust to Hyperparameters:** Works well with default settings across many problems
4. **Handles Non-Stationary Objectives:** Effective for problems where the objective function changes during training
5. **Mini-Batch Friendly:** Works efficiently with mini-batch training, unlike LM which requires full batch

**For This Project:**
- **Scalability:** Handles large datasets efficiently with mini-batch processing
- **Stability:** Less sensitive to initial conditions than second-order methods
- **Flexibility:** Easy to tune and adjust during experimentation
- **Memory Efficient:** Doesn't require computing and storing the Hessian matrix
- **Proven Performance:** Widely used in deep learning with excellent empirical results

### PyTorch Implementation

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,           # Learning rate
    betas=(0.9, 0.999), # Exponential decay rates
    eps=1e-8,          # Numerical stability
    weight_decay=0     # L2 regularization (if needed)
)
```

**Training Loop:**
```python
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

---

## Levenberg-Marquardt Optimization

### Algorithm Overview

The LM algorithm is a hybrid optimization method combining:
- **Gradient Descent:** Robust when far from optimum
- **Gauss-Newton:** Fast convergence near optimum

**Note:** While LM is available in this project, Adam is the primary optimizer due to its better scalability and ease of use.

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

#### 1. scipy.optimize.least_squares
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
- Robust convergence for small-scale problems

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

### When to Use LM vs Adam

| Aspect | Adam | Levenberg-Marquardt |
|--------|------|---------------------|
| **Dataset Size** | Large (any size) | Small to medium |
| **Batch Processing** | Mini-batch friendly | Requires full batch |
| **Memory Usage** | Low | High (stores Jacobian) |
| **Convergence Speed** | Moderate | Fast (near optimum) |
| **Ease of Use** | Very easy | Requires tuning |
| **Recommended For** | General use, production | Research, small datasets |

### Convergence Criteria

Training stops when:
- **Function tolerance:** `|Δloss| < ftol = 1e-8`
- **Parameter tolerance:** `|Δp| < xtol = 1e-8`
- **Gradient tolerance:** `|∇loss| < gtol = 1e-8`
- **Max iterations:** Reached max_nfev or early stopping triggered

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
2. Solve BVP for each combination using `scipy.integrate.solve_bvp`
3. Extract f(η), θ(η) and derivatives at 400 points per case
4. Save to `data/training_data.csv`

**Parameter Ranges:**
```python
PARAM_RANGES = {
    'M': [0.5, 1.0, 2.0],      # Magnetic parameter (3 values)
    'Nr': [0.5, 1.0, 1.5],     # Radiation parameter (3 values)
    'Nh': [0.1, 0.5],          # Convective heat transfer (2 values)
    'lam': [0.5, 1.0, 1.5],    # Mixed convection (3 values)
    'beta': [0.1, 0.2],        # Velocity slip (2 values)
    'Pr': [6.2],               # Prandtl number (1 value)
    'n': [1.0],                # Power-law index (1 value)
    'Tr': [1.5],               # Temperature ratio (1 value)
    'As': [1.0],               # Suction/injection (1 value)
}
# Total combinations: 3 × 3 × 2 × 3 × 2 = 108 cases
```

**Dataset Size:**
- **Parameter combinations:** 108
- **Points per case:** 400
- **Total data points:** 43,200 samples
- **Features per point:** η (input), f, θ (outputs), plus derivatives and parameters

**Output format:**
```csv
case_id, eta, f, fp, fpp, theta, thetap, M, Nr, Nh, lam, beta, Pr, n, Tr, As, Cf, Nu
```

### 2. Data Loading

```
CSV → Normalization → Train/Val/Test Split → PyTorch Tensors
```

**Normalization:**
- **η (input):** MinMaxScaler to [0, 1]
- **f (output):** MinMaxScaler to [0, 1]
- **θ (output):** MinMaxScaler to [0, 1]

All scalers are saved to `outputs/models/` for use during inference.

**Split Ratios:**
- **Train:** 80% (~34,560 samples)
- **Validation:** 10% (~4,320 samples)
- **Test:** 10% (~4,320 samples)

### 3. Training

```
Batches → Forward Pass → Loss Computation → Adam Optimizer → Weight Update
```

**Training Configuration:**
```python
TRAIN_PARAMS = {
    'epochs': 100,
    'batch_size': 512,                    # Mini-batch training
    'optimizer': 'adam',                  # Primary optimizer
    'early_stopping_patience': 1000,      # Epochs before stopping
    'test_split': 0.1,
    'val_split': 0.1
}
```

**Loss function:** Mean Squared Error (MSE)
```python
loss = (1/N) * Σ[(f_pred - f_true)^2 + (θ_pred - θ_true)^2]
```

**Optimizer Settings:**
- **Algorithm:** Adam (Adaptive Moment Estimation)
- **Learning rate:** 1e-3 (default)
- **Batch size:** 512 samples per iteration
- **Early stopping:** Monitors validation loss with patience of 1000 epochs

---

## Visualization Architecture

The project includes multiple visualization capabilities for different purposes:

### 1. Real-Time Training Visualization

**Location:** `src/visualizer.py` → `RealTimeVisualizer` class

**Purpose:** Monitor training progress in real-time during model training

**Features:**
- Live loss curves (training and validation)
- Network architecture diagram
- Weight distribution histograms
- Gradient magnitude tracking

**Usage:** Automatically enabled when `visualize=True` in `Trainer`

### 2. Training History Plots

**Location:** `src/visualizer.py` → `plot_training_history()` function

**Purpose:** Create publication-quality training history plots

**Features:**
- Dual-axis loss curves with logarithmic scale
- Epoch markers and grid
- Professional Seaborn styling
- Automatic best epoch highlighting

**Generated by:**
- `python src/main.py train` (automatically after training)
- `python src/main.py regenerate` (from saved checkpoint)

### 3. Prediction Comparison Plots

**Location:** `src/visualizer.py` → `plot_predictions()` function

**Purpose:** Compare model predictions against test data

**Features:**
- Multi-panel layout for multiple test cases
- Side-by-side comparison of f(η) and θ(η)
- Residual error plots
- Overall performance metrics

**Generated by:**
- `python src/visualizer.py` (standalone script)

### 4. Validation Plots

**Location:** `src/validate_model.py` → `ModelValidator.plot_validation_results()`

**Purpose:** Rigorous validation against numerical solutions

**Features:**
- Detailed error analysis
- Multiple error metrics (MSE, RMSE, MAE, R²)
- Residual plots
- Statistical validation

**Generated by:**
- `python src/validate_model.py`

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
                                      (includes: weights, optimizer state, history)
                                              ↓
                        ┌─────────────────────┼─────────────────────┐
                        ↓                     ↓                     ↓
              src/visualizer.py    src/validate_model.py    src/main.py regenerate
                        ↓                     ↓                     ↓
         outputs/plots/          outputs/plots/          outputs/plots/
         model_predictions.png   validation_*.png        training_history.png
                                 validation_summary.csv
```

**Checkpoint Structure:**

The `best_model.pth` file contains:
- `model_state_dict`: Trained neural network weights
- `optimizer_state_dict`: Optimizer state (for resuming training)
- `history`: Training and validation loss curves
- `epoch`: Final epoch number
- `best_val_loss`: Best validation loss achieved

This comprehensive checkpoint enables:
1. **Model inference**: Load weights for predictions
2. **Training resumption**: Continue training from checkpoint
3. **History regeneration**: Recreate training plots without retraining

---

## Performance Considerations

### Memory Usage

- **Training data:** 43,200 samples × 2 outputs = ~86,400 values (~691 KB for float64)
- **Model parameters:** 8,732 floats ≈ 35 KB
- **Batch processing:** 512 samples per batch ≈ 4 KB per batch
- **Scalers:** 3 scaler objects (η, f, θ) ≈ 10 KB total
- **Total training memory:** < 10 MB (very lightweight)

### Computational Complexity

- **Forward pass:** O(L × N × M) where L=10 layers, N=30 neurons, M=batch_size (512)
- **Adam optimizer:** O(P) where P=8,732 parameters (linear complexity)
- **Data generation:** O(C × I) where C=108 cases, I=iterations per BVP solve
- **Training time:** ~5-15 minutes on CPU (depends on convergence)

### Optimization Tips

1. **Use Adam optimizer:** Default choice for scalability and ease of use
2. **Mini-batch training:** Batch size of 512 balances speed and memory
3. **GPU acceleration:** Move to CUDA for faster training (optional)
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   ```
4. **Early stopping:** Prevents overfitting and saves computation time
5. **Parallel data generation:** Use multiprocessing for ODE solving
6. **Data caching:** Generated datasets are reused across training runs

### Scalability

**Current Configuration:**
- Dataset: 108 cases × 400 points = 43,200 samples
- Training time: ~5-15 minutes on modern CPU
- Memory footprint: < 10 MB

**Scaling Up:**
- Can handle 10× more data (1,000+ cases) with same architecture
- Increase batch size for larger datasets (e.g., 1024 or 2048)
- GPU training recommended for datasets > 100,000 samples
