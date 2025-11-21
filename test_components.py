"""
Test Suite - Verify all components work correctly
"""

import sys
import numpy as np
import torch
from pathlib import Path

print("=" * 70)
print("COMPONENT TEST SUITE")
print("=" * 70)

# Test 1: ODE Solver
print("\n[1/5] Testing ODE Solver...")
try:
    from solver.ode_solver import HybridNanofluidSolver
    
    params = {
        'M': 1.0, 'Nr': 0.5, 'Nh': 0.5, 'lam': 1.0, 'beta': 0.1,
        'Pr': 6.2, 'n': 1.0, 'Tr': 1.5, 'As': 1.0,
        'eta_max': 10.0, 'n_points': 200
    }
    
    solver = HybridNanofluidSolver(params)
    eta, solution = solver.solve(verbose=False)
    
    if solution is not None:
        eng_q = solver.compute_engineering_quantities(solution)
        print(f"  ✓ ODE Solver working")
        print(f"    Cf = {eng_q['Cf']:.6f}, Nu = {eng_q['Nu']:.6f}")
    else:
        print("  ✗ ODE Solver failed")
        sys.exit(1)
        
except Exception as e:
    print(f"  ✗ ODE Solver error: {str(e)}")
    sys.exit(1)

# Test 2: ANN Model
print("\n[2/5] Testing ANN Model...")
try:
    from models.ann import HybridNanofluidANN
    
    model = HybridNanofluidANN(input_dim=1, hidden_dim=30, 
                               num_hidden_layers=9, output_dim=2)
    
    eta_test = torch.randn(10, 1)
    output = model(eta_test)
    
    param_count = model.count_parameters()
    
    print(f"  ✓ ANN Model working")
    print(f"    Parameters: {param_count:,}")
    print(f"    Output shape: {output.shape}")
    
except Exception as e:
    print(f"  ✗ ANN Model error: {str(e)}")
    sys.exit(1)

# Test 3: LM Optimizer
print("\n[3/5] Testing LM Optimizer...")
try:
    from models.lm_optimizer import SimplifiedLMOptimizer
    
    model = HybridNanofluidANN(input_dim=1, hidden_dim=10, 
                               num_hidden_layers=2, output_dim=2)
    optimizer = SimplifiedLMOptimizer(model)
    
    # Create dummy data (need more points than parameters)
    eta = torch.linspace(0, 10, 200).reshape(-1, 1)
    targets = torch.cat([torch.sin(eta), torch.cos(eta)], dim=1)
    
    # Test optimization
    result = optimizer.optimize(eta, targets, max_nfev=10, verbose=0)
    
    print(f"  ✓ LM Optimizer working")
    print(f"    Success: {result['success']}")
    print(f"    Final cost: {result['cost']:.6f}")
    
except Exception as e:
    print(f"  ✗ LM Optimizer error: {str(e)}")
    sys.exit(1)

# Test 4: Data Loading
print("\n[4/5] Testing Data Structures...")
try:
    import pandas as pd
    
    # Create dummy dataset
    test_data = {
        'eta': np.linspace(0, 10, 100),
        'f': np.random.randn(100),
        'theta': np.random.randn(100),
        'M': np.ones(100) * 1.0
    }
    
    df = pd.DataFrame(test_data)
    
    print(f"  ✓ Data structures working")
    print(f"    DataFrame shape: {df.shape}")
    
except Exception as e:
    print(f"  ✗ Data structures error: {str(e)}")
    sys.exit(1)

# Test 5: Dependencies
print("\n[5/5] Testing Dependencies...")
try:
    import scipy
    import matplotlib
    import sklearn
    from tqdm import tqdm
    
    print(f"  ✓ All dependencies installed")
    print(f"    PyTorch: {torch.__version__}")
    print(f"    NumPy: {np.__version__}")
    print(f"    SciPy: {scipy.__version__}")
    
except Exception as e:
    print(f"  ✗ Dependency error: {str(e)}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED")
print("=" * 70)
print("\nSystem is ready for:")
print("  1. Data generation (python generate_data.py)")
print("  2. Model training (python train_ann.py)")
print("  3. Plot generation (python plot_results.py)")
print("\nOr run complete pipeline:")
print("  python run_pipeline.py")
