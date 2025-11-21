"""
Evaluation Script for Hybrid Nanofluid ANN
Generates manuscript-quality plots comparing ANN vs Numerical solutions
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from models.ann import HybridNanofluidANN, ANNWithDerivatives
from solver.ode_solver import HybridNanofluidSolver

def evaluate_and_plot():
    # Paths
    MODEL_PATH = "models/checkpoints/best_model.pth"
    SCALER_DIR = "models/checkpoints"
    PLOT_DIR = "plots"
    Path(PLOT_DIR).mkdir(exist_ok=True)
    
    print("Loading resources...")
    
    # Load Scalers
    with open(f"{SCALER_DIR}/scaler_eta.pkl", "rb") as f:
        scaler_eta = pickle.load(f)
    with open(f"{SCALER_DIR}/scaler_f.pkl", "rb") as f:
        scaler_f = pickle.load(f)
    with open(f"{SCALER_DIR}/scaler_theta.pkl", "rb") as f:
        scaler_theta = pickle.load(f)
        
    # Load Model
    checkpoint = torch.load(MODEL_PATH)
    model = HybridNanofluidANN(input_dim=1, hidden_dim=30, num_hidden_layers=9, output_dim=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Wrap with derivative model
    deriv_model = ANNWithDerivatives(model)
    
    # Define a standard test case
    params = {
        'M': 1.0, 'Nr': 0.5, 'Nh': 0.5, 'lam': 0.5, 'beta': 0.1,
        'Pr': 6.2, 'n': 1.0, 'Tr': 1.5, 'As': 1.0,
        'eta_max': 10.0, 'n_points': 100
    }
    
    print(f"Solving Numerical Case: {params}")
    solver = HybridNanofluidSolver(params)
    eta_true, sol_true = solver.solve()
    
    if sol_true is None:
        print("Numerical solver failed!")
        return

    results_true = solver.compute_derivatives(eta_true, sol_true)
    f_true = results_true['f']
    fp_true = results_true['fp']
    theta_true = results_true['theta']
    
    # ANN Prediction
    print("Generating ANN Predictions...")
    
    # Prepare input
    eta_norm = scaler_eta.transform(eta_true.reshape(-1, 1))
    eta_tensor = torch.FloatTensor(eta_norm)
    
    # Predict
    results_pred = deriv_model(eta_tensor)
    
    # Extract and Inverse Scale
    f_pred_norm = results_pred['f'].detach().numpy()
    theta_pred_norm = results_pred['theta'].detach().numpy()
    fp_pred_norm = results_pred['fp'].detach().numpy() # df_norm / deta_norm
    
    f_pred = scaler_f.inverse_transform(f_pred_norm).flatten()
    theta_pred = scaler_theta.inverse_transform(theta_pred_norm).flatten()
    
    # Chain rule for derivative: df/deta = (df_norm/deta_norm) * (scale_f / scale_eta)
    scale_f = scaler_f.data_range_[0]
    scale_eta = scaler_eta.data_range_[0]
    fp_pred = fp_pred_norm.flatten() * (scale_f / scale_eta)
    
    print("\nDebug: Velocity Profile near wall (eta < 1.0)")
    print(f"{'eta':<10} | {'fp_true':<10} | {'fp_pred':<10} | {'Diff':<10}")
    print("-" * 46)
    for i in range(10):
        print(f"{eta_true[i]:<10.4f} | {fp_true[i]:<10.4f} | {fp_pred[i]:<10.4f} | {fp_true[i]-fp_pred[i]:<10.4f}")
    print("-" * 46)
    
    # Plotting
    print("Generating Plots...")
    
    # 1. Velocity Profile
    plt.figure(figsize=(8, 6))
    plt.plot(eta_true, fp_true, 'k-', linewidth=2, label='Numerical (Exact)')
    plt.plot(eta_true, fp_pred, 'r--', linewidth=2, label='ANN Prediction')
    plt.xlabel('$\eta$', fontsize=14)
    plt.ylabel("$f'(\eta)$ (Velocity)", fontsize=14)
    plt.title(f"Velocity Profile Comparison\n$M={params['M']}, \lambda={params['lam']}$", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/eval_velocity_profile.png", dpi=300)
    
    # 2. Temperature Profile
    plt.figure(figsize=(8, 6))
    plt.plot(eta_true, theta_true, 'k-', linewidth=2, label='Numerical (Exact)')
    plt.plot(eta_true, theta_pred, 'b--', linewidth=2, label='ANN Prediction')
    plt.xlabel('$\eta$', fontsize=14)
    plt.ylabel("$\theta(\eta)$ (Temperature)", fontsize=14)
    plt.title(f"Temperature Profile Comparison\n$M={params['M']}, \lambda={params['lam']}$", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/eval_temperature_profile.png", dpi=300)
    
    print(f"Plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    evaluate_and_plot()
