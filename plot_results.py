"""
Plotting and Visualization Module
Reproduce manuscript-style figures for velocity, temperature, Cf, and Nu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import pickle
from typing import Dict, List

from models.ann import HybridNanofluidANN
from solver.ode_solver import HybridNanofluidSolver


class Plotter:
    """Generate manuscript-style plots"""
    
    def __init__(self, model_path: str, scaler_dir: str, output_dir: str = "plots"):
        """
        Initialize plotter
        
        Parameters:
        -----------
        model_path : str
            Path to trained model checkpoint
        scaler_dir : str
            Directory containing saved scalers
        output_dir : str
            Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load model
        self.model = HybridNanofluidANN(input_dim=1, hidden_dim=30, 
                                        num_hidden_layers=9, output_dim=2)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scalers
        scaler_dir = Path(scaler_dir)
        with open(scaler_dir / 'scaler_eta.pkl', 'rb') as f:
            self.scaler_eta = pickle.load(f)
        with open(scaler_dir / 'scaler_f.pkl', 'rb') as f:
            self.scaler_f = pickle.load(f)
        with open(scaler_dir / 'scaler_theta.pkl', 'rb') as f:
            self.scaler_theta = pickle.load(f)
        
        print("✓ Model and scalers loaded successfully")
    
    def predict(self, eta: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict f and θ for given η values
        
        Returns:
        --------
        dict with keys: eta, f, theta
        """
        # Normalize eta
        eta_normalized = self.scaler_eta.transform(eta.reshape(-1, 1))
        
        # Predict
        with torch.no_grad():
            eta_tensor = torch.FloatTensor(eta_normalized)
            predictions = self.model(eta_tensor).numpy()
        
        # Denormalize
        f = self.scaler_f.inverse_transform(predictions[:, 0:1]).flatten()
        theta = self.scaler_theta.inverse_transform(predictions[:, 1:2]).flatten()
        
        return {
            'eta': eta,
            'f': f,
            'theta': theta
        }
    
    def compute_derivatives_numerical(self, eta: np.ndarray, f: np.ndarray, 
                                     theta: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute derivatives using finite differences
        """
        fp = np.gradient(f, eta)
        fpp = np.gradient(fp, eta)
        thetap = np.gradient(theta, eta)
        
        return {
            'fp': fp,
            'fpp': fpp,
            'thetap': thetap
        }
    
    def plot_velocity_profile(self, param_variations: List[Dict], 
                              vary_param: str, save_name: str = "velocity_profile.png"):
        """
        Plot velocity profile f'(η) for different parameter values
        
        Parameters:
        -----------
        param_variations : list of dict
            List of parameter dictionaries
        vary_param : str
            Name of varying parameter (for legend)
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        eta = np.linspace(0, 10, 200)
        
        for params in param_variations:
            # Solve numerically
            solver = HybridNanofluidSolver(params)
            eta_sol, solution = solver.solve(verbose=False)
            
            if solution is not None:
                fp = solution[1]  # f'
                
                param_value = params[vary_param]
                label = f"{vary_param} = {param_value}"
                ax.plot(eta_sol, fp, linewidth=2, label=label, marker='o', 
                       markersize=4, markevery=20)
        
        ax.set_xlabel(r'$\eta$', fontsize=14)
        ax.set_ylabel(r"$f'(\eta)$", fontsize=14)
        ax.set_title(f"Velocity Profile - Effect of {vary_param}", 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 10])
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_temperature_profile(self, param_variations: List[Dict], 
                                 vary_param: str, save_name: str = "temperature_profile.png"):
        """
        Plot temperature profile θ(η) for different parameter values
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        eta = np.linspace(0, 10, 200)
        
        for params in param_variations:
            # Solve numerically
            solver = HybridNanofluidSolver(params)
            eta_sol, solution = solver.solve(verbose=False)
            
            if solution is not None:
                theta = solution[3]  # θ
                
                param_value = params[vary_param]
                label = f"{vary_param} = {param_value}"
                ax.plot(eta_sol, theta, linewidth=2, label=label, marker='s', 
                       markersize=4, markevery=20)
        
        ax.set_xlabel(r'$\eta$', fontsize=14)
        ax.set_ylabel(r'$\theta(\eta)$', fontsize=14)
        ax.set_title(f"Temperature Profile - Effect of {vary_param}", 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 10])
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_comparison_ann_vs_numerical(self, params: Dict, 
                                         save_name: str = "ann_vs_numerical.png"):
        """
        Compare ANN predictions with numerical solution
        """
        # Numerical solution
        solver = HybridNanofluidSolver(params)
        eta_num, solution_num = solver.solve(verbose=False)
        
        if solution_num is None:
            print("⚠ Numerical solution failed")
            return
        
        f_num = solution_num[0]
        fp_num = solution_num[1]
        theta_num = solution_num[3]
        
        # ANN prediction
        ann_pred = self.predict(eta_num)
        f_ann = ann_pred['f']
        theta_ann = ann_pred['theta']
        
        # Compute derivatives
        deriv_ann = self.compute_derivatives_numerical(eta_num, f_ann, theta_ann)
        fp_ann = deriv_ann['fp']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot f
        axes[0, 0].plot(eta_num, f_num, 'b-', linewidth=2, label='Numerical')
        axes[0, 0].plot(eta_num, f_ann, 'r--', linewidth=2, label='ANN')
        axes[0, 0].set_xlabel(r'$\eta$', fontsize=12)
        axes[0, 0].set_ylabel(r'$f(\eta)$', fontsize=12)
        axes[0, 0].set_title(r'$f(\eta)$', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot f'
        axes[0, 1].plot(eta_num, fp_num, 'b-', linewidth=2, label='Numerical')
        axes[0, 1].plot(eta_num, fp_ann, 'r--', linewidth=2, label='ANN')
        axes[0, 1].set_xlabel(r'$\eta$', fontsize=12)
        axes[0, 1].set_ylabel(r"$f'(\eta)$", fontsize=12)
        axes[0, 1].set_title(r"$f'(\eta)$", fontsize=13, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot θ
        axes[1, 0].plot(eta_num, theta_num, 'b-', linewidth=2, label='Numerical')
        axes[1, 0].plot(eta_num, theta_ann, 'r--', linewidth=2, label='ANN')
        axes[1, 0].set_xlabel(r'$\eta$', fontsize=12)
        axes[1, 0].set_ylabel(r'$\theta(\eta)$', fontsize=12)
        axes[1, 0].set_title(r'$\theta(\eta)$', fontsize=13, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot errors
        error_f = np.abs(f_num - f_ann)
        error_theta = np.abs(theta_num - theta_ann)
        
        axes[1, 1].plot(eta_num, error_f, 'g-', linewidth=2, label=r'$|f_{num} - f_{ANN}|$')
        axes[1, 1].plot(eta_num, error_theta, 'm-', linewidth=2, label=r'$|\theta_{num} - \theta_{ANN}|$')
        axes[1, 1].set_xlabel(r'$\eta$', fontsize=12)
        axes[1, 1].set_ylabel('Absolute Error', fontsize=12)
        axes[1, 1].set_title('Prediction Errors', fontsize=13, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_skin_friction_variation(self, param_name: str, param_values: List[float],
                                     fixed_params: Dict, save_name: str = "cf_variation.png"):
        """
        Plot skin friction coefficient Cf vs parameter
        """
        Cf_values = []
        
        for param_val in param_values:
            params = fixed_params.copy()
            params[param_name] = param_val
            
            solver = HybridNanofluidSolver(params)
            eta_sol, solution = solver.solve(verbose=False)
            
            if solution is not None:
                eng_quantities = solver.compute_engineering_quantities(solution)
                Cf_values.append(eng_quantities['Cf'])
            else:
                Cf_values.append(np.nan)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.plot(param_values, Cf_values, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel(f'{param_name}', fontsize=14)
        ax.set_ylabel(r'Skin Friction $C_f$', fontsize=14)
        ax.set_title(f'Skin Friction Coefficient vs {param_name}', 
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_nusselt_variation(self, param_name: str, param_values: List[float],
                               fixed_params: Dict, save_name: str = "nu_variation.png"):
        """
        Plot Nusselt number Nu vs parameter
        """
        Nu_values = []
        
        for param_val in param_values:
            params = fixed_params.copy()
            params[param_name] = param_val
            
            solver = HybridNanofluidSolver(params)
            eta_sol, solution = solver.solve(verbose=False)
            
            if solution is not None:
                eng_quantities = solver.compute_engineering_quantities(solution)
                Nu_values.append(eng_quantities['Nu'])
            else:
                Nu_values.append(np.nan)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.plot(param_values, Nu_values, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel(f'{param_name}', fontsize=14)
        ax.set_ylabel(r'Nusselt Number $Nu$', fontsize=14)
        ax.set_title(f'Nusselt Number vs {param_name}', 
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()


def generate_all_plots():
    """Generate all manuscript-style plots"""
    print("=" * 70)
    print("Generating Manuscript-Style Plots")
    print("=" * 70)
    
    # Initialize plotter
    plotter = Plotter(
        model_path="models/checkpoints/best_model.pth",
        scaler_dir="models/checkpoints",
        output_dir="plots"
    )
    
    # Base parameters
    base_params = {
        'M': 1.0,
        'Nr': 0.5,
        'Nh': 0.5,
        'lam': 1.0,
        'beta': 0.1,
        'Pr': 6.2,
        'n': 1.0,
        'Tr': 1.5,
        'As': 1.0,
        'eta_max': 10.0,
        'n_points': 400
    }
    
    # 1. Velocity profiles for varying M
    print("\n1. Generating velocity profiles (varying M)...")
    M_variations = [
        {**base_params, 'M': 0.5},
        {**base_params, 'M': 1.0},
        {**base_params, 'M': 2.0}
    ]
    plotter.plot_velocity_profile(M_variations, 'M', 'velocity_profile_M.png')
    
    # 2. Temperature profiles for varying Nr
    print("2. Generating temperature profiles (varying Nr)...")
    Nr_variations = [
        {**base_params, 'Nr': 0.2},
        {**base_params, 'Nr': 0.5},
        {**base_params, 'Nr': 1.0}
    ]
    plotter.plot_temperature_profile(Nr_variations, 'Nr', 'temperature_profile_Nr.png')
    
    # 3. ANN vs Numerical comparison
    print("3. Generating ANN vs Numerical comparison...")
    plotter.plot_comparison_ann_vs_numerical(base_params, 'ann_vs_numerical.png')
    
    # 4. Skin friction variation
    print("4. Generating skin friction variation (M)...")
    M_values = np.linspace(0.5, 2.5, 10)
    plotter.plot_skin_friction_variation('M', M_values, base_params, 'cf_vs_M.png')
    
    # 5. Nusselt number variation
    print("5. Generating Nusselt number variation (Nr)...")
    Nr_values = np.linspace(0.2, 1.2, 10)
    plotter.plot_nusselt_variation('Nr', Nr_values, base_params, 'nu_vs_Nr.png')
    
    print("\n" + "=" * 70)
    print("✓ All plots generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_plots()
