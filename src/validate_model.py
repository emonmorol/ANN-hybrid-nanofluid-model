

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import pickle
from typing import Dict, List, Tuple
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.ann import HybridNanofluidANN
from src.solver.ode_solver import HybridNanofluidSolver


class ModelValidator:

    
    def __init__(self, model_path: str, scaler_dir: str):

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
        
        print("✓ Model and scalers loaded for validation")
    
    def predict(self, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # Normalize eta
        eta_normalized = self.scaler_eta.transform(eta.reshape(-1, 1))
        
        # Predict
        with torch.no_grad():
            eta_tensor = torch.FloatTensor(eta_normalized)
            predictions = self.model(eta_tensor).numpy()
        
        # Denormalize
        f = self.scaler_f.inverse_transform(predictions[:, 0:1]).flatten()
        theta = self.scaler_theta.inverse_transform(predictions[:, 1:2]).flatten()
        
        return f, theta
    
    def compute_error_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

        # Absolute errors
        abs_error = np.abs(y_true - y_pred)
        
        # Mean Squared Error (MSE)
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error (MAE)
        mae = np.mean(abs_error)
        
        # Maximum Absolute Error
        max_abs_error = np.max(abs_error)
        
        # Relative Error (%)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_error = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100
            relative_error = relative_error[np.isfinite(relative_error)]
        
        mean_relative_error = np.mean(relative_error) if len(relative_error) > 0 else 0.0
        max_relative_error = np.max(relative_error) if len(relative_error) > 0 else 0.0
        
        # R-squared (Coefficient of Determination)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Correlation coefficient
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Max_Abs_Error': max_abs_error,
            'Mean_Relative_Error_%': mean_relative_error,
            'Max_Relative_Error_%': max_relative_error,
            'R_squared': r_squared,
            'Correlation': correlation
        }
    
    def validate_single_case(self, params: Dict, verbose: bool = True) -> Dict:

        # Solve numerically
        solver = HybridNanofluidSolver(params)
        eta_num, solution_num = solver.solve(verbose=False)
        
        if solution_num is None:
            print("⚠ Numerical solution failed")
            return None
        
        # Extract numerical solution
        f_num = solution_num[0]
        theta_num = solution_num[3]
        
        # ANN prediction
        f_ann, theta_ann = self.predict(eta_num)
        
        # Compute metrics for f
        metrics_f = self.compute_error_metrics(f_num, f_ann)
        
        # Compute metrics for θ
        metrics_theta = self.compute_error_metrics(theta_num, theta_ann)
        
        if verbose:
            print("\n" + "="*70)
            print("VALIDATION RESULTS")
            print("="*70)
            print(f"\nParameters: {params}")
            print("\n--- f(η) Metrics ---")
            for key, value in metrics_f.items():
                print(f"  {key:25s}: {value:.6e}")
            
            print("\n--- θ(η) Metrics ---")
            for key, value in metrics_theta.items():
                print(f"  {key:25s}: {value:.6e}")
            print("="*70)
        
        return {
            'params': params,
            'eta': eta_num,
            'f_numerical': f_num,
            'f_ann': f_ann,
            'theta_numerical': theta_num,
            'theta_ann': theta_ann,
            'metrics_f': metrics_f,
            'metrics_theta': metrics_theta
        }
    
    def validate_multiple_cases(self, test_cases: List[Dict]) -> pd.DataFrame:

        print("\n" + "="*70)
        print("VALIDATING MULTIPLE TEST CASES")
        print("="*70)
        
        results = []
        
        for i, params in enumerate(test_cases):
            print(f"\nCase {i+1}/{len(test_cases)}: {params}")
            
            result = self.validate_single_case(params, verbose=False)
            
            if result is not None:
                # Extract key metrics
                row = {
                    'Case': i+1,
                    'M': params.get('M', '-'),
                    'Nr': params.get('Nr', '-'),
                    'Nh': params.get('Nh', '-'),
                    'lam': params.get('lam', '-'),
                    'f_MSE': result['metrics_f']['MSE'],
                    'f_MAE': result['metrics_f']['MAE'],
                    'f_Max_Error': result['metrics_f']['Max_Abs_Error'],
                    'f_R²': result['metrics_f']['R_squared'],
                    'θ_MSE': result['metrics_theta']['MSE'],
                    'θ_MAE': result['metrics_theta']['MAE'],
                    'θ_Max_Error': result['metrics_theta']['Max_Abs_Error'],
                    'θ_R²': result['metrics_theta']['R_squared']
                }
                results.append(row)
                
                print(f"  f: MSE={row['f_MSE']:.2e}, MAE={row['f_MAE']:.2e}, R²={row['f_R²']:.6f}")
                print(f"  θ: MSE={row['θ_MSE']:.2e}, MAE={row['θ_MAE']:.2e}, R²={row['θ_R²']:.6f}")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))
        
        # Overall statistics
        print("\n" + "="*70)
        print("OVERALL STATISTICS")
        print("="*70)
        print("\nf(η) Statistics:")
        print(f"  Mean MSE:        {summary_df['f_MSE'].mean():.6e}")
        print(f"  Mean MAE:        {summary_df['f_MAE'].mean():.6e}")
        print(f"  Mean Max Error:  {summary_df['f_Max_Error'].mean():.6e}")
        print(f"  Mean R²:         {summary_df['f_R²'].mean():.6f}")
        
        print("\nθ(η) Statistics:")
        print(f"  Mean MSE:        {summary_df['θ_MSE'].mean():.6e}")
        print(f"  Mean MAE:        {summary_df['θ_MAE'].mean():.6e}")
        print(f"  Mean Max Error:  {summary_df['θ_Max_Error'].mean():.6e}")
        print(f"  Mean R²:         {summary_df['θ_R²'].mean():.6f}")
        
        return summary_df
    
    def plot_validation_results(self, result: Dict, save_path: str = None):

        eta = result['eta']
        f_num = result['f_numerical']
        f_ann = result['f_ann']
        theta_num = result['theta_numerical']
        theta_ann = result['theta_ann']
        
        # Create figure with 6 subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. f(η) comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(eta, f_num, 'b-', linewidth=2, label='Numerical')
        ax1.plot(eta, f_ann, 'r--', linewidth=2, label='ANN')
        ax1.set_xlabel(r'$\eta$', fontsize=11)
        ax1.set_ylabel(r'$f(\eta)$', fontsize=11)
        ax1.set_title(r'$f(\eta)$ Comparison', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. θ(η) comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(eta, theta_num, 'b-', linewidth=2, label='Numerical')
        ax2.plot(eta, theta_ann, 'r--', linewidth=2, label='ANN')
        ax2.set_xlabel(r'$\eta$', fontsize=11)
        ax2.set_ylabel(r'$\theta(\eta)$', fontsize=11)
        ax2.set_title(r'$\theta(\eta)$ Comparison', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Absolute errors
        ax3 = fig.add_subplot(gs[0, 2])
        error_f = np.abs(f_num - f_ann)
        error_theta = np.abs(theta_num - theta_ann)
        ax3.semilogy(eta, error_f, 'g-', linewidth=2, label=r'$|f_{num} - f_{ANN}|$')
        ax3.semilogy(eta, error_theta, 'm-', linewidth=2, label=r'$|\theta_{num} - \theta_{ANN}|$')
        ax3.set_xlabel(r'$\eta$', fontsize=11)
        ax3.set_ylabel('Absolute Error (log scale)', fontsize=11)
        ax3.set_title('Absolute Errors', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. f scatter plot
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(f_num, f_ann, alpha=0.6, s=20)
        min_val = min(f_num.min(), f_ann.min())
        max_val = max(f_num.max(), f_ann.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
        ax4.set_xlabel(r'$f_{numerical}$', fontsize=11)
        ax4.set_ylabel(r'$f_{ANN}$', fontsize=11)
        ax4.set_title(r'$f$ Scatter Plot (R²={:.6f})'.format(result['metrics_f']['R_squared']), 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. θ scatter plot
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(theta_num, theta_ann, alpha=0.6, s=20, color='orange')
        min_val = min(theta_num.min(), theta_ann.min())
        max_val = max(theta_num.max(), theta_ann.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
        ax5.set_xlabel(r'$\theta_{numerical}$', fontsize=11)
        ax5.set_ylabel(r'$\theta_{ANN}$', fontsize=11)
        ax5.set_title(r'$\theta$ Scatter Plot (R²={:.6f})'.format(result['metrics_theta']['R_squared']), 
                     fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # 6. Error distribution histogram
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(error_f, bins=30, alpha=0.6, label='f errors', color='green')
        ax6.hist(error_theta, bins=30, alpha=0.6, label='θ errors', color='magenta')
        ax6.set_xlabel('Absolute Error', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. Metrics table for f
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        metrics_f = result['metrics_f']
        metrics_theta = result['metrics_theta']
        
        table_data = [
            ['Metric', 'f(η)', 'θ(η)'],
            ['MSE', f"{metrics_f['MSE']:.6e}", f"{metrics_theta['MSE']:.6e}"],
            ['RMSE', f"{metrics_f['RMSE']:.6e}", f"{metrics_theta['RMSE']:.6e}"],
            ['MAE', f"{metrics_f['MAE']:.6e}", f"{metrics_theta['MAE']:.6e}"],
            ['Max Abs Error', f"{metrics_f['Max_Abs_Error']:.6e}", f"{metrics_theta['Max_Abs_Error']:.6e}"],
            ['Mean Rel Error (%)', f"{metrics_f['Mean_Relative_Error_%']:.4f}", f"{metrics_theta['Mean_Relative_Error_%']:.4f}"],
            ['R²', f"{metrics_f['R_squared']:.8f}", f"{metrics_theta['R_squared']:.8f}"],
            ['Correlation', f"{metrics_f['Correlation']:.8f}", f"{metrics_theta['Correlation']:.8f}"]
        ]
        
        table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.suptitle('Model Validation Results', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Validation plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():

    print("="*70)
    print("ANN MODEL VALIDATION")
    print("="*70)
    
    # Initialize validator
    validator = ModelValidator(
        model_path="models/checkpoints/best_model.pth",
        scaler_dir="models/checkpoints"
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
    
    # Test Case 1: Single case validation
    print("\n" + "="*70)
    print("TEST CASE 1: Single Case Validation")
    print("="*70)
    result = validator.validate_single_case(base_params, verbose=True)
    
    # Plot results
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    validator.plot_validation_results(result, save_path=output_dir / "validation_single_case.png")
    
    # Test Case 2: Multiple cases validation
    print("\n" + "="*70)
    print("TEST CASE 2: Multiple Cases Validation")
    print("="*70)
    
    test_cases = [
        {**base_params, 'M': 0.5},
        {**base_params, 'M': 1.0},
        {**base_params, 'M': 1.5},
        {**base_params, 'Nr': 0.2},
        {**base_params, 'Nr': 0.8},
        {**base_params, 'Nh': 0.3},
        {**base_params, 'Nh': 0.7},
        {**base_params, 'lam': 0.5},
        {**base_params, 'lam': 1.5},
    ]
    
    summary_df = validator.validate_multiple_cases(test_cases)
    
    # Save summary to CSV
    summary_df.to_csv(output_dir / "validation_summary.csv", index=False)
    print(f"\n✓ Validation summary saved to: {output_dir / 'validation_summary.csv'}")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
