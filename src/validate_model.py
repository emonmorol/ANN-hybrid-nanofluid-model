
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import pickle
from typing import Dict, List, Tuple
from scipy import stats
import sys
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.ann import HybridNanofluidANN
from src.solver.ode_solver import HybridNanofluidSolver
from src import config


class ModelValidator:
    
    def __init__(self, model_path: str, scaler_dir: str):
        # Load model
        self.model = HybridNanofluidANN(
            input_dim=config.MODEL_PARAMS['input_dim'],
            hidden_dim=config.MODEL_PARAMS['hidden_dim'],
            num_hidden_layers=config.MODEL_PARAMS['num_hidden_layers'],
            output_dim=config.MODEL_PARAMS['output_dim']
        )
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
        
        # Set professional style
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        colors = sns.color_palette("deep")
        
        # Create figure with 6 subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. f(η) comparison
        ax1 = fig.add_subplot(gs[0, 0])
        sns.lineplot(x=eta, y=f_num, ax=ax1, color=colors[0], label='Numerical', linewidth=2.5)
        sns.lineplot(x=eta, y=f_ann, ax=ax1, color=colors[3], label='ANN', linestyle='--', linewidth=2.5)
        ax1.set_xlabel(r'$\eta$', fontweight='bold')
        ax1.set_ylabel(r'$f(\eta)$', fontweight='bold')
        ax1.set_title(r'$f(\eta)$ Comparison', fontweight='bold')
        ax1.legend()
        
        # 2. θ(η) comparison
        ax2 = fig.add_subplot(gs[0, 1])
        sns.lineplot(x=eta, y=theta_num, ax=ax2, color=colors[0], label='Numerical', linewidth=2.5)
        sns.lineplot(x=eta, y=theta_ann, ax=ax2, color=colors[3], label='ANN', linestyle='--', linewidth=2.5)
        ax2.set_xlabel(r'$\eta$', fontweight='bold')
        ax2.set_ylabel(r'$\theta(\eta)$', fontweight='bold')
        ax2.set_title(r'$\theta(\eta)$ Comparison', fontweight='bold')
        ax2.legend()
        
        # 3. Absolute errors
        ax3 = fig.add_subplot(gs[0, 2])
        error_f = np.abs(f_num - f_ann)
        error_theta = np.abs(theta_num - theta_ann)
        sns.lineplot(x=eta, y=error_f, ax=ax3, color='green', label=r'$|Error_f|$', linewidth=1.5)
        sns.lineplot(x=eta, y=error_theta, ax=ax3, color='orange', label=r'$|Error_\theta|$', linewidth=1.5)
        ax3.set_yscale('log')
        ax3.set_xlabel(r'$\eta$', fontweight='bold')
        ax3.set_ylabel('Absolute Error (Log)', fontweight='bold')
        ax3.set_title('Error Analysis', fontweight='bold')
        ax3.legend()
        
        # 4. f scatter plot
        ax4 = fig.add_subplot(gs[1, 0])
        sns.scatterplot(x=f_num, y=f_ann, ax=ax4, color=colors[0], alpha=0.6, s=30)
        # Perfect fit line
        min_val, max_val = min(f_num.min(), f_ann.min()), max(f_num.max(), f_ann.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
        ax4.set_xlabel(r'$f_{numerical}$', fontweight='bold')
        ax4.set_ylabel(r'$f_{ANN}$', fontweight='bold')
        ax4.set_title(r'$f$ Correlation ($R^2$={:.6f})'.format(result['metrics_f']['R_squared']), fontweight='bold')
        
        # 5. θ scatter plot
        ax5 = fig.add_subplot(gs[1, 1])
        sns.scatterplot(x=theta_num, y=theta_ann, ax=ax5, color=colors[1], alpha=0.6, s=30)
        min_val, max_val = min(theta_num.min(), theta_ann.min()), max(theta_num.max(), theta_ann.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
        ax5.set_xlabel(r'$\theta_{numerical}$', fontweight='bold')
        ax5.set_ylabel(r'$\theta_{ANN}$', fontweight='bold')
        ax5.set_title(r'$\theta$ Correlation ($R^2$={:.6f})'.format(result['metrics_theta']['R_squared']), fontweight='bold')
        
        # 6. Error distribution
        ax6 = fig.add_subplot(gs[1, 2])
        sns.kdeplot(data=error_f, ax=ax6, fill=True, color='green', label='f error', alpha=0.3)
        sns.kdeplot(data=error_theta, ax=ax6, fill=True, color='orange', label='θ error', alpha=0.3)
        ax6.set_xlabel('Absolute Error', fontweight='bold')
        ax6.set_title('Error Distribution (KDE)', fontweight='bold')
        ax6.legend()
        
        # 7. Metrics table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        metrics_f = result['metrics_f']
        metrics_theta = result['metrics_theta']
        
        table_data = [
            ['Metric', 'f(η)', 'θ(η)'],
            ['MSE', f"{metrics_f['MSE']:.2e}", f"{metrics_theta['MSE']:.2e}"],
            ['MAE', f"{metrics_f['MAE']:.2e}", f"{metrics_theta['MAE']:.2e}"],
            ['Max Error', f"{metrics_f['Max_Abs_Error']:.2e}", f"{metrics_theta['Max_Abs_Error']:.2e}"],
            ['R²', f"{metrics_f['R_squared']:.6f}", f"{metrics_theta['R_squared']:.6f}"]
        ]
        
        table = ax7.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.2, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style table
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(colors[0])
                cell.set_text_props(color='white', weight='bold')
            elif row % 2 == 0:
                cell.set_facecolor('#f2f2f2')
        
        plt.suptitle('Model Validation Results', fontsize=20, fontweight='bold', y=0.98)
        
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
        model_path=config.BEST_MODEL_PATH,
        scaler_dir=config.SCALER_DIR
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
    output_dir = config.PLOT_DIR
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
