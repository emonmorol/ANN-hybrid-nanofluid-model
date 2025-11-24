
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import sys
import joblib
import seaborn as sns

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.ann import HybridNanofluidANN
    from src import config

class RealTimeVisualizer:
    
    def __init__(self):
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(20, 12))
        self.gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])
        
        # Initialize subplots
        self.ax_loss = self.fig.add_subplot(self.gs[0, 0])
        self.ax_network = self.fig.add_subplot(self.gs[0, 1:])  # Network architecture
        self.ax_weights = self.fig.add_subplot(self.gs[1, :])   # Weight distributions
        self.ax_gradients = self.fig.add_subplot(self.gs[2, :]) # Gradient magnitudes
        
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
        # Setup loss plot
        self.ax_loss.set_title('Training & Validation Loss', fontweight='bold', fontsize=12)
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('MSE Loss (Log Scale)')
        self.ax_loss.set_yscale('log')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Setup network architecture plot
        self.ax_network.set_title('Neural Network Architecture (9 Hidden Layers × 30 Neurons)', 
                                  fontweight='bold', fontsize=12)
        self.ax_network.axis('off')
        
        # Setup weights plot
        self.ax_weights.set_title('Weight Distribution by Layer', fontweight='bold', fontsize=12)
        self.ax_weights.set_xlabel('Weight Value')
        self.ax_weights.set_ylabel('Density')
        self.ax_weights.grid(True, alpha=0.3)
        
        # Setup gradients plot
        self.ax_gradients.set_title('Gradient Magnitude by Layer', fontweight='bold', fontsize=12)
        self.ax_gradients.set_xlabel('Layer')
        self.ax_gradients.set_ylabel('Mean Gradient Magnitude (Log Scale)')
        self.ax_gradients.set_yscale('log')
        self.ax_gradients.grid(True, alpha=0.3)
        
        # Lines for updating
        self.line_train, = self.ax_loss.plot([], [], 'b-', label='Train', alpha=0.7, linewidth=2)
        self.line_val, = self.ax_loss.plot([], [], 'r-', label='Val', alpha=0.7, linewidth=2)
        self.ax_loss.legend()
        
        plt.tight_layout()
    
    def update(self, epoch, train_loss, val_loss, model=None):

        # Update loss data
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.line_train.set_data(self.epochs, self.train_losses)
        self.line_val.set_data(self.epochs, self.val_losses)
        
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
        # Visualize network architecture (only update every 10 epochs to save time)
        if model is not None and epoch % 10 == 0:
            self._draw_network_architecture(model)
            self._draw_weight_distributions(model)
            self._draw_gradient_magnitudes(model)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _draw_network_architecture(self, model):

        self.ax_network.clear()
        self.ax_network.set_title('Neural Network Architecture (9 Hidden Layers × 30 Neurons)', 
                                  fontweight='bold', fontsize=12)
        self.ax_network.axis('off')
        
        # Get layer sizes
        layer_sizes = [1]  # Input
        for i, param in enumerate(model.parameters()):
            if i % 2 == 0:  # Only weights, skip biases
                layer_sizes.append(param.shape[0])
        
        # Limit neurons shown per layer for visualization
        max_neurons_shown = 10
        
        # Calculate positions
        n_layers = len(layer_sizes)
        layer_spacing = 1.0 / (n_layers + 1)
        
        # Draw neurons and connections
        for layer_idx in range(n_layers):
            n_neurons = min(layer_sizes[layer_idx], max_neurons_shown)
            actual_neurons = layer_sizes[layer_idx]
            
            x = (layer_idx + 1) * layer_spacing
            neuron_spacing = 0.8 / (n_neurons + 1)
            
            # Draw neurons
            for neuron_idx in range(n_neurons):
                y = 0.1 + (neuron_idx + 1) * neuron_spacing
                
                # Color based on layer
                if layer_idx == 0:
                    color = 'lightgreen'
                    label = 'Input'
                elif layer_idx == n_layers - 1:
                    color = 'lightcoral'
                    label = 'Output'
                else:
                    color = 'lightblue'
                    label = f'H{layer_idx}'
                
                circle = plt.Circle((x, y), 0.015, color=color, ec='black', linewidth=1.5, zorder=4)
                self.ax_network.add_patch(circle)
                
                # Add label for first neuron of each layer
                if neuron_idx == 0:
                    if actual_neurons > max_neurons_shown:
                        label_text = f'{label}\n({actual_neurons})'
                    else:
                        label_text = label
                    self.ax_network.text(x, 0.05, label_text, ha='center', va='top', 
                                        fontsize=8, fontweight='bold')
            
            # Draw connections to next layer
            if layer_idx < n_layers - 1:
                n_neurons_next = min(layer_sizes[layer_idx + 1], max_neurons_shown)
                x_next = (layer_idx + 2) * layer_spacing
                neuron_spacing_next = 0.8 / (n_neurons_next + 1)
                
                # Draw sample connections (not all to avoid clutter)
                for i in range(min(3, n_neurons)):
                    y1 = 0.1 + (i + 1) * neuron_spacing
                    for j in range(min(3, n_neurons_next)):
                        y2 = 0.1 + (j + 1) * neuron_spacing_next
                        self.ax_network.plot([x, x_next], [y1, y2], 'gray', 
                                           alpha=0.2, linewidth=0.5, zorder=1)
        
        self.ax_network.set_xlim(0, 1)
        self.ax_network.set_ylim(0, 1)
    
    def _draw_weight_distributions(self, model):

        self.ax_weights.clear()
        self.ax_weights.set_title('Weight Distribution by Layer', fontweight='bold', fontsize=12)
        self.ax_weights.set_xlabel('Weight Value')
        self.ax_weights.set_ylabel('Density')
        self.ax_weights.grid(True, alpha=0.3)
        
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
        for i, param in enumerate(model.parameters()):
            if i % 2 == 0:  # Only weights, skip biases
                layer_idx = i // 2
                weights = param.detach().cpu().numpy().flatten()
                
                # Only show every other layer to avoid clutter
                if layer_idx % 2 == 0:
                    self.ax_weights.hist(weights, bins=30, alpha=0.5, 
                                        label=f'Layer {layer_idx}', 
                                        color=colors[layer_idx % 10], density=True)
        
        self.ax_weights.legend(fontsize=8, ncol=5)
    
    def _draw_gradient_magnitudes(self, model):

        self.ax_gradients.clear()
        self.ax_gradients.set_title('Gradient Magnitude by Layer', fontweight='bold', fontsize=12)
        self.ax_gradients.set_xlabel('Layer')
        self.ax_gradients.set_ylabel('Mean Gradient Magnitude (Log Scale)')
        self.ax_gradients.set_yscale('log')
        self.ax_gradients.grid(True, alpha=0.3)
        
        layer_names = []
        grad_magnitudes = []
        
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                layer_idx = i // 2
                param_type = 'W' if i % 2 == 0 else 'b'
                layer_names.append(f'L{layer_idx}{param_type}')
                
                grad_mag = torch.mean(torch.abs(param.grad)).item()
                grad_magnitudes.append(grad_mag if grad_mag > 0 else 1e-10)
        
        if grad_magnitudes:
            x_pos = np.arange(len(layer_names))
            self.ax_gradients.bar(x_pos, grad_magnitudes, color='steelblue', alpha=0.7)
            self.ax_gradients.set_xticks(x_pos[::2])  # Show every other label
            self.ax_gradients.set_xticklabels(layer_names[::2], rotation=45, ha='right', fontsize=8)
        
    def close(self):
        plt.ioff()
        plt.show()


def plot_training_history(history: dict, save_path: str = None):
    
    # Set professional style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    colors = sns.color_palette("deep")
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
    
    epochs = range(1, len(history['train_loss']) + 1)
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    
    # Calculate smoothing (Rolling Average)
    window = max(2, len(epochs) // 20)  # Dynamic window size
    train_smooth = pd.Series(train_loss).rolling(window=window, min_periods=1).mean()
    val_smooth = pd.Series(val_loss).rolling(window=window, min_periods=1).mean()
    
    # ===== Plot 1: Loss Curves (Main) =====
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot raw data (faint)
    sns.lineplot(x=epochs, y=train_loss, ax=ax1, color=colors[0], alpha=0.3, linewidth=1, label='_nolegend_')
    sns.lineplot(x=epochs, y=val_loss, ax=ax1, color=colors[3], alpha=0.3, linewidth=1, label='_nolegend_')
    
    # Plot smoothed data (bold)
    sns.lineplot(x=epochs, y=train_smooth, ax=ax1, color=colors[0], linewidth=3, label=f'Training (Smoothed w={window})')
    sns.lineplot(x=epochs, y=val_smooth, ax=ax1, color=colors[3], linewidth=3, label=f'Validation (Smoothed w={window})')
    
    # Mark best epoch
    best_idx = np.argmin(val_loss)
    best_epoch = epochs[best_idx]
    best_val_loss = val_loss[best_idx]
    
    ax1.scatter([best_epoch], [best_val_loss], color='gold', s=250, zorder=5, 
                edgecolors='black', linewidth=2, marker='*', label=f'Best Model (Ep {best_epoch})')
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('MSE Loss (Log Scale)', fontweight='bold')
    ax1.set_title('Training & Validation Convergence', fontweight='bold', pad=15)
    ax1.legend(frameon=True, framealpha=0.9, loc='upper right')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.5, linestyle='--', which='both')
    
    # ===== Plot 2: Statistics Panel =====
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # Stats calculations
    total_time = sum(history['epoch_time'])
    avg_time = np.mean(history['epoch_time'])
    improvement = (train_loss[0] - train_loss[-1]) / train_loss[0] * 100
    
    stats_text = (
        f"📊 TRAINING SUMMARY\n{'='*30}\n\n"
        f"⏱️ Duration:\n"
        f"  • Total Time:   {total_time:.1f}s\n"
        f"  • Avg/Epoch:    {avg_time:.3f}s\n\n"
        f"📉 Performance:\n"
        f"  • Best Epoch:   {best_epoch}\n"
        f"  • Best Val Loss:{best_val_loss:.2e}\n"
        f"  • Final Train:  {train_loss[-1]:.2e}\n"
        f"  • Improvement:  {improvement:.1f}%\n\n"
        f"🔍 Status:\n"
        f"  • Converged:    {'Yes' if train_loss[-1] < 1e-3 else 'No'}\n"
        f"  • Overfitting:  {'No' if val_loss[-1] < train_loss[-1]*1.5 else 'Possible'}"
    )
    
    ax2.text(0.5, 0.5, stats_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', edgecolor='gray', alpha=0.5))
    
    # ===== Plot 3: Train vs Val Correlation =====
    ax3 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(x=train_loss, y=val_loss, ax=ax3, hue=epochs, palette='viridis', 
                    size=epochs, sizes=(20, 100), alpha=0.7, legend=False)
    
    # Perfect fit line
    min_l = min(min(train_loss), min(val_loss))
    max_l = max(max(train_loss), max(val_loss))
    ax3.plot([min_l, max_l], [min_l, max_l], 'k--', alpha=0.5, label='Ideal (Train=Val)')
    
    ax3.set_xlabel('Training Loss', fontweight='bold')
    ax3.set_ylabel('Validation Loss', fontweight='bold')
    ax3.set_title('Generalization Gap', fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # ===== Plot 4: Loss Distribution (KDE) =====
    ax4 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(data=train_loss, ax=ax4, fill=True, color=colors[0], alpha=0.3, label='Train')
    sns.kdeplot(data=val_loss, ax=ax4, fill=True, color=colors[3], alpha=0.3, label='Val')
    
    ax4.set_xlabel('Loss Magnitude', fontweight='bold')
    ax4.set_title('Loss Distribution Density', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ===== Plot 5: Learning Speed (First Derivative) =====
    ax5 = fig.add_subplot(gs[1, 2])
    loss_diff = np.diff(train_smooth)
    sns.lineplot(x=epochs[1:], y=loss_diff, ax=ax5, color='purple', linewidth=1.5)
    ax5.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    ax5.set_xlabel('Epoch', fontweight='bold')
    ax5.set_ylabel('Δ Loss / Epoch', fontweight='bold')
    ax5.set_title('Convergence Rate (Slope)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle('ANN Training Dynamics & Performance Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Enhanced training history plot saved to {save_path}")
    
    plt.close()

def compute_derivatives(eta, f, theta):

    # First derivatives
    fp = np.gradient(f, eta)
    thetap = np.gradient(theta, eta)
    
    # Second derivative of f
    fpp = np.gradient(fp, eta)
    
    return fp, fpp, thetap

def plot_predictions(model_path, test_data_path, scalers_dir, save_path=None):

    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path)
    model = HybridNanofluidANN(
        input_dim=config.MODEL_PARAMS['input_dim'],
        hidden_dim=config.MODEL_PARAMS['hidden_dim'],
        num_hidden_layers=config.MODEL_PARAMS['num_hidden_layers'],
        output_dim=config.MODEL_PARAMS['output_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scalers
    print("Loading scalers...")
    scaler_eta = joblib.load(scalers_dir / "scaler_eta.pkl")
    scaler_f = joblib.load(scalers_dir / "scaler_f.pkl")
    scaler_theta = joblib.load(scalers_dir / "scaler_theta.pkl")
    
    # Load test data
    print("Loading test data...")
    df = pd.read_csv(test_data_path)
    
    # Select a representative test case (first case)
    case_id = df['case_id'].unique()[0]
    case_data = df[df['case_id'] == case_id].sort_values('eta')
    
    # Extract data
    eta = case_data['eta'].values
    f_true = case_data['f'].values
    theta_true = case_data['theta'].values
    
    # Get parameters for this case
    params = case_data.iloc[0]
    
    # Make predictions
    print("Making predictions...")
    with torch.no_grad():
        eta_scaled = scaler_eta.transform(eta.reshape(-1, 1))
        eta_tensor = torch.FloatTensor(eta_scaled)
        
        predictions = model(eta_tensor).numpy()
        
        # Inverse transform
        f_pred = scaler_f.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
        theta_pred = scaler_theta.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()
    
    # Compute derivatives for predictions
    # We need f, f'', f''', theta, theta'
    # First derivatives
    fp_pred = np.gradient(f_pred, eta)
    thetap_pred = np.gradient(theta_pred, eta)
    
    # Second derivative
    fpp_pred = np.gradient(fp_pred, eta)
    
    # Third derivative
    fppp_pred = np.gradient(fpp_pred, eta)
    
    # Calculate errors (using numerical derivatives of true data for comparison if available, 
    # otherwise just showing predictions)
    # Note: True derivatives might be in the dataset, let's check
    has_derivs = 'fpp' in case_data.columns
    
    if has_derivs:
        fpp_true = case_data['fpp'].values
        thetap_true = case_data['thetap'].values
        # Calculate f''' numerically from f'' true
        fppp_true = np.gradient(fpp_true, eta)
    
    # Create figure with professional styling
    print("Creating visualization...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    
    # Custom color palette
    colors = sns.color_palette("deep")
    color_num = colors[0]  # Blue-ish
    color_ann = colors[3]  # Red-ish
    color_err = 'gray'
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Helper function for consistent plotting
    def plot_panel(ax, x, y_true, y_pred, title, ylabel, show_legend=True):
        # Plot Numerical (Line)
        sns.lineplot(x=x, y=y_true, ax=ax, color=color_num, linewidth=2.5, 
                    label='Numerical (bvp)' if show_legend else None, alpha=0.9)
        # Plot ANN (Dashed)
        sns.lineplot(x=x, y=y_pred, ax=ax, color=color_ann, linewidth=2.5, linestyle='--',
                    label='ANN Prediction' if show_legend else None, alpha=0.9)
        # Error Band
        ax.fill_between(x, y_true, y_pred, color=color_err, alpha=0.2, label='Error' if show_legend else None)
        
        ax.set_xlabel(r'$\eta$', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=10)
        if show_legend:
            ax.legend(frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.5, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Plot 1: f vs η
    ax1 = fig.add_subplot(gs[0, 0])
    plot_panel(ax1, eta, f_true, f_pred, r'Stream Function $f(\eta)$', r'$f(\eta)$')
    
    # Plot 2: f'' vs η
    ax2 = fig.add_subplot(gs[0, 1])
    if has_derivs:
        plot_panel(ax2, eta, fpp_true, fpp_pred, r'Shear Stress $f^{\prime\prime}(\eta)$', r'$f^{\prime\prime}(\eta)$', show_legend=False)
    else:
        sns.lineplot(x=eta, y=fpp_pred, ax=ax2, color=color_ann, linewidth=2.5, linestyle='--')
        ax2.set_title(r'Shear Stress $f^{\prime\prime}(\eta)$')
    
    # Plot 3: f''' vs η
    ax3 = fig.add_subplot(gs[0, 2])
    if has_derivs:
        plot_panel(ax3, eta, fppp_true, fppp_pred, r'Shear Gradient $f^{\prime\prime\prime}(\eta)$', r'$f^{\prime\prime\prime}(\eta)$', show_legend=False)
    else:
        sns.lineplot(x=eta, y=fppp_pred, ax=ax3, color=color_ann, linewidth=2.5, linestyle='--')
        ax3.set_title(r'Shear Gradient $f^{\prime\prime\prime}(\eta)$')
    
    # Plot 4: θ vs η
    ax4 = fig.add_subplot(gs[1, 0])
    plot_panel(ax4, eta, theta_true, theta_pred, r'Temperature $\theta(\eta)$', r'$\theta(\eta)$', show_legend=False)
    
    # Plot 5: θ' vs η
    ax5 = fig.add_subplot(gs[1, 1])
    if has_derivs:
        plot_panel(ax5, eta, thetap_true, thetap_pred, r'Heat Flux $\theta^{\prime}(\eta)$', r'$\theta^{\prime}(\eta)$', show_legend=False)
    else:
        sns.lineplot(x=eta, y=thetap_pred, ax=ax5, color=color_ann, linewidth=2.5, linestyle='--')
        ax5.set_title(r'Heat Flux $\theta^{\prime}(\eta)$')
    
    # Plot 6: Training vs Validation vs Testing Loss
    ax6 = fig.add_subplot(gs[1, 2])
    history = checkpoint.get('history', {})
    
    if history:
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Use seaborn for loss plot
        sns.lineplot(x=epochs, y=history['train_loss'], ax=ax6, color=colors[0], linewidth=2, label='Training')
        sns.lineplot(x=epochs, y=history['val_loss'], ax=ax6, color=colors[3], linewidth=2, label='Validation')
        
        if 'test_loss' in history:
            if isinstance(history['test_loss'], list) and len(history['test_loss']) == len(epochs):
                sns.lineplot(x=epochs, y=history['test_loss'], ax=ax6, color=colors[2], linewidth=2, label='Testing')
            elif isinstance(history['test_loss'], (float, int)):
                 ax6.axhline(y=history['test_loss'], color=colors[2], linestyle='--', linewidth=2, label=f'Test Loss: {history["test_loss"]:.2e}')
        
        ax6.set_xlabel('Epoch', fontweight='bold')
        ax6.set_ylabel('MSE Loss (Log Scale)', fontweight='bold')
        ax6.set_title('Model Convergence', fontweight='bold', pad=10)
        ax6.set_yscale('log')
        ax6.legend(frameon=True)
        ax6.grid(True, alpha=0.5, linestyle='--')
    else:
        ax6.text(0.5, 0.5, 'No History', ha='center', va='center')
        ax6.axis('off')

    # Overall title
    plt.suptitle(f'ANN vs Numerical Solution (M={params["M"]:.1f}, Nr={params["Nr"]:.1f})', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Prediction plots saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {}

if __name__ == "__main__":
    # Paths
    model_path = config.BEST_MODEL_PATH
    test_data_path = config.TEST_DATA_PATH
    scalers_dir = config.SCALER_DIR
    save_path = config.PLOT_DIR / "model_predictions.png"
    
    # Create visualization
    metrics = plot_predictions(model_path, test_data_path, scalers_dir, save_path)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
