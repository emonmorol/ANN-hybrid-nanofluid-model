
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

class RealTimeVisualizer:

    
    def __init__(self):
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
        
        # Initialize subplots
        self.ax_loss = self.fig.add_subplot(self.gs[0, 0])
        self.ax_f = self.fig.add_subplot(self.gs[0, 1])
        self.ax_theta = self.fig.add_subplot(self.gs[0, 2])
        self.ax_weights = self.fig.add_subplot(self.gs[1, :])
        
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
        # Setup static plot elements
        self.ax_loss.set_title('Training & Validation Loss', fontweight='bold')
        self.ax_loss.set_xlabel('Iteration')
        self.ax_loss.set_ylabel('MSE Loss (Log Scale)')
        self.ax_loss.set_yscale('log')
        self.ax_loss.grid(True, alpha=0.3)
        
        self.ax_f.set_title('f(η) Approximation', fontweight='bold')
        self.ax_f.set_xlabel('η')
        self.ax_f.set_ylabel('f(η)')
        self.ax_f.grid(True, alpha=0.3)
        
        self.ax_theta.set_title('θ(η) Approximation', fontweight='bold')
        self.ax_theta.set_xlabel('η')
        self.ax_theta.set_ylabel('θ(η)')
        self.ax_theta.grid(True, alpha=0.3)
        
        self.ax_weights.set_title('Weight Distribution (First & Last Layers)', fontweight='bold')
        self.ax_weights.set_xlabel('Weight Value')
        self.ax_weights.set_ylabel('Frequency')
        self.ax_weights.grid(True, alpha=0.3)
        
        # Lines for updating
        self.line_train, = self.ax_loss.plot([], [], 'b-', label='Train', alpha=0.7)
        self.line_val, = self.ax_loss.plot([], [], 'r-', label='Val', alpha=0.7)
        self.ax_loss.legend()
        
        self.line_f_true, = self.ax_f.plot([], [], 'k-', label='True', linewidth=2, alpha=0.5)
        self.line_f_pred, = self.ax_f.plot([], [], 'r--', label='Pred', linewidth=2)
        self.ax_f.legend()
        
        self.line_theta_true, = self.ax_theta.plot([], [], 'k-', label='True', linewidth=2, alpha=0.5)
        self.line_theta_pred, = self.ax_theta.plot([], [], 'b--', label='Pred', linewidth=2)
        self.ax_theta.legend()
        
        plt.tight_layout()
        
    def update_plots(self, epoch, train_loss, val_loss, eta, f_true, f_pred, theta_true, theta_pred, model):

        
        # Update Loss
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.line_train.set_data(self.epochs, self.train_losses)
        self.line_val.set_data(self.epochs, self.val_losses)
        
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
        # Update f(eta)
        # Sort by eta for clean plotting
        sort_idx = np.argsort(eta.flatten())
        eta_sorted = eta.flatten()[sort_idx]
        
        self.line_f_true.set_data(eta_sorted, f_true.flatten()[sort_idx])
        self.line_f_pred.set_data(eta_sorted, f_pred.flatten()[sort_idx])
        
        self.ax_f.relim()
        self.ax_f.autoscale_view()
        
        # Update theta(eta)
        self.line_theta_true.set_data(eta_sorted, theta_true.flatten()[sort_idx])
        self.line_theta_pred.set_data(eta_sorted, theta_pred.flatten()[sort_idx])
        
        self.ax_theta.relim()
        self.ax_theta.autoscale_view()
        
        # Update Weights Histogram
        self.ax_weights.clear()
        self.ax_weights.set_title('Weight Distribution (First & Last Layers)', fontweight='bold')
        self.ax_weights.grid(True, alpha=0.3)
        
        # Extract weights
        first_layer_weights = list(model.parameters())[0].detach().cpu().numpy().flatten()
        last_layer_weights = list(model.parameters())[-2].detach().cpu().numpy().flatten() # -2 because bias is last
        
        self.ax_weights.hist(first_layer_weights, bins=50, alpha=0.5, label='First Layer', color='blue')
        self.ax_weights.hist(last_layer_weights, bins=50, alpha=0.5, label='Last Layer', color='green')
        self.ax_weights.legend()
        
        # Draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        plt.ioff()
        plt.show()


def plot_training_history(history: dict, save_path: str = None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to {save_path}")
    
    plt.close()
