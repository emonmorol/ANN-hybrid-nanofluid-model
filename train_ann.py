"""
Training Pipeline for ANN Hybrid Nanofluid Model
Implements training, validation, and testing with LM optimizer
"""

import torch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm
import time

from models.ann import HybridNanofluidANN, ANNWithDerivatives
from models.lm_optimizer import LevenbergMarquardtOptimizer, SimplifiedLMOptimizer


class DataLoader:
    """Load and preprocess training data"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.scaler_eta = MinMaxScaler()
        self.scaler_f = MinMaxScaler()
        self.scaler_theta = MinMaxScaler()
        
    def load_data(self, normalize: bool = True) -> dict:
        """
        Load and preprocess data from CSV
        
        Returns:
        --------
        dict with train/val/test splits
        """
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        print(f"  Total samples: {len(df)}")
        print(f"  Unique cases: {df['case_id'].nunique()}")
        
        # Extract features and targets
        eta = df['eta'].values.reshape(-1, 1)
        f = df['f'].values.reshape(-1, 1)
        theta = df['theta'].values.reshape(-1, 1)
        
        # Normalize η to [0, 1]
        if normalize:
            eta_normalized = self.scaler_eta.fit_transform(eta)
            f_normalized = self.scaler_f.fit_transform(f)
            theta_normalized = self.scaler_theta.fit_transform(theta)
        else:
            eta_normalized = eta
            f_normalized = f
            theta_normalized = theta
        
        # Combine targets
        targets = np.hstack([f_normalized, theta_normalized])
        
        # Split data: 80% train, 10% val, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            eta_normalized, targets, test_size=0.2, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        
        return {
            'X_train': torch.FloatTensor(X_train),
            'y_train': torch.FloatTensor(y_train),
            'X_val': torch.FloatTensor(X_val),
            'y_val': torch.FloatTensor(y_val),
            'X_test': torch.FloatTensor(X_test),
            'y_test': torch.FloatTensor(y_test),
            'eta_raw': eta,
            'f_raw': f,
            'theta_raw': theta
        }
    
    def save_scalers(self, output_dir: str):
        """Save scalers for later use"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        with open(output_dir / 'scaler_eta.pkl', 'wb') as f:
            pickle.dump(self.scaler_eta, f)
        with open(output_dir / 'scaler_f.pkl', 'wb') as f:
            pickle.dump(self.scaler_f, f)
        with open(output_dir / 'scaler_theta.pkl', 'wb') as f:
            pickle.dump(self.scaler_theta, f)
        
        print(f"✓ Scalers saved to {output_dir}")


class Trainer:
    """Training manager for ANN model"""
    
    def __init__(self, model: nn.Module, optimizer_type: str = 'lm_custom',
                 device: str = 'cpu'):
        """
        Initialize trainer
        
        Parameters:
        -----------
        model : nn.Module
            ANN model
        optimizer_type : str
            'lm_custom' or 'lm_scipy'
        device : str
            'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer_type = optimizer_type
        
        # Initialize optimizer
        if optimizer_type == 'lm_custom':
            self.optimizer = LevenbergMarquardtOptimizer(model)
        elif optimizer_type == 'lm_scipy':
            self.optimizer = SimplifiedLMOptimizer(model)
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        elif optimizer_type == 'lbfgs':
            # L-BFGS with strong default parameters for convergence
            self.optimizer = optim.LBFGS(model.parameters(), 
                                        lr=1, 
                                        max_iter=20, 
                                        history_size=100,
                                        line_search_fn='strong_wolfe')
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_time': []
        }
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute MSE loss"""
        mse = torch.mean((predictions - targets) ** 2)
        return mse.item()
    
    def train_epoch_custom(self, X_train: torch.Tensor, y_train: torch.Tensor,
                          batch_size: int = 1000, max_steps: int = 10) -> float:
        """
        Train one epoch using custom LM optimizer
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Process in batches
        n_samples = len(X_train)
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X_train[batch_indices].to(self.device)
            y_batch = y_train[batch_indices].to(self.device)
            
            # Perform LM steps
            for _ in range(max_steps):
                loss, success = self.optimizer.step(X_batch, y_batch, use_efficient=True)
                if success:
                    break
            
            total_loss += loss
            n_batches += 1
        
        return total_loss / n_batches
    
    def train_epoch_scipy(self, X_train: torch.Tensor, y_train: torch.Tensor,
                         max_nfev: int = 50) -> float:
        """
        Train one epoch using scipy LM optimizer
        """
        self.model.train()
        
        result = self.optimizer.optimize(X_train, y_train, max_nfev=max_nfev, verbose=0)
        
        # Compute final loss
        with torch.no_grad():
            predictions = self.model(X_train)
            loss = self.compute_loss(predictions, y_train)
        
        return loss

    def train_epoch_lbfgs(self, X_train: torch.Tensor, y_train: torch.Tensor) -> float:
        """
        Train one epoch using L-BFGS optimizer (Full Batch)
        """
        self.model.train()
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        # Define closure for L-BFGS (re-evaluates loss)
        def closure():
            self.optimizer.zero_grad()
            predictions = self.model(X_train)
            loss = torch.nn.MSELoss()(predictions, y_train)
            loss.backward()
            
            # Optional: Print progress inside the step for visibility
            print(f"\r    L-BFGS Loss: {loss.item():.6f}", end="")
            return loss

        # Perform optimization step
        self.optimizer.step(closure)
        
        # Return final loss
        with torch.no_grad():
            pred = self.model(X_train)
            loss = self.compute_loss(pred, y_train)
            
        return loss

    def train_epoch_standard(self, X_train: torch.Tensor, y_train: torch.Tensor,
                           batch_size: int = 64) -> float:
        """
        Train one epoch using standard optimizer (Adam, SGD, etc.)
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Process in batches
        n_samples = len(X_train)
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X_train[batch_indices].to(self.device)
            y_batch = y_train[batch_indices].to(self.device)
            
            # Standard optimization step
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = torch.nn.MSELoss()(predictions, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """Validate model"""
        self.model.eval()
        
        with torch.no_grad():
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            predictions = self.model(X_val)
            loss = self.compute_loss(predictions, y_val)
        
        return loss
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
              X_val: torch.Tensor, y_val: torch.Tensor,
              epochs: int = 100, batch_size: int = 1000,
              early_stopping_patience: int = 20) -> dict:
        """
        Full training loop
        
        Returns:
        --------
        history : dict
            Training history
        """
        print(f"\nTraining with {self.optimizer_type} optimizer")
        print("=" * 70)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            if self.optimizer_type == 'lm_custom':
                train_loss = self.train_epoch_custom(X_train, y_train, 
                                                     batch_size=batch_size, max_steps=5)
            elif self.optimizer_type == 'lm_scipy':
                train_loss = self.train_epoch_scipy(X_train, y_train, max_nfev=50)
            elif self.optimizer_type == 'lbfgs':
                # L-BFGS uses full batch
                train_loss = self.train_epoch_lbfgs(X_train, y_train)
            else:
                train_loss = self.train_epoch_standard(X_train, y_train, batch_size=batch_size)
            
            # Validate
            val_loss = self.validate(X_val, y_val)
            
            epoch_time = time.time() - start_time
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_time'].append(epoch_time)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        self.model.load_state_dict(self.best_model_state)
        
        print(f"\n✓ Training complete!")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        
        return self.history
    
    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> dict:
        """
        Evaluate model on test set
        
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            predictions = self.model(X_test)
        
        # Compute metrics
        mse = torch.mean((predictions - y_test) ** 2).item()
        mae = torch.mean(torch.abs(predictions - y_test)).item()
        max_error = torch.max(torch.abs(predictions - y_test)).item()
        
        # Per-output metrics
        f_pred = predictions[:, 0]
        theta_pred = predictions[:, 1]
        f_true = y_test[:, 0]
        theta_true = y_test[:, 1]
        
        mse_f = torch.mean((f_pred - f_true) ** 2).item()
        mse_theta = torch.mean((theta_pred - theta_true) ** 2).item()
        
        mae_f = torch.mean(torch.abs(f_pred - f_true)).item()
        mae_theta = torch.mean(torch.abs(theta_pred - theta_true)).item()
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'mse_f': mse_f,
            'mse_theta': mse_theta,
            'mae_f': mae_f,
            'mae_theta': mae_theta
        }
        
        return metrics
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        print(f"✓ Model loaded from {path}")


def plot_training_history(history: dict, save_path: str = None):
    """Plot training and validation loss"""
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


def main():
    """Main training execution"""
    print("=" * 70)
    print("ANN Hybrid Nanofluid - Training Pipeline")
    print("=" * 70)
    
    # Configuration
    DATA_PATH = "data/training_data.csv"
    MODEL_DIR = Path("models/checkpoints")
    PLOT_DIR = Path("plots")
    
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Load data
    data_loader = DataLoader(DATA_PATH)
    data = data_loader.load_data(normalize=True)
    data_loader.save_scalers(MODEL_DIR)
    
    # Create model
    print("\nInitializing model...")
    model = HybridNanofluidANN(
        input_dim=1,
        hidden_dim=30,
        num_hidden_layers=9,
        output_dim=2
    )
    print(model.get_architecture_summary())
    
    # Create trainer
    # Create trainer
    # Use 'lbfgs' for high-precision convergence
    print("Using L-BFGS optimizer...")
    trainer = Trainer(model, optimizer_type='lbfgs', device='cpu')
    
    # Train model
    # L-BFGS prefers full batch training
    full_batch_size = len(data['X_train'])
    print(f"Training with full batch size: {full_batch_size}")
    
    history = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        epochs=100,
        batch_size=full_batch_size,
        early_stopping_patience=20
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(data['X_test'], data['y_test'])
    
    print("\nTest Set Metrics:")
    print(f"  MSE (overall): {metrics['mse']:.6e}")
    print(f"  MAE (overall): {metrics['mae']:.6e}")
    print(f"  Max Error: {metrics['max_error']:.6e}")
    print(f"  MSE (f): {metrics['mse_f']:.6e}")
    print(f"  MSE (θ): {metrics['mse_theta']:.6e}")
    print(f"  MAE (f): {metrics['mae_f']:.6e}")
    print(f"  MAE (θ): {metrics['mae_theta']:.6e}")
    
    # Save model
    trainer.save_model(MODEL_DIR / "best_model.pth")
    
    # Plot training history
    plot_training_history(history, save_path=PLOT_DIR / "training_history.png")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
