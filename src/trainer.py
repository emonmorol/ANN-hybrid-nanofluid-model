

import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.lm_optimizer import LevenbergMarquardtOptimizer, SimplifiedLMOptimizer
from src.visualizer import RealTimeVisualizer

class Trainer:

    
    def __init__(self, model: nn.Module, optimizer_type: str = 'lm_custom',
                 device: str = 'cpu', visualize: bool = False):

        self.model = model.to(device)
        self.device = device
        self.optimizer_type = optimizer_type
        self.visualize = visualize
        
        if self.visualize:
            self.visualizer = RealTimeVisualizer()
            self.iteration_count = 0
        
        # Initialize optimizer
        if optimizer_type == 'lm_custom':
            self.optimizer = LevenbergMarquardtOptimizer(model)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_time': []
        }
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:

        mse = torch.mean((predictions - targets) ** 2)
        return mse.item()
    
    def train_epoch_custom(self, X_train: torch.Tensor, y_train: torch.Tensor,
                          batch_size: int = 1000, max_steps: int = 10) -> float:

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
    
    def train_epoch_standard(self, X_train: torch.Tensor, y_train: torch.Tensor,
                           batch_size: int = 64) -> float:

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
                print()
                print(f"    Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Time: {epoch_time:.2f}s")
                print("=" * 70)
            
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

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        print(f"✓ Model loaded from {path}")
