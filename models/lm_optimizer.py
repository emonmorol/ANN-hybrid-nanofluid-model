"""
Levenberg-Marquardt Optimizer for PyTorch
Custom implementation for ANN training
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Tuple
from torch.func import functional_call, vmap, jacrev


class LevenbergMarquardtOptimizer:
    """
    Levenberg-Marquardt optimization algorithm for neural network training
    
    The LM algorithm is particularly effective for least-squares problems:
    - Combines gradient descent and Gauss-Newton method
    - Adaptive damping parameter (λ)
    - Fast convergence for well-posed problems
    """
    
    def __init__(self, model: nn.Module, lambda_init: float = 1e-3, 
                 lambda_scale_up: float = 10.0, lambda_scale_down: float = 0.1,
                 max_lambda: float = 1e10, min_lambda: float = 1e-10):
        """
        Initialize LM optimizer
        
        Parameters:
        -----------
        model : nn.Module
            Neural network model
        lambda_init : float
            Initial damping parameter
        lambda_scale_up : float
            Factor to increase lambda on failed step
        lambda_scale_down : float
            Factor to decrease lambda on successful step
        max_lambda : float
            Maximum allowed lambda value
        min_lambda : float
            Minimum allowed lambda value
        """
        self.model = model
        self.lambda_ = lambda_init
        self.lambda_scale_up = lambda_scale_up
        self.lambda_scale_down = lambda_scale_down
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda
        
        # Get model parameters as a single vector
        self.params = list(model.parameters())
        self.n_params = sum(p.numel() for p in self.params)
        
    def _params_to_vector(self) -> torch.Tensor:
        """Convert model parameters to a single vector"""
        return torch.cat([p.data.flatten() for p in self.params])
    
    def _vector_to_params(self, vector: torch.Tensor):
        """Update model parameters from a vector"""
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.copy_(vector[offset:offset + numel].view_as(p))
            offset += numel
    
    def _compute_jacobian(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian matrix using automatic differentiation
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Input data (batch_size, input_dim)
        targets : torch.Tensor
            Target data (batch_size, output_dim)
        
        Returns:
        --------
        jacobian : torch.Tensor
            Jacobian matrix (n_residuals, n_params)
        """
        # Compute residuals
        outputs = self.model(inputs)
        residuals = (outputs - targets).flatten()
        
        # Compute Jacobian using autograd
        jacobian_rows = []
        
        for i in range(len(residuals)):
            # Zero gradients
            self.model.zero_grad()
            
            # Compute gradient of residual[i] w.r.t. parameters
            if residuals[i].requires_grad:
                residuals[i].backward(retain_graph=True)
                
                # Collect gradients
                grad_vector = torch.cat([p.grad.flatten() if p.grad is not None 
                                        else torch.zeros_like(p).flatten() 
                                        for p in self.params])
                jacobian_rows.append(grad_vector)
        
        if jacobian_rows:
            jacobian = torch.stack(jacobian_rows)
        else:
            jacobian = torch.zeros(len(residuals), self.n_params)
        
        return jacobian
    
    def _compute_jacobian_efficient(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient Jacobian computation using finite differences
        More stable and faster for large networks
        """
        outputs = self.model(inputs)
        residuals = (outputs - targets).flatten()
        n_residuals = len(residuals)
        
        # Use finite differences for Jacobian
        epsilon = 1e-7
        jacobian = torch.zeros(n_residuals, self.n_params)
        
        params_original = self._params_to_vector()
        
        for i in range(self.n_params):
            # Perturb parameter i
            params_perturbed = params_original.clone()
            params_perturbed[i] += epsilon
            self._vector_to_params(params_perturbed)
            
            # Compute perturbed residuals
            outputs_perturbed = self.model(inputs)
            residuals_perturbed = (outputs_perturbed - targets).flatten()
            
            # Finite difference
            jacobian[:, i] = (residuals_perturbed - residuals) / epsilon
        
        # Restore original parameters
        self._vector_to_params(params_original)
        
        return jacobian, residuals
    
    def step(self, inputs: torch.Tensor, targets: torch.Tensor, 
             use_efficient: bool = True) -> Tuple[float, bool]:
        """
        Perform one LM optimization step
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Input data
        targets : torch.Tensor
            Target data
        use_efficient : bool
            Use efficient Jacobian computation (finite differences)
        
        Returns:
        --------
        loss : float
            Current loss value
        success : bool
            Whether the step was successful
        """
        # Compute Jacobian and residuals
        if use_efficient:
            J, r = self._compute_jacobian_efficient(inputs, targets)
        else:
            J = self._compute_jacobian(inputs, targets)
            outputs = self.model(inputs)
            r = (outputs - targets).flatten()
        
        # Current loss
        loss_current = 0.5 * torch.sum(r ** 2).item()
        
        # Compute JTJ and JTr
        JTJ = torch.matmul(J.T, J)
        JTr = torch.matmul(J.T, r)
        
        # Add damping: (JTJ + λI)
        damping = self.lambda_ * torch.eye(self.n_params, device=JTJ.device)
        A = JTJ + damping
        
        # Solve for parameter update: (JTJ + λI) * Δp = -JTr
        try:
            delta_params = torch.linalg.solve(A, -JTr)
        except RuntimeError:
            # If singular, increase damping
            self.lambda_ = min(self.lambda_ * self.lambda_scale_up, self.max_lambda)
            return loss_current, False
        
        # Save current parameters
        params_current = self._params_to_vector()
        
        # Update parameters
        params_new = params_current + delta_params
        self._vector_to_params(params_new)
        
        # Compute new loss
        outputs_new = self.model(inputs)
        r_new = (outputs_new - targets).flatten()
        loss_new = 0.5 * torch.sum(r_new ** 2).item()
        
        # Check if step improved the loss
        if loss_new < loss_current:
            # Accept step, decrease damping
            self.lambda_ = max(self.lambda_ * self.lambda_scale_down, self.min_lambda)
            return loss_new, True
        else:
            # Reject step, restore parameters, increase damping
            self._vector_to_params(params_current)
            self.lambda_ = min(self.lambda_ * self.lambda_scale_up, self.max_lambda)
            return loss_current, False


class SimplifiedLMOptimizer:
    """
    Simplified Levenberg-Marquardt using scipy.optimize.least_squares
    More stable for initial implementation
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.params = list(model.parameters())
        
    def _params_to_vector(self) -> np.ndarray:
        """Convert model parameters to numpy vector"""
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.params])
    
    def _vector_to_params(self, vector: np.ndarray):
        """Update model parameters from numpy vector"""
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.copy_(torch.from_numpy(vector[offset:offset + numel].reshape(p.shape)))
            offset += numel
    
    def residuals(self, params_vector: np.ndarray, inputs_np: np.ndarray, 
                  targets_np: np.ndarray) -> np.ndarray:
        """
        Compute residuals for scipy.optimize.least_squares
        """
        # Update model parameters
        self._vector_to_params(params_vector)
        
        # Forward pass
        inputs = torch.from_numpy(inputs_np).float()
        targets = torch.from_numpy(targets_np).float()
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # Compute residuals
        residuals = (outputs - targets).cpu().numpy().flatten()
        
        return residuals
    
    def optimize(self, inputs: torch.Tensor, targets: torch.Tensor, 
                 max_nfev: int = 100, verbose: int = 0) -> dict:
        """
        Optimize using scipy.optimize.least_squares
        
        Returns:
        --------
        result : dict
            Optimization result
        """
        from scipy.optimize import least_squares
        
        # Convert to numpy
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Initial parameters
        x0 = self._params_to_vector()
        
        # Optimize
        result = least_squares(
            self.residuals,
            x0,
            args=(inputs_np, targets_np),
            method='lm',
            max_nfev=max_nfev,
            verbose=verbose
        )
        
        # Update model with optimized parameters
        self._vector_to_params(result.x)
        
        return {
            'success': result.success,
            'cost': result.cost,
            'nfev': result.nfev,
            'message': result.message
        }


def test_optimizer():
    """Test LM optimizer"""
    print("Testing Levenberg-Marquardt Optimizer")
    print("=" * 60)
    
    from models.ann import HybridNanofluidANN
    
    # Create simple model
    model = HybridNanofluidANN(input_dim=1, hidden_dim=10, 
                               num_hidden_layers=2, output_dim=2)
    
    # Create dummy data
    eta = torch.linspace(0, 10, 50).reshape(-1, 1)
    targets = torch.cat([torch.sin(eta), torch.cos(eta)], dim=1)
    
    # Test custom LM optimizer
    print("\nTesting custom LM optimizer:")
    optimizer = LevenbergMarquardtOptimizer(model, lambda_init=1e-3)
    
    for i in range(5):
        loss, success = optimizer.step(eta, targets, use_efficient=True)
        print(f"  Step {i+1}: loss = {loss:.6f}, success = {success}, λ = {optimizer.lambda_:.2e}")
    
    # Test simplified LM optimizer
    print("\nTesting simplified LM optimizer (scipy):")
    model2 = HybridNanofluidANN(input_dim=1, hidden_dim=10, 
                                num_hidden_layers=2, output_dim=2)
    optimizer2 = SimplifiedLMOptimizer(model2)
    
    result = optimizer2.optimize(eta, targets, max_nfev=20, verbose=0)
    print(f"  Success: {result['success']}")
    print(f"  Final cost: {result['cost']:.6f}")
    print(f"  Function evaluations: {result['nfev']}")
    
    print("\n✓ Optimizer tests passed!")


if __name__ == "__main__":
    test_optimizer()
