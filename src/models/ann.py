
import torch
import torch.nn as nn
import numpy as np


class HybridNanofluidANN(nn.Module):

    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 30, 
                 num_hidden_layers: int = 9, output_dim: int = 2):

        super(HybridNanofluidANN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier uniform
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, eta: torch.Tensor) -> torch.Tensor:
        return self.network(eta)
    
    def predict(self, eta: np.ndarray) -> np.ndarray:

        self.eval()
        with torch.no_grad():
            if len(eta.shape) == 1:
                eta = eta.reshape(-1, 1)
            
            eta_tensor = torch.FloatTensor(eta)
            output = self.forward(eta_tensor)
            return output.numpy()
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_summary(self) -> str:
        """Get human-readable architecture summary"""
        summary = []
        summary.append("=" * 60)
        summary.append("ANN Architecture Summary")
        summary.append("=" * 60)
        summary.append(f"Input dimension:        {self.input_dim}")
        summary.append(f"Hidden layers:          {self.num_hidden_layers}")
        summary.append(f"Neurons per layer:      {self.hidden_dim}")
        summary.append(f"Activation function:    Tanh")
        summary.append(f"Output dimension:       {self.output_dim}")
        summary.append(f"Total parameters:       {self.count_parameters():,}")
        summary.append("=" * 60)
        
        return "\n".join(summary)


class ANNWithDerivatives(nn.Module):
    
    def __init__(self, base_model: HybridNanofluidANN):
        super(ANNWithDerivatives, self).__init__()
        self.model = base_model
    
    def forward(self, eta: torch.Tensor) -> dict:
        eta.requires_grad_(True)
        
        # Forward pass
        output = self.model(eta)
        f = output[:, 0:1]
        theta = output[:, 1:2]
        
        # Compute first derivatives
        fp = torch.autograd.grad(f, eta, torch.ones_like(f), create_graph=True)[0]
        thetap = torch.autograd.grad(theta, eta, torch.ones_like(theta), create_graph=True)[0]
        
        # Compute second derivatives
        fpp = torch.autograd.grad(fp, eta, torch.ones_like(fp), create_graph=True)[0]
        
        return {
            'f': f,
            'theta': theta,
            'fp': fp,
            'fpp': fpp,
            'thetap': thetap
        }


def test_model():
    """Test the ANN model"""
    print("Testing ANN Model")
    print("=" * 60)
    
    # Create model
    model = HybridNanofluidANN(input_dim=1, hidden_dim=30, 
                               num_hidden_layers=9, output_dim=2)
    
    # Print architecture
    print(model.get_architecture_summary())
    
    # Test forward pass
    eta_test = torch.randn(10, 1)
    output = model(eta_test)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape:  {eta_test.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test numpy prediction
    eta_np = np.linspace(0, 10, 100)
    predictions = model.predict(eta_np)
    
    print(f"\nTest numpy prediction:")
    print(f"  Input shape:  {eta_np.shape}")
    print(f"  Output shape: {predictions.shape}")
    
    # Test derivative model
    print(f"\nTesting derivative computation:")
    deriv_model = ANNWithDerivatives(model)
    eta_test_deriv = torch.linspace(0, 10, 50).reshape(-1, 1)
    derivatives = deriv_model(eta_test_deriv)
    
    for key, value in derivatives.items():
        print(f"  {key}: shape {value.shape}")
    
    print("\n✓ Model tests passed!")


if __name__ == "__main__":
    test_model()
