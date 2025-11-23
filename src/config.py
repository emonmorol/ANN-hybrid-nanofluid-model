"""
Central Configuration for ANN Hybrid Nanofluid Project
"""

from pathlib import Path

# ==========================================
# Paths
# ==========================================
BASE_DIR = Path(__file__).parent.parent  # Go up from src/ to root
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "outputs" / "models"
PLOT_DIR = BASE_DIR / "outputs" / "plots"

DATA_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_DATA_PATH = DATA_DIR / "training_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
SCALER_DIR = MODEL_DIR
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"

# ==========================================
# Physics Parameters (Data Generation)
# ==========================================
# Reduced ranges for faster training (Total: ~48k data points)
# 5 × 4 × 3 × 2 × 2 × 1 × 1 × 1 × 1 = 120 combinations × 400 points = 48,000 rows
PARAM_RANGES = {
    'M': [0.1, 0.5, 1.0, 2.0, 3.0],                # Magnetic parameter (5 values)
    'Nr': [0.1, 0.5, 1.0, 1.5],                    # Radiation parameter (4 values)
    'Nh': [0.1, 0.5, 1.0],                         # Heat generation parameter (3 values)
    'lam': [0.5, 2.0],                             # Mixed convection parameter (2 values)
    'beta': [0.1, 0.3],                            # Casson parameter (2 values)
    'Pr': [6.2],                                   # Prandtl number (Water)
    'n': [1.0],                                    # Power law index
    'Tr': [1.5],                                   # Temperature ratio
    'As': [1.0],                                   # Stratification parameter
}

# Solver settings
ETA_MAX = 10.0
N_POINTS = 400

# ==========================================
# Model Hyperparameters
# ==========================================
MODEL_PARAMS = {
    'input_dim': 1,
    'hidden_dim': 30,
    'num_hidden_layers': 9,
    'output_dim': 2,  # [f, theta]
    'activation': 'tanh'
}

# ==========================================
# Training Settings
# ==========================================
TRAIN_PARAMS = {
    'epochs': 100,
    'batch_size': 'full',  # 'full' or int
    'learning_rate': 1.0,  # For L-BFGS
    'optimizer': 'lbfgs',  # 'lbfgs', 'adam', 'lm_custom'
    'early_stopping_patience': 20,
    'test_split': 0.1,
    'val_split': 0.1
}
