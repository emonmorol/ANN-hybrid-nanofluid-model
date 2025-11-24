
from pathlib import Path




BASE_DIR = Path(__file__).parent.parent
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

PARAM_RANGES = {
    'M': [0.5, 1.0, 2.0],           # 3 values
    'Nr': [0.5, 1.0, 1.5],          # 3 values
    'Nh': [0.1, 0.5],               # 2 values
    'lam': [0.5, 1.0, 1.5],         # 3 values
    'beta': [0.1, 0.2],             # 2 values
    'Pr': [6.2],                    # 1 value
    'n': [1.0],                     # 1 value
    'Tr': [1.5],                    # 1 value
    'As': [1.0],                    # 1 value
}
# Total: 3×3×2×3×2 = 108 combinations



ETA_MAX = 10.0
N_POINTS = 400




MODEL_PARAMS = {
    'input_dim': 1,
    'hidden_dim': 30,
    'num_hidden_layers': 9,
    'output_dim': 2,
    'activation': 'tanh'
}




TRAIN_PARAMS = {
    'epochs': 100,
    'batch_size': 512,              
    'optimizer': 'adam',
    'early_stopping_patience': 1000,
    'test_split': 0.1,
    'val_split': 0.1
}
