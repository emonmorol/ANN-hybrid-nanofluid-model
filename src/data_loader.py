

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config

class DataLoader:

    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.scaler_eta = MinMaxScaler()
        self.scaler_f = MinMaxScaler()
        self.scaler_theta = MinMaxScaler()
        
    def load_data(self, normalize: bool = True) -> dict:

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
            eta_normalized, targets, 
            test_size=(config.TRAIN_PARAMS['val_split'] + config.TRAIN_PARAMS['test_split']), 
            random_state=42
        )
        
        # Split temp into val and test (50/50 of the remaining 20% -> 10% each)
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

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        with open(output_dir / 'scaler_eta.pkl', 'wb') as f:
            pickle.dump(self.scaler_eta, f)
        with open(output_dir / 'scaler_f.pkl', 'wb') as f:
            pickle.dump(self.scaler_f, f)
        with open(output_dir / 'scaler_theta.pkl', 'wb') as f:
            pickle.dump(self.scaler_theta, f)
        
        print(f"✓ Scalers saved to {output_dir}")
