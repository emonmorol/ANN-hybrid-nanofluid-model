

import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src import generate_data
from src.data_loader import DataLoader
from src.trainer import Trainer
from src.visualizer import plot_training_history
from src.models.ann import HybridNanofluidANN

def run_data_generation():
    print("\n[1/3] Generating Dataset...")
    generator = generate_data.DatasetGenerator(output_dir=config.DATA_DIR)
    train_data = generator.generate_dataset(filename="training_data.csv")
    if train_data is not None:
        generator.generate_test_cases(n_cases=10, filename="test_data.csv")

def run_training(visualize=True):
    print("\n[2/3] Training Model...")
    
    # Check data
    if not config.TRAIN_DATA_PATH.exists():
        print("Error: Training data not found. Please generate data first.")
        return

    # Load data
    data_loader = DataLoader(config.TRAIN_DATA_PATH)
    data = data_loader.load_data(normalize=True)
    data_loader.save_scalers(config.MODEL_DIR)
    
    # Initialize model
    print("\nInitializing model...")
    model = HybridNanofluidANN(
        input_dim=config.MODEL_PARAMS['input_dim'],
        hidden_dim=config.MODEL_PARAMS['hidden_dim'],
        num_hidden_layers=config.MODEL_PARAMS['num_hidden_layers'],
        output_dim=config.MODEL_PARAMS['output_dim']
    )
    print(model.get_architecture_summary())
    
    # Initialize trainer
    print(f"Using {config.TRAIN_PARAMS['optimizer']} optimizer...")
    trainer = Trainer(
        model, 
        optimizer_type=config.TRAIN_PARAMS['optimizer'], 
        device='cpu', 
        visualize=visualize
    )
    
    # Determine batch size
    if config.TRAIN_PARAMS['batch_size'] == 'full':
        batch_size = len(data['X_train'])
    else:
        batch_size = int(config.TRAIN_PARAMS['batch_size'])
        
    print(f"Training with batch size: {batch_size}")
    
    # Train
    history = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        epochs=config.TRAIN_PARAMS['epochs'],
        batch_size=batch_size,
        early_stopping_patience=config.TRAIN_PARAMS['early_stopping_patience']
    )
    
    # Evaluate
    print("\n[3/3] Evaluating on test set...")
    metrics = trainer.evaluate(data['X_test'], data['y_test'])
    
    print("\nTest Set Metrics:")
    print(f"  MSE (overall): {metrics['mse']:.6e}")
    print(f"  MAE (overall): {metrics['mae']:.6e}")
    print(f"  Max Error: {metrics['max_error']:.6e}")
    
    # Save model
    trainer.save_model(config.BEST_MODEL_PATH)
    
    # Plot history
    plot_training_history(history, save_path=config.PLOT_DIR / "training_history.png")
    
    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)


def run_clean():
    print("\nCleaning up generated files...")
    
    files_to_remove = [
        config.TRAIN_DATA_PATH,
        config.TEST_DATA_PATH,
        config.BEST_MODEL_PATH,
        config.MODEL_DIR / "scaler_eta.pkl",
        config.MODEL_DIR / "scaler_f.pkl",
        config.MODEL_DIR / "scaler_theta.pkl"
    ]
    
    for file_path in files_to_remove:
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✓ Removed: {file_path}")
            except Exception as e:
                print(f"✗ Failed to remove {file_path}: {e}")
        else:
            print(f"  Skipped (not found): {file_path}")
            
    print("\nCleanup complete. You can now start fresh.")

def run_regenerate_plot():
    print("\nRegenerating training history plot...")
    
    # Check if model exists
    if not config.BEST_MODEL_PATH.exists():
        print(f"Error: Model checkpoint not found at {config.BEST_MODEL_PATH}")
        print("Please train the model first using 'python src/main.py train'")
        return
    
    # Load the saved model checkpoint
    checkpoint = torch.load(config.BEST_MODEL_PATH)
    
    # Extract training history
    history = checkpoint['history']
    
    # Regenerate the plot with the enhanced design
    save_path = config.PLOT_DIR / "training_history.png"
    plot_training_history(history, save_path=save_path)
    
    print(f"\n✓ Training history plot regenerated successfully!")
    print(f"  Location: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="ANN Hybrid Nanofluid Pipeline")
    parser.add_argument('action', choices=['generate', 'train', 'all', 'clean', 'regenerate'], 
                        help="Action to perform")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if args.action == 'generate':
        run_data_generation()
    elif args.action == 'train':
        run_training(visualize=True)
    elif args.action == 'clean':
        run_clean()
    elif args.action == 'regenerate':
        run_regenerate_plot()
    elif args.action == 'all':
        run_clean()
        run_data_generation()
        run_training(visualize=True)

if __name__ == "__main__":
    main()
