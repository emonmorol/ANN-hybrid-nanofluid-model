"""
Quick Start Script
Runs the complete pipeline: data generation → training → plotting
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False


def main():
    """Run complete pipeline"""
    print("=" * 70)
    print("ANN HYBRID NANOFLUID - COMPLETE PIPELINE")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Generate training data (numerical ODE solutions)")
    print("  2. Train ANN model with Levenberg-Marquardt")
    print("  3. Generate manuscript-style plots")
    print("\nEstimated time: 15-45 minutes")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Check if data already exists
    data_file = Path("data/training_data.csv")
    
    if data_file.exists():
        print(f"\n⚠ Found existing data file: {data_file}")
        response = input("Regenerate data? (y/n): ")
        if response.lower() == 'y':
            generate_data = True
        else:
            generate_data = False
            print("Skipping data generation...")
    else:
        generate_data = True
    
    # Step 1: Generate data
    if generate_data:
        if not run_command("python generate_data.py", "Data Generation"):
            print("\n✗ Pipeline failed at data generation")
            return
    
    # Step 2: Train model
    if not run_command("python train_ann.py", "Model Training"):
        print("\n✗ Pipeline failed at training")
        return
    
    # Step 3: Generate plots
    if not run_command("python plot_results.py", "Plot Generation"):
        print("\n✗ Pipeline failed at plotting")
        return
    
    # Success
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  📊 Data: data/training_data.csv")
    print("  🤖 Model: models/checkpoints/best_model.pth")
    print("  📈 Plots: plots/*.png")
    print("\nNext steps:")
    print("  - Review plots in the 'plots' directory")
    print("  - Check training metrics in training output")
    print("  - Modify parameters in generate_data.py for new cases")


if __name__ == "__main__":
    main()
