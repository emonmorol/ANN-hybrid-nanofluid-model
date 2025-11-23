

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import itertools

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.solver.ode_solver import HybridNanofluidSolver


class DatasetGenerator:

    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_parameter_grid(self) -> list:

        from src import config
        
        # Use itertools.product to generate all combinations
        keys = config.PARAM_RANGES.keys()
        values = config.PARAM_RANGES.values()
        
        param_combinations = []
        
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            # Add solver settings
            params['eta_max'] = config.ETA_MAX
            params['n_points'] = config.N_POINTS
            
            param_combinations.append(params)
        
        return param_combinations
    
    def solve_single_case(self, params: dict, case_id: int) -> pd.DataFrame:

        solver = HybridNanofluidSolver(params)
        eta, solution = solver.solve(verbose=False)
        
        if solution is None:
            return None
        
        # Extract results
        results = solver.compute_derivatives(eta, solution)
        eng_quantities = solver.compute_engineering_quantities(solution)
        
        # Create DataFrame
        df = pd.DataFrame({
            'case_id': case_id,
            'eta': results['eta'],
            'f': results['f'],
            'fp': results['fp'],
            'fpp': results['fpp'],
            'theta': results['theta'],
            'thetap': results['thetap'],
            'M': params['M'],
            'Nr': params['Nr'],
            'Nh': params['Nh'],
            'lam': params['lam'],
            'beta': params['beta'],
            'Pr': params['Pr'],
            'n': params['n'],
            'Tr': params['Tr'],
            'As': params['As'],
            'Cf': eng_quantities['Cf'],
            'Nu': eng_quantities['Nu']
        })
        
        return df
    
    def generate_dataset(self, filename: str = "training_data.csv") -> pd.DataFrame:

        param_combinations = self.generate_parameter_grid()
        
        print(f"Generating dataset with {len(param_combinations)} parameter combinations...")
        print(f"Total data points: {len(param_combinations) * 400}")
        
        all_data = []
        failed_cases = []
        
        for case_id, params in enumerate(tqdm(param_combinations, desc="Solving ODE cases")):
            df = self.solve_single_case(params, case_id)
            
            if df is not None:
                all_data.append(df)
            else:
                failed_cases.append(case_id)
                print(f"\n⚠ Warning: Case {case_id} failed with params: {params}")
        
        # Concatenate all data
        if all_data:
            full_dataset = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV
            output_path = self.output_dir / filename
            full_dataset.to_csv(output_path, index=False)
            
            print(f"\n✓ Dataset saved to: {output_path}")
            print(f"  Total cases: {len(param_combinations)}")
            print(f"  Successful: {len(all_data)}")
            print(f"  Failed: {len(failed_cases)}")
            print(f"  Total rows: {len(full_dataset)}")
            print(f"  Columns: {list(full_dataset.columns)}")
            
            # Print statistics
            print("\nDataset Statistics:")
            print(full_dataset.describe())
            
            return full_dataset
        else:
            print("✗ No successful cases!")
            return None
    
    def generate_test_cases(self, n_cases: int = 10, filename: str = "test_data.csv") -> pd.DataFrame:

        print(f"\nGenerating {n_cases} random test cases...")
        
        np.random.seed(42)
        test_data = []
        
        for case_id in tqdm(range(n_cases), desc="Generating test cases"):
            params = {
                'M': np.random.uniform(0.3, 2.5),
                'Nr': np.random.uniform(0.1, 1.2),
                'Nh': np.random.uniform(0.1, 1.2),
                'lam': np.random.uniform(0.3, 2.5),
                'beta': 0.1,
                'Pr': 6.2,
                'n': 1.0,
                'Tr': 1.5,
                'As': 1.0,
                'eta_max': 10.0,
                'n_points': 400
            }
            
            df = self.solve_single_case(params, case_id + 1000)
            if df is not None:
                test_data.append(df)
        
        if test_data:
            test_dataset = pd.concat(test_data, ignore_index=True)
            output_path = self.output_dir / filename
            test_dataset.to_csv(output_path, index=False)
            
            print(f"✓ Test dataset saved to: {output_path}")
            print(f"  Total rows: {len(test_dataset)}")
            
            return test_dataset
        else:
            return None


def main():

    print("=" * 70)
    print("ANN Hybrid Nanofluid - Dataset Generator")
    print("=" * 70)
    
    generator = DatasetGenerator(output_dir="data")
    
    # Generate training dataset
    train_data = generator.generate_dataset(filename="training_data.csv")
    
    # Generate test dataset
    if train_data is not None:
        test_data = generator.generate_test_cases(n_cases=10, filename="test_data.csv")
    
    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
