import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime, timedelta

def generate_sample_data(output_dir: str = 'data/sample_data', n_subjects: int = 1000, n_cases: int = 100, n_phecodes: int = 50):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    np.random.seed(42)
    random.seed(42)
    
    # Generate GRIDs
    grids = [f'R{i:09d}' for i in range(1, n_subjects + 1)]
    
    # 1. Demographics
    print("Generating demographics...")
    birth_dates = [datetime(1950, 1, 1) + timedelta(days=random.randint(0, 20000)) for _ in range(n_subjects)]
    genders = np.random.choice(['M', 'F'], n_subjects)
    races = np.random.choice(['W', 'B', 'A', 'U'], n_subjects, p=[0.7, 0.2, 0.05, 0.05])
    ethnicities = np.random.choice(['Not Hispanic', 'Hispanic', 'Unknown'], n_subjects, p=[0.8, 0.1, 0.1])
    
    demo_df = pd.DataFrame({
        'grid': grids,
        'birth_datetime': [d.strftime('%Y-%m-%d') for d in birth_dates],
        'gender_source_value': genders,
        'race_source_value': races,
        'ethnicity_source_value': ethnicities
    })
    demo_df.to_csv(output_path / 'sd_demographics.csv', index=False)
    
    # 2. Depth of Record
    print("Generating depth of record...")
    depths = np.random.uniform(0, 20, n_subjects)
    depth_df = pd.DataFrame({
        'grid': grids,
        'depth_of_record': depths
    })
    depth_df.to_csv(output_path / 'depth_of_record.csv', index=False)
    
    # 3. Cases
    print("Generating cases...")
    case_grids = np.random.choice(grids, n_cases, replace=False)
    # Add some ICD counts
    icd_counts = np.random.randint(1, 10, n_cases)
    case_df = pd.DataFrame({
        'grid': case_grids,
        'icd_code_count': icd_counts
    })
    case_df.to_csv(output_path / 'cases.csv', index=False)
    
    # 4. Control Exclusion List (Optional)
    print("Generating control exclusion list...")
    # Just a few random grids
    excluded_grids = np.random.choice([g for g in grids if g not in case_grids], 10, replace=False)
    pd.DataFrame({'grid': excluded_grids}).to_csv(output_path / 'excluded_controls.txt', index=False, header=False)
    
    # 5. Phecode Binary
    print("Generating phecode binary data...")
    # Generate some fake phecodes
    phecodes = [f'{i}.{j}' for i in range(100, 100 + int(n_phecodes/2)) for j in range(2)]
    phecodes = phecodes[:n_phecodes]
    
    # Create a binary matrix
    phecode_matrix = np.random.choice([0, 1], size=(n_subjects, len(phecodes)), p=[0.9, 0.1])
    phecode_df = pd.DataFrame(phecode_matrix, columns=phecodes)
    phecode_df['grid'] = grids
    
    # Move grid to first column
    cols = ['grid'] + [c for c in phecode_df.columns if c != 'grid']
    phecode_df = phecode_df[cols]
    
    phecode_df.to_feather(output_path / 'phecode_binary.feather')
    
    print(f"Data generated in {output_path}")

if __name__ == "__main__":
    generate_sample_data()
