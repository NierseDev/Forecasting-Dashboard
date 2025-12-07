"""
Data Augmentation Script
Combines Synthetic and Real datasets in a 70:30 ratio
Creates augmented datasets combining 70% synthetic and 30% real data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict

# Define paths
BASE_DIR = Path(__file__).parent
SYN_DATA_DIR = BASE_DIR / "DATA" / "SYN-DATA"
REAL_DATA_DIR = BASE_DIR / "DATA" / "REAL-DATA"
OUTPUT_DIR = BASE_DIR / "DATA" / "AUGMENTED-DATA"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping of synthetic file names to real data file names
LOCATION_MAPPING = {
    "daet_daily_synthetic.csv": "Daet Daily Data.csv",
    "juban_daily_synthetic.csv": "Juban Daily Data.csv",
    "legazpi_daily_synthetic.csv": "Legazpi Daily Data.csv",
    "masbate_daily_synthetic.csv": "Masbate Daily Data.csv",
    "virac_synop_daily_synthetic.csv": "Virac Synop Daily Data.csv",
}


def load_and_normalize_data(filepath: Path) -> pd.DataFrame:
    """
    Load CSV data and normalize column names to uppercase
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with normalized columns
    """
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip().upper() for c in df.columns]
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def augment_datasets(synthetic_ratio: float = 0.7, real_ratio: float = 0.3) -> Dict[str, Tuple[pd.DataFrame, str]]:
    """
    Augment synthetic datasets by combining with real data in specified ratio
    
    Args:
        synthetic_ratio: Ratio of synthetic data (default 0.7 for 70%)
        real_ratio: Ratio of real data (default 0.3 for 30%)
        
    Returns:
        Dictionary mapping location names to (augmented_dataframe, output_path) tuples
    """
    
    augmented_datasets = {}
    
    for syn_file, real_file in LOCATION_MAPPING.items():
        syn_path = SYN_DATA_DIR / syn_file
        real_path = REAL_DATA_DIR / real_file
        
        print(f"\nProcessing {syn_file}...")
        
        # Load synthetic and real data
        syn_df = load_and_normalize_data(syn_path)
        real_df = load_and_normalize_data(real_path)
        
        if syn_df is None or real_df is None:
            print(f"  ⚠ Skipping {syn_file} - failed to load data")
            continue
        
        print(f"  Synthetic data shape: {syn_df.shape}")
        print(f"  Real data shape: {real_df.shape}")
        
        # Calculate sample sizes based on ratio
        total_synthetic = len(syn_df)
        total_real = len(real_df)
        
        # Calculate how many rows we need from each dataset
        # If we want 70% synthetic and 30% real in final dataset
        # We need to use all synthetic data and calculate required real data
        synthetic_count = total_synthetic
        real_count = int(total_synthetic * (real_ratio / synthetic_ratio))
        
        # Ensure we don't exceed available real data
        real_count = min(real_count, total_real)
        
        print(f"  Using {synthetic_count} synthetic records ({synthetic_count/(synthetic_count + real_count)*100:.1f}%)")
        print(f"  Using {real_count} real records ({real_count/(synthetic_count + real_count)*100:.1f}%)")
        
        # Sample from real data (random sampling without replacement)
        if real_count > 0:
            real_sample = real_df.sample(n=min(real_count, total_real), random_state=42)
        else:
            real_sample = pd.DataFrame()
        
        # Ensure both dataframes have the same columns
        # Keep only columns that exist in both datasets
        common_cols = list(set(syn_df.columns) & set(real_sample.columns)) if len(real_sample) > 0 else list(syn_df.columns)
        common_cols = sorted(common_cols)
        
        syn_aligned = syn_df[common_cols]
        
        if len(real_sample) > 0:
            real_aligned = real_sample[common_cols]
            
            # Combine datasets
            augmented_df = pd.concat([syn_aligned, real_aligned], ignore_index=True)
            
            # Shuffle the combined dataset
            augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            augmented_df = syn_aligned.reset_index(drop=True)
        
        print(f"  Augmented data shape: {augmented_df.shape}")
        
        # Generate output filename
        location_name = syn_file.replace("_daily_synthetic.csv", "").title()
        output_filename = f"{location_name}_augmented_70_30.csv"
        output_path = OUTPUT_DIR / output_filename
        
        # Save augmented dataset
        augmented_df.to_csv(output_path, index=False)
        print(f"  ✓ Saved to {output_path}")
        
        augmented_datasets[location_name] = (augmented_df, str(output_path))
    
    return augmented_datasets


def generate_augmentation_report(augmented_datasets: Dict[str, Tuple[pd.DataFrame, str]]) -> None:
    """
    Generate a summary report of augmentation process
    
    Args:
        augmented_datasets: Dictionary of augmented datasets
    """
    
    report_path = OUTPUT_DIR / "AUGMENTATION_REPORT.json"
    
    report = {
        "augmentation_config": {
            "synthetic_ratio": 0.7,
            "real_ratio": 0.3,
            "method": "Random sampling and concatenation"
        },
        "datasets": {}
    }
    
    for location, (df, path) in augmented_datasets.items():
        report["datasets"][location] = {
            "output_file": path,
            "total_rows": len(df),
            "columns": list(df.columns),
            "shape": [len(df), len(df.columns)]
        }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to {report_path}")


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("DATA AUGMENTATION - 70% Synthetic, 30% Real Data")
    print("=" * 70)
    
    # Check if directories exist
    if not SYN_DATA_DIR.exists():
        print(f"Error: Synthetic data directory not found: {SYN_DATA_DIR}")
        return
    
    if not REAL_DATA_DIR.exists():
        print(f"Error: Real data directory not found: {REAL_DATA_DIR}")
        return
    
    # Perform augmentation
    augmented_datasets = augment_datasets(synthetic_ratio=0.7, real_ratio=0.3)
    
    if augmented_datasets:
        print("\n" + "=" * 70)
        print("AUGMENTATION SUMMARY")
        print("=" * 70)
        
        for location, (df, path) in augmented_datasets.items():
            print(f"\n{location}:")
            print(f"  Total rows: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Output: {path}")
        
        # Generate report
        generate_augmentation_report(augmented_datasets)
        
        print("\n" + "=" * 70)
        print("✓ Augmentation completed successfully!")
        print(f"Augmented datasets saved to: {OUTPUT_DIR}")
        print("=" * 70)
    else:
        print("\nNo datasets were augmented. Check error messages above.")


if __name__ == "__main__":
    main()
