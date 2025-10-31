import pandas as pd
from pathlib import Path

def prepare_sample_data():
    ROOT = Path("E:/Final-Year-Project")
    DATA_IN = ROOT / "data" / "processed" / "xgboost"
    OUTPUT_DIR = ROOT / "data" / "processed"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("Loading house data...")
    # Load only first 3 houses for testing
    dfs = []
    for i in range(1, 4):  # Houses 1-3
        house_file = DATA_IN / f"house_{i:02d}_15min.csv"
        print(f"Checking file: {house_file}")
        if house_file.exists():
            print(f"Loading house {i}...")
            df = pd.read_csv(house_file)
            df['house_id'] = i
            dfs.append(df)
            print(f"House {i} shape: {df.shape}")
    
    if not dfs:
        print("No house data found!")
        return
        
    print("Combining data...")
    # Combine data
    df = pd.concat(dfs, ignore_index=True)
    
    # Rename columns to match training script expectations
    print("Processing columns...")
    df = df.rename(columns={
        'datetime': 'timestamp',
        'pv_power': 'generation',
        'load_power': 'usage'
    })
    
    # Sort by house and time
    df = df.sort_values(['house_id', 'timestamp'])
    
    # Save as CSV
    output_file = OUTPUT_DIR / "multi_home_15min.csv"
    print(f"Saving to {output_file}")
    df.to_csv(output_file, index=False)
    print(f"Saved sample dataset to {output_file}")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

if __name__ == "__main__":
    prepare_sample_data()