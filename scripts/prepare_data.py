"""
Convert raw.xlsx data to format required for DL model training
"""

import pandas as pd
import numpy as np
from pathlib import Path


def prepare_raw_dataset(input_file='data/raw.xlsx', output_file='data/prepared_raw_data.csv'):
    """
    Load raw.xlsx and prepare it for DL route optimizer training.
    
    Required columns in output:
    - route_id, stop_id, driver_id, country, day_of_week
    - indexp, indexa (planned and actual positions)
    - distancep, distancea
    - earliest_time, latest_time
    - depot, delivery
    - delay_flag, delay_minutes (optional)
    """
    
    print("=" * 80)
    print("PREPARING RAW DATASET FOR DL TRAINING")
    print("=" * 80)
    
    # Try to install openpyxl if not available
    try:
        import openpyxl
    except ImportError:
        print("\nInstalling openpyxl...")
        import subprocess
        subprocess.run(['pip', 'install', 'openpyxl'], check=True)
        import openpyxl
    
    # Load raw data
    print(f"\nLoading data from {input_file}...")
    df = pd.read_excel(input_file)
    
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check if data is already in correct format
    required_cols = ['route_id', 'indexp', 'indexa']
    
    if all(col in df.columns for col in required_cols):
        print("\nData already has required columns!")
        
        # Ensure all required columns exist, add defaults if missing
        if 'stop_id' not in df.columns:
            df['stop_id'] = df.groupby('route_id').cumcount().apply(lambda x: f'S{x}')
        
        if 'driver_id' not in df.columns:
            df['driver_id'] = df['route_id'].apply(lambda x: f'D{x % 100:03d}')
        
        if 'country' not in df.columns:
            df['country'] = 'Netherlands'
        
        if 'day_of_week' not in df.columns:
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df['day_of_week'] = df['route_id'].apply(lambda x: days[x % 7])
        
        if 'depot' not in df.columns:
            df['depot'] = 0
            df.loc[df.groupby('route_id').head(1).index, 'depot'] = 1
        
        if 'delivery' not in df.columns:
            df['delivery'] = 1 - df['depot']
        
        if 'earliest_time' not in df.columns:
            # Generate time windows based on position
            df['earliest_time'] = df['indexp'].apply(lambda x: f"{8 + (x * 30) // 60:02d}:{(x * 30) % 60:02d}:00")
        
        if 'latest_time' not in df.columns:
            df['latest_time'] = df['indexp'].apply(lambda x: f"{9 + (x * 30) // 60:02d}:{(x * 30) % 60:02d}:00")
        
        # Ensure distance columns exist
        if 'distancep' not in df.columns:
            df['distancep'] = np.random.uniform(5, 20, len(df))
        
        if 'distancea' not in df.columns:
            df['distancea'] = df['distancep'] * np.random.uniform(0.95, 1.05, len(df))
        
        # Add delay information if not present
        if 'delay_flag' not in df.columns:
            df['delay_flag'] = 0
        
        if 'delay_minutes' not in df.columns:
            df['delay_minutes'] = 0.0
        
    else:
        print("\nData needs to be converted to required format")
        print("Assuming raw.xlsx has original format (RouteID, DriverID, StopID, etc.)")
        
        # Map original column names to required names (handle spaces)
        column_mapping = {
            'Route ID': 'route_id',
            'RouteID': 'route_id',
            'Driver ID': 'driver_id',
            'DriverID': 'driver_id',
            'Stop ID': 'stop_id',
            'StopID': 'stop_id',
            'Address ID': 'address_id',
            'AddressID': 'address_id',
            'Week ID': 'week_id',
            'WeekID': 'week_id',
            'Country': 'country',
            'Day of Week': 'day_of_week',
            'DayOfWeek': 'day_of_week',
            'IndexP': 'indexp',
            'IndexA': 'indexa',
            'DistanceP': 'distancep',
            'DistanceA': 'distancea',
            'Depot': 'depot',
            'Delivery': 'delivery',
            'Earliest Time': 'earliest_time',
            'EarliestTime': 'earliest_time',
            'Latest Time': 'latest_time',
            'LatestTime': 'latest_time',
            'Arrived Time': 'arrived_time',
            'ArrivedTime': 'arrived_time'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        print(f"\nColumns after renaming: {df.columns.tolist()}")
        
        # Add missing columns with defaults
        if 'delay_flag' not in df.columns and 'arrived_time' in df.columns and 'latest_time' in df.columns:
            # Calculate delay flag
            df['delay_flag'] = (pd.to_datetime(df['arrived_time']) > pd.to_datetime(df['latest_time'])).astype(int)
        else:
            df['delay_flag'] = 0
        
        if 'delay_minutes' not in df.columns:
            df['delay_minutes'] = 0.0
    
    # Select and order columns
    output_columns = [
        'route_id', 'stop_id', 'driver_id', 'country', 'day_of_week',
        'indexp', 'indexa', 'distancep', 'distancea',
        'depot', 'delivery', 'earliest_time', 'latest_time',
        'delay_flag', 'delay_minutes'
    ]
    
    # Ensure all columns exist
    for col in output_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, adding default values")
            df[col] = 0 if col in ['depot', 'delivery', 'delay_flag'] else ''
    
    df_output = df[output_columns]
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_output.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Output saved to: {output_path.absolute()}")
    print(f"Total rows: {len(df_output):,}")
    print(f"Unique routes: {df_output['route_id'].nunique():,}")
    print(f"Columns: {output_columns}")
    
    # Print sample
    print("\nSample of prepared data:")
    print(df_output.head(10))
    
    return df_output


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare raw data for DL training')
    parser.add_argument('--input', type=str, default='data/raw.xlsx',
                       help='Input file (Excel)')
    parser.add_argument('--output', type=str, default='data/prepared_raw_data.csv',
                       help='Output file (CSV)')
    
    args = parser.parse_args()
    
    prepare_raw_dataset(args.input, args.output)

