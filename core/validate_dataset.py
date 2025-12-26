"""
Dataset Validation Script
Quick checks to ensure dataset quality before training
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def validate_dataset(df_path):
    """
    Run validation checks on dataset
    
    Args:
        df_path: Path to dataset CSV file
    
    Returns:
        dict: Validation results
    """
    print("\n" + "=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)
    print(f"\nValidating: {df_path}")
    
    # Load dataset
    try:
        df = pd.read_csv(df_path)
        print(f"✅ Dataset loaded: {len(df):,} records")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    checks = {}
    all_passed = True
    
    # Check 1: No missing values
    print("\n" + "-" * 80)
    print("CHECK 1: Missing Values")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) == 0:
        checks["no_missing"] = True
        print("✅ No missing values")
    else:
        checks["no_missing"] = False
        all_passed = False
        print(f"❌ Missing values found:")
        for col, count in missing_cols.items():
            print(f"   {col}: {count} ({count/len(df):.2%})")
    
    # Check 2: Delay rate is realistic
    print("\n" + "-" * 80)
    print("CHECK 2: Delay Rate")
    print("-" * 80)
    if 'delay_flag' in df.columns:
        delay_rate = df['delay_flag'].mean()
        if 0.05 <= delay_rate <= 0.20:
            checks["delay_rate"] = True
            print(f"✅ Delay rate: {delay_rate:.2%} (realistic: 5-20%)")
        else:
            checks["delay_rate"] = False
            all_passed = False
            print(f"❌ Delay rate: {delay_rate:.2%} (should be 5-20%)")
    else:
        checks["delay_rate"] = False
        all_passed = False
        print("❌ 'delay_flag' column not found")
    
    # Check 3: Sufficient variance in numerical features
    print("\n" + "-" * 80)
    print("CHECK 3: Feature Variance")
    print("-" * 80)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    low_variance_cols = []
    for col in numeric_cols:
        if col in ['route_id', 'driver_id', 'stop_id', 'address_id', 'week_id', 
                   'country', 'depot', 'delivery', 'delay_flag']:
            continue  # Skip ID and binary columns
        std = df[col].std()
        if std < 0.01:
            low_variance_cols.append((col, std))
    
    if len(low_variance_cols) == 0:
        checks["sufficient_variance"] = True
        print("✅ All numerical features have sufficient variance (>0.01)")
    else:
        checks["sufficient_variance"] = False
        all_passed = False
        print(f"❌ Low variance features found:")
        for col, std in low_variance_cols:
            print(f"   {col}: std={std:.6f}")
    
    # Check 4: Reasonable delay values
    print("\n" + "-" * 80)
    print("CHECK 4: Delay Values")
    print("-" * 80)
    if 'delay_minutes' in df.columns:
        max_delay = df['delay_minutes'].max()
        mean_delay = df[df['delay_minutes'] > 0]['delay_minutes'].mean() if (df['delay_minutes'] > 0).any() else 0
        
        if max_delay < 300:  # Less than 5 hours
            checks["reasonable_delays"] = True
            print(f"✅ Max delay: {max_delay:.2f} minutes (< 5 hours)")
        else:
            checks["reasonable_delays"] = False
            all_passed = False
            print(f"❌ Max delay: {max_delay:.2f} minutes (unrealistic: > 5 hours)")
        
        if mean_delay > 0:
            print(f"   Mean delay (when delayed): {mean_delay:.2f} minutes")
    else:
        checks["reasonable_delays"] = False
        all_passed = False
        print("❌ 'delay_minutes' column not found")
    
    # Check 5: Route sizes are reasonable
    print("\n" + "-" * 80)
    print("CHECK 5: Route Sizes")
    print("-" * 80)
    if 'route_id' in df.columns:
        route_sizes = df.groupby('route_id').size()
        avg_route_size = route_sizes.mean()
        min_route_size = route_sizes.min()
        max_route_size = route_sizes.max()
        
        if 5 <= avg_route_size <= 40:
            checks["route_sizes"] = True
            print(f"✅ Average route size: {avg_route_size:.1f} stops (reasonable: 5-40)")
            print(f"   Range: {min_route_size}-{max_route_size} stops")
        else:
            checks["route_sizes"] = False
            all_passed = False
            print(f"❌ Average route size: {avg_route_size:.1f} stops (should be 5-40)")
    else:
        checks["route_sizes"] = False
        all_passed = False
        print("❌ 'route_id' column not found")
    
    # Check 6: Required columns exist
    print("\n" + "-" * 80)
    print("CHECK 6: Required Columns")
    print("-" * 80)
    required_cols = [
        'route_id', 'driver_id', 'stop_id', 'country', 'day_of_week',
        'indexp', 'indexa', 'arrived_time', 'earliest_time', 'latest_time',
        'distancep', 'distancea', 'depot', 'delivery'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if len(missing_cols) == 0:
        checks["required_columns"] = True
        print("✅ All required columns present")
    else:
        checks["required_columns"] = False
        all_passed = False
        print(f"❌ Missing required columns: {missing_cols}")
    
    # Check 7: Data types are correct
    print("\n" + "-" * 80)
    print("CHECK 7: Data Types")
    print("-" * 80)
    type_issues = []
    
    # Check time columns can be parsed
    time_cols = ['arrived_time', 'earliest_time', 'latest_time']
    for col in time_cols:
        if col in df.columns:
            try:
                pd.to_datetime(df[col].iloc[0], format='%H:%M:%S', errors='coerce')
            except:
                type_issues.append(f"{col} (time format)")
    
    # Check numeric columns
    numeric_cols_expected = ['route_id', 'driver_id', 'stop_id', 'distancep', 'distancea']
    for col in numeric_cols_expected:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            type_issues.append(f"{col} (should be numeric)")
    
    if len(type_issues) == 0:
        checks["data_types"] = True
        print("✅ Data types are correct")
    else:
        checks["data_types"] = False
        all_passed = False
        print(f"❌ Data type issues found:")
        for issue in type_issues:
            print(f"   {issue}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(checks.values())
    total = len(checks)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if all_passed:
        print("\n✅ All validation checks passed! Dataset is ready for training.")
    else:
        print("\n⚠️  Some validation checks failed. Please review and fix issues.")
        print("\nFailed checks:")
        for check, passed in checks.items():
            if not passed:
                print(f"  ❌ {check}")
    
    return checks


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate dataset quality')
    parser.add_argument('--dataset', type=str, 
                       default='data/improved_delivery_data.csv',
                       help='Path to dataset CSV file')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Please generate the dataset first:")
        print(f"   python generate_improved_dataset.py")
        sys.exit(1)
    
    checks = validate_dataset(dataset_path)
    
    if checks and all(checks.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

