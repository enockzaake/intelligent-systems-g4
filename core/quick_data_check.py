"""
Quick Data Check - Fast insights without full analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path

def quick_check():
    """Quick check of data characteristics"""
    
    data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    print("\n" + "=" * 70)
    print("QUICK DATA CHECK")
    print("=" * 70)
    
    df = pd.read_csv(data_path)
    
    # Convert delay if needed
    if 'actual_arrival_delay' in df.columns:
        df['delay_minutes'] = np.maximum(0, df['actual_arrival_delay'] * 24 * 60)
        df['delayed_flag'] = (df['delay_minutes'] > 0).astype(int)
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  Total records: {len(df):,}")
    print(f"  Total routes: {df['route_id'].nunique():,}")
    print(f"  Avg stops/route: {len(df) / df['route_id'].nunique():.1f}")
    
    # Delay stats
    if 'delayed_flag' in df.columns:
        delay_rate = df['delayed_flag'].mean()
        print(f"\n⏱️  Delay Statistics:")
        print(f"  Delay rate: {delay_rate:.2%}")
        print(f"  On-time rate: {1-delay_rate:.2%}")
        
        if 'delay_minutes' in df.columns:
            delayed_only = df[df['delayed_flag'] == 1]
            
            print(f"\n  Delay magnitudes (for delayed stops only):")
            print(f"    Mean: {delayed_only['delay_minutes'].mean():.4f} min ({delayed_only['delay_minutes'].mean()*60:.2f} sec)")
            print(f"    Median: {delayed_only['delay_minutes'].median():.4f} min")
            print(f"    Max: {delayed_only['delay_minutes'].max():.4f} min ({delayed_only['delay_minutes'].max()*60:.2f} sec)")
            print(f"    Std: {delayed_only['delay_minutes'].std():.4f} min")
            
            # Check data quality issues
            print(f"\n🔍 Data Quality Checks:")
            
            # 1. Class imbalance
            if delay_rate < 0.15:
                print(f"  ⚠️  SEVERE CLASS IMBALANCE: {delay_rate:.1%} delayed")
                print(f"      → Need class weights or SMOTE")
            else:
                print(f"  ✅ Class balance OK: {delay_rate:.1%} delayed")
            
            # 2. Target variance
            variance = df['delay_minutes'].var()
            if variance < 0.01:
                print(f"  ⚠️  VERY LOW VARIANCE: {variance:.6f}")
                print(f"      → Regression will struggle")
            else:
                print(f"  ✅ Variance OK: {variance:.6f}")
            
            # 3. Zero inflation
            zero_rate = (df['delay_minutes'] == 0).mean()
            if zero_rate > 0.8:
                print(f"  ⚠️  ZERO-INFLATED: {zero_rate:.1%} are zero")
                print(f"      → Focus on classification")
            else:
                print(f"  ✅ Zero rate OK: {zero_rate:.1%}")
            
            # 4. Delays are tiny
            if delayed_only['delay_minutes'].max() < 1.0:
                print(f"  ⚠️  TINY DELAYS: Max {delayed_only['delay_minutes'].max()*60:.1f} seconds")
                print(f"      → Consider converting to seconds")
            else:
                print(f"  ✅ Delay scale OK")
    
    # Feature availability
    print(f"\n📋 Available Features:")
    key_features = ['indexp', 'indexa', 'distancep', 'distancea', 'driver_id', 
                    'day_of_week', 'country', 'arrived_time']
    for feat in key_features:
        if feat in df.columns:
            print(f"  ✅ {feat}")
        else:
            print(f"  ❌ {feat}")
    
    # Recommendation
    print(f"\n💡 Recommendations:")
    
    if 'delayed_flag' in df.columns:
        if delay_rate < 0.15:
            print(f"  1. Use class weights (weight_1 ≈ {1/delay_rate/2:.1f})")
        
        if variance < 0.01:
            print(f"  2. Skip regression, focus on classification")
        
        if delayed_only['delay_minutes'].mean() < 0.5:
            print(f"  3. Delays are very small - verify data collection")
        
        print(f"  4. Best approach: Classification with balanced weights")
    
    print("\n" + "=" * 70)
    print("CHECK COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    quick_check()

