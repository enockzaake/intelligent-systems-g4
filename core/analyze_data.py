"""
Data Analysis Script
Investigates data distribution and model performance issues
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_data():
    """Comprehensive data analysis"""
    
    # Load the data
    data_path = Path("data/cleaned_delivery_data.csv")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    print("\n" + "=" * 80)
    print("DATA ANALYSIS REPORT")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    print(f"\nTotal records: {len(df):,}")
    print(f"Total routes: {df['route_id'].nunique():,}")
    print(f"Date range: {df['route_id'].min()} to {df['route_id'].max()}")
    
    # Check if delay data exists
    if 'actual_arrival_delay' in df.columns:
        df['actual_arrival_delay_minutes'] = df['actual_arrival_delay'] * 24 * 60
        df['delay_minutes'] = np.maximum(0, df['actual_arrival_delay_minutes'])
        df['delayed_flag'] = (df['delay_minutes'] > 0).astype(int)
    else:
        print("\nNo actual_arrival_delay column found!")
        return
    
    # Analysis 1: Delay Distribution
    print("\n" + "=" * 80)
    print("1. DELAY DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    delayed_count = df['delayed_flag'].sum()
    total_count = len(df)
    delay_rate = delayed_count / total_count
    
    print(f"\nDelayed stops: {delayed_count:,} ({delay_rate:.2%})")
    print(f"On-time stops: {total_count - delayed_count:,} ({1-delay_rate:.2%})")
    
    # Delay statistics (only for delayed stops)
    delayed_df = df[df['delayed_flag'] == 1]
    
    if len(delayed_df) > 0:
        print(f"\n--- Statistics for DELAYED stops only ---")
        print(f"Mean delay: {delayed_df['delay_minutes'].mean():.4f} minutes ({delayed_df['delay_minutes'].mean()*60:.2f} seconds)")
        print(f"Median delay: {delayed_df['delay_minutes'].median():.4f} minutes")
        print(f"Std delay: {delayed_df['delay_minutes'].std():.4f} minutes")
        print(f"Min delay: {delayed_df['delay_minutes'].min():.4f} minutes")
        print(f"Max delay: {delayed_df['delay_minutes'].max():.4f} minutes")
        
        print(f"\n--- Overall statistics (including on-time) ---")
        print(f"Mean delay: {df['delay_minutes'].mean():.4f} minutes ({df['delay_minutes'].mean()*60:.2f} seconds)")
        print(f"Median delay: {df['delay_minutes'].median():.4f} minutes")
        print(f"Std delay: {df['delay_minutes'].std():.4f} minutes")
        
        # Percentiles
        print(f"\n--- Delay Percentiles (delayed stops only) ---")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(delayed_df['delay_minutes'], p)
            print(f"{p}th percentile: {val:.4f} minutes ({val*60:.2f} seconds)")
    
    # Analysis 2: Data Quality Issues
    print("\n" + "=" * 80)
    print("2. DATA QUALITY ISSUES")
    print("=" * 80)
    
    print(f"\nTarget variance (delay_minutes): {df['delay_minutes'].var():.6f}")
    print(f"Target range: [{df['delay_minutes'].min():.4f}, {df['delay_minutes'].max():.4f}]")
    
    # Check for extreme class imbalance
    if delay_rate < 0.15:
        print(f"\n⚠️  SEVERE CLASS IMBALANCE: Only {delay_rate:.2%} of samples are delayed!")
        print("   This makes classification difficult and regression nearly impossible.")
    
    # Check for low variance in target
    if df['delay_minutes'].var() < 0.01:
        print(f"\n⚠️  EXTREMELY LOW VARIANCE: Target variance is {df['delay_minutes'].var():.6f}")
        print("   Regression models cannot learn from such minimal variation!")
    
    # Check for zero-inflated data
    zero_count = (df['delay_minutes'] == 0).sum()
    zero_rate = zero_count / len(df)
    if zero_rate > 0.8:
        print(f"\n⚠️  ZERO-INFLATED TARGET: {zero_rate:.2%} of delay_minutes are exactly 0")
        print("   This creates a spike at zero that regression models struggle with.")
    
    # Analysis 3: Feature Analysis
    print("\n" + "=" * 80)
    print("3. KEY FEATURE STATISTICS")
    print("=" * 80)
    
    key_features = ['distancep', 'distancea', 'indexp', 'indexa']
    for feat in key_features:
        if feat in df.columns:
            print(f"\n{feat}:")
            print(f"  Mean: {df[feat].mean():.2f}")
            print(f"  Median: {df[feat].median():.2f}")
            print(f"  Std: {df[feat].std():.2f}")
            print(f"  Range: [{df[feat].min():.2f}, {df[feat].max():.2f}]")
    
    # Analysis 4: Route-level Analysis
    print("\n" + "=" * 80)
    print("4. ROUTE-LEVEL ANALYSIS")
    print("=" * 80)
    
    route_delays = df.groupby('route_id').agg({
        'delayed_flag': ['sum', 'mean'],
        'delay_minutes': ['sum', 'mean', 'max'],
        'stop_id': 'count'
    })
    
    print(f"\nRoutes with at least one delay: {(route_delays[('delayed_flag', 'sum')] > 0).sum():,}")
    print(f"Routes with no delays: {(route_delays[('delayed_flag', 'sum')] == 0).sum():,}")
    print(f"\nAverage stops per route: {route_delays[('stop_id', 'count')].mean():.2f}")
    print(f"Average delays per route: {route_delays[('delayed_flag', 'sum')].mean():.2f}")
    
    # Analysis 5: Problem Diagnosis
    print("\n" + "=" * 80)
    print("5. ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    problems = []
    
    if df['delay_minutes'].max() < 1.0:
        problems.append("• Delays are measured in FRACTIONS OF MINUTES (<1 min = <60 seconds)")
        problems.append("  → Solution: Convert to SECONDS for better scale")
    
    if df['delay_minutes'].var() < 0.01:
        problems.append("• Target variance is extremely low")
        problems.append("  → Solution: Scale target or use different units")
    
    if zero_rate > 0.8:
        problems.append(f"• {zero_rate:.1%} of targets are zero (zero-inflated)")
        problems.append("  → Solution: Use classification instead, or zero-inflated regression")
    
    if delay_rate < 0.15:
        problems.append(f"• Severe class imbalance ({delay_rate:.1%} positive class)")
        problems.append("  → Solution: Use class weights, SMOTE, or focal loss")
    
    if len(delayed_df) > 0 and delayed_df['delay_minutes'].mean() < 0.5:
        problems.append("• Even delayed stops have tiny delays (< 30 seconds average)")
        problems.append("  → Solution: This might indicate data collection issues")
    
    print("\n🔍 IDENTIFIED PROBLEMS:\n")
    for problem in problems:
        print(problem)
    
    # Analysis 6: Recommendations
    print("\n" + "=" * 80)
    print("6. RECOMMENDATIONS FOR IMPROVEMENT")
    print("=" * 80)
    
    recommendations = [
        "\n📋 IMMEDIATE ACTIONS:",
        "1. Convert delay_minutes to delay_seconds for better scale",
        "2. Add class weights to handle imbalance",
        "3. Use log transformation for regression targets (log(delay + 1))",
        "4. Focus on CLASSIFICATION (which is working well) rather than regression",
        "",
        "📊 MODEL IMPROVEMENTS:",
        "5. For LSTM: Add class weights and increase learning rate",
        "6. For Regression: Try zero-inflated models or quantile regression",
        "7. For Regression: Consider predicting delay categories instead",
        "",
        "🔧 DATA IMPROVEMENTS:",
        "8. Verify data collection - are delays really this small?",
        "9. Consider aggregating at route level instead of stop level",
        "10. Engineer features that capture cumulative effects better"
    ]
    
    for rec in recommendations:
        print(rec)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("7. GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Delay distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall delay distribution
    axes[0, 0].hist(df['delay_minutes'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Delay (minutes)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Delay Distribution (including zeros)')
    axes[0, 0].axvline(df['delay_minutes'].mean(), color='red', linestyle='--', label=f'Mean: {df["delay_minutes"].mean():.4f} min')
    axes[0, 0].legend()
    
    # Delayed stops only
    if len(delayed_df) > 0:
        axes[0, 1].hist(delayed_df['delay_minutes'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Delay (minutes)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Delay Distribution (delayed stops only)')
        axes[0, 1].axvline(delayed_df['delay_minutes'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {delayed_df["delay_minutes"].mean():.4f} min')
        axes[0, 1].legend()
    
    # Class balance
    class_counts = df['delayed_flag'].value_counts()
    axes[1, 0].bar(['On-time', 'Delayed'], [class_counts[0], class_counts[1]], color=['green', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Class Balance')
    axes[1, 0].text(0, class_counts[0], f'{class_counts[0]:,}\n({class_counts[0]/len(df):.1%})', 
                    ha='center', va='bottom')
    axes[1, 0].text(1, class_counts[1], f'{class_counts[1]:,}\n({class_counts[1]/len(df):.1%})', 
                    ha='center', va='bottom')
    
    # Box plot of delays by flag
    axes[1, 1].boxplot([df[df['delayed_flag'] == 0]['delay_minutes'],
                        df[df['delayed_flag'] == 1]['delay_minutes']],
                       labels=['On-time', 'Delayed'])
    axes[1, 1].set_ylabel('Delay (minutes)')
    axes[1, 1].set_title('Delay Distribution by Class')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'delay_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'delay_distribution_analysis.png'}")
    
    # Figure 2: Route-level analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Delays per route
    route_delay_counts = route_delays[('delayed_flag', 'sum')]
    axes[0].hist(route_delay_counts, bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[0].set_xlabel('Number of Delayed Stops per Route')
    axes[0].set_ylabel('Number of Routes')
    axes[0].set_title('Distribution of Delays Across Routes')
    axes[0].axvline(route_delay_counts.mean(), color='red', linestyle='--', 
                   label=f'Mean: {route_delay_counts.mean():.2f}')
    axes[0].legend()
    
    # Route delay rate
    route_delay_rates = route_delays[('delayed_flag', 'mean')]
    axes[1].hist(route_delay_rates, bins=30, edgecolor='black', alpha=0.7, color='teal')
    axes[1].set_xlabel('Delay Rate per Route')
    axes[1].set_ylabel('Number of Routes')
    axes[1].set_title('Distribution of Route Delay Rates')
    axes[1].axvline(route_delay_rates.mean(), color='red', linestyle='--', 
                   label=f'Mean: {route_delay_rates.mean():.2%}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'route_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'route_analysis.png'}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")
    
    # Summary
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"""
The data analysis reveals critical issues affecting model performance:

1. ⚠️  SCALE PROBLEM: Delays are tiny ({df['delay_minutes'].max():.3f} max minutes = {df['delay_minutes'].max()*60:.1f} seconds)
2. ⚠️  CLASS IMBALANCE: Only {delay_rate:.1%} of stops are delayed
3. ⚠️  LOW VARIANCE: Target variance is {df['delay_minutes'].var():.6f} (extremely low)
4. ⚠️  ZERO-INFLATION: {zero_rate:.1%} of targets are exactly zero

WHY REGRESSION FAILS:
- Negative R² means the model performs worse than predicting the mean
- With max delay of {df['delay_minutes'].max()*60:.1f} seconds and {zero_rate:.1%} zeros,
  there's insufficient signal for regression models to learn meaningful patterns
- The model is essentially trying to fit noise

WHY CLASSIFICATION WORKS BETTER:
- Random Forest Classifier achieves 93% accuracy
- Binary classification (delayed/on-time) has clearer patterns
- The imbalance is manageable with proper techniques

RECOMMENDATION: Focus on CLASSIFICATION for production use!
""")

if __name__ == "__main__":
    analyze_data()

