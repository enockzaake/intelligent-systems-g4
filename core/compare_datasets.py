"""
Compare Real vs Synthetic Data
Analyze differences and validate synthetic data quality
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_and_prepare_data(file_path):
    """Load and prepare dataset for comparison"""
    df = pd.read_csv(file_path)
    
    # Convert delay if needed
    if 'actual_arrival_delay' in df.columns:
        df['delay_minutes'] = df['actual_arrival_delay'] * 24 * 60
        df['delayed_flag'] = (df['delay_minutes'] > 0).astype(int)
    
    return df


def compare_datasets(real_path, synthetic_path):
    """Compare real and synthetic datasets"""
    
    print("\n" + "=" * 80)
    print("DATASET COMPARISON: Real vs Synthetic")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading datasets...")
    real_df = load_and_prepare_data(real_path)
    synth_df = load_and_prepare_data(synthetic_path)
    
    print(f"  Real data: {len(real_df):,} records")
    print(f"  Synthetic data: {len(synth_df):,} records")
    
    # Basic statistics comparison
    print("\n" + "=" * 80)
    print("1. BASIC STATISTICS")
    print("=" * 80)
    
    stats_comparison = []
    
    # Dataset size
    stats_comparison.append({
        'Metric': 'Total Records',
        'Real': f"{len(real_df):,}",
        'Synthetic': f"{len(synth_df):,}",
        'Difference': f"{len(synth_df) - len(real_df):+,}"
    })
    
    # Number of routes
    stats_comparison.append({
        'Metric': 'Number of Routes',
        'Real': f"{real_df['route_id'].nunique():,}",
        'Synthetic': f"{synth_df['route_id'].nunique():,}",
        'Difference': '-'
    })
    
    # Avg stops per route
    real_avg_stops = len(real_df) / real_df['route_id'].nunique()
    synth_avg_stops = len(synth_df) / synth_df['route_id'].nunique()
    stats_comparison.append({
        'Metric': 'Avg Stops/Route',
        'Real': f"{real_avg_stops:.1f}",
        'Synthetic': f"{synth_avg_stops:.1f}",
        'Difference': f"{synth_avg_stops - real_avg_stops:+.1f}"
    })
    
    comparison_df = pd.DataFrame(stats_comparison)
    print("\n" + comparison_df.to_string(index=False))
    
    # Delay statistics
    print("\n" + "=" * 80)
    print("2. DELAY STATISTICS")
    print("=" * 80)
    
    delay_stats = []
    
    # Delay rate
    real_delay_rate = real_df['delayed_flag'].mean()
    synth_delay_rate = synth_df['delayed_flag'].mean()
    delay_stats.append({
        'Metric': 'Delay Rate',
        'Real': f"{real_delay_rate:.2%}",
        'Synthetic': f"{synth_delay_rate:.2%}",
        'Difference': f"{synth_delay_rate - real_delay_rate:+.2%}"
    })
    
    # Get delayed stops only
    real_delayed = real_df[real_df['delayed_flag'] == 1]
    synth_delayed = synth_df[synth_df['delayed_flag'] == 1]
    
    if len(real_delayed) > 0 and len(synth_delayed) > 0:
        # Mean delay
        delay_stats.append({
            'Metric': 'Mean Delay (min)',
            'Real': f"{real_delayed['delay_minutes'].mean():.2f}",
            'Synthetic': f"{synth_delayed['delay_minutes'].mean():.2f}",
            'Difference': f"{synth_delayed['delay_minutes'].mean() - real_delayed['delay_minutes'].mean():+.2f}"
        })
        
        # Median delay
        delay_stats.append({
            'Metric': 'Median Delay (min)',
            'Real': f"{real_delayed['delay_minutes'].median():.2f}",
            'Synthetic': f"{synth_delayed['delay_minutes'].median():.2f}",
            'Difference': f"{synth_delayed['delay_minutes'].median() - real_delayed['delay_minutes'].median():+.2f}"
        })
        
        # Max delay
        delay_stats.append({
            'Metric': 'Max Delay (min)',
            'Real': f"{real_delayed['delay_minutes'].max():.2f}",
            'Synthetic': f"{synth_delayed['delay_minutes'].max():.2f}",
            'Difference': f"{synth_delayed['delay_minutes'].max() - real_delayed['delay_minutes'].max():+.2f}"
        })
        
        # Variance
        delay_stats.append({
            'Metric': 'Variance',
            'Real': f"{real_df['delay_minutes'].var():.6f}",
            'Synthetic': f"{synth_df['delay_minutes'].var():.6f}",
            'Difference': f"{synth_df['delay_minutes'].var() - real_df['delay_minutes'].var():+.6f}"
        })
    
    delay_df = pd.DataFrame(delay_stats)
    print("\n" + delay_df.to_string(index=False))
    
    # Data quality assessment
    print("\n" + "=" * 80)
    print("3. DATA QUALITY ASSESSMENT")
    print("=" * 80)
    
    print("\n📊 Real Data:")
    real_zero_rate = (real_df['delay_minutes'] == 0).mean()
    print(f"  Zero delays: {real_zero_rate:.2%}")
    print(f"  Variance: {real_df['delay_minutes'].var():.6f}")
    if real_df['delay_minutes'].var() < 0.01:
        print(f"  Status: ❌ TOO LOW - Models will struggle")
    else:
        print(f"  Status: ✅ Good")
    
    print("\n📊 Synthetic Data:")
    synth_zero_rate = (synth_df['delay_minutes'] == 0).mean()
    print(f"  Zero delays: {synth_zero_rate:.2%}")
    print(f"  Variance: {synth_df['delay_minutes'].var():.6f}")
    if synth_df['delay_minutes'].var() < 0.01:
        print(f"  Status: ⚠️  TOO LOW - Models will struggle")
    else:
        print(f"  Status: ✅ Good - Models can learn!")
    
    # Feature distribution comparison
    print("\n" + "=" * 80)
    print("4. FEATURE DISTRIBUTIONS")
    print("=" * 80)
    
    common_features = ['indexp', 'indexa', 'distancep', 'distancea', 'depot', 'delivery']
    
    for feature in common_features:
        if feature in real_df.columns and feature in synth_df.columns:
            real_mean = real_df[feature].mean()
            synth_mean = synth_df[feature].mean()
            print(f"\n  {feature}:")
            print(f"    Real: mean={real_mean:.2f}, std={real_df[feature].std():.2f}")
            print(f"    Synth: mean={synth_mean:.2f}, std={synth_df[feature].std():.2f}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("5. GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Delay distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Real vs Synthetic Data Comparison', fontsize=16, fontweight='bold')
    
    # Delay distribution (all stops)
    ax = axes[0, 0]
    ax.hist(real_df['delay_minutes'], bins=50, alpha=0.6, label='Real', color='blue', edgecolor='black')
    ax.hist(synth_df['delay_minutes'], bins=50, alpha=0.6, label='Synthetic', color='green', edgecolor='black')
    ax.set_xlabel('Delay (minutes)')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Delay Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Delay distribution (delayed stops only)
    ax = axes[0, 1]
    if len(real_delayed) > 0 and len(synth_delayed) > 0:
        ax.hist(real_delayed['delay_minutes'], bins=50, alpha=0.6, label='Real', color='blue', edgecolor='black')
        ax.hist(synth_delayed['delay_minutes'], bins=50, alpha=0.6, label='Synthetic', color='green', edgecolor='black')
        ax.set_xlabel('Delay (minutes)')
        ax.set_ylabel('Frequency')
        ax.set_title('Delay Distribution (Delayed Stops Only)')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Box plot comparison
    ax = axes[1, 0]
    data_to_plot = [real_df['delay_minutes'], synth_df['delay_minutes']]
    ax.boxplot(data_to_plot, labels=['Real', 'Synthetic'])
    ax.set_ylabel('Delay (minutes)')
    ax.set_title('Delay Distribution Box Plot')
    ax.grid(alpha=0.3)
    
    # Delay rate comparison
    ax = axes[1, 1]
    categories = ['Real', 'Synthetic']
    delay_rates = [real_delay_rate * 100, synth_delay_rate * 100]
    bars = ax.bar(categories, delay_rates, color=['blue', 'green'], alpha=0.7)
    ax.set_ylabel('Delay Rate (%)')
    ax.set_title('Delay Rate Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, delay_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_vs_synthetic_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir / 'real_vs_synthetic_comparison.png'}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("6. RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n🎯 For Model Training:")
    
    if synth_df['delay_minutes'].var() > real_df['delay_minutes'].var() * 10:
        print("  ✅ Synthetic data has MUCH better variance")
        print("  ✅ Models should learn better patterns")
        print("  ✅ Recommended: Use synthetic data for training")
    elif synth_df['delay_minutes'].var() > 0.01:
        print("  ✅ Synthetic data has sufficient variance")
        print("  ✅ Recommended: Use synthetic data for training")
    else:
        print("  ⚠️  Synthetic data still has low variance")
        print("  ⚠️  May need to regenerate with different parameters")
    
    if abs(synth_delay_rate - real_delay_rate) < 0.05:
        print("  ✅ Delay rates are similar (good)")
    else:
        print(f"  ⚠️  Delay rates differ by {abs(synth_delay_rate - real_delay_rate):.2%}")
        print("  💡 This is often okay if variance is better")
    
    print("\n📊 Expected Model Performance:")
    if synth_df['delay_minutes'].var() > 0.1:
        print("  ✅ Classification: Should achieve >80% F1-score")
        print("  ✅ Regression: Should achieve positive R²")
        print("  ✅ LSTM: Should achieve >50% recall")
    elif synth_df['delay_minutes'].var() > 0.01:
        print("  ✅ Classification: Should achieve >70% F1-score")
        print("  ⚠️  Regression: May still struggle")
        print("  ✅ LSTM: Should achieve >40% recall")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


def main():
    """Run comparison"""
    
    real_path = Path("data/cleaned_delivery_data.csv")
    synthetic_path = Path("data/synthetic_delivery_data.csv")
    
    if not real_path.exists():
        print(f"❌ Real data not found: {real_path}")
        print("   Using current dataset location...")
        return
    
    if not synthetic_path.exists():
        print(f"❌ Synthetic data not found: {synthetic_path}")
        print("   Please generate it first:")
        print("   python generate_synthetic_data.py")
        return
    
    compare_datasets(real_path, synthetic_path)


if __name__ == "__main__":
    main()

