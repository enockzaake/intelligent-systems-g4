"""
Quick comparison script to visualize original vs improved model results
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(results_path):
    """Load evaluation results JSON"""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {results_path} not found")
        return None

def compare_results():
    """Compare original vs improved model results"""
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Load results
    original_dir = Path("outputs/results")
    improved_dir = Path("outputs_improved/results")
    
    # Find the latest results files
    original_files = list(original_dir.glob("evaluation_results_*.json"))
    improved_files = list(improved_dir.glob("evaluation_results_*.json"))
    
    if not original_files:
        print("\n❌ Original results not found. Run: python train.py")
        return
    
    if not improved_files:
        print("\n❌ Improved results not found. Run: python train_improved.py")
        return
    
    # Load latest files
    original_results = load_results(sorted(original_files)[-1])
    improved_results = load_results(sorted(improved_files)[-1])
    
    # Compare classification models
    print("\n" + "=" * 80)
    print("CLASSIFICATION MODELS COMPARISON")
    print("=" * 80)
    
    classification_models = [
        "Logistic Regression",
        "Random Forest Classifier", 
        "LSTM Classifier"
    ]
    
    comparison_data = []
    
    for model_name in classification_models:
        # Check both exact name and with "(Improved)" suffix
        orig_name = model_name
        imp_name = f"{model_name} (Improved)"
        
        if orig_name in original_results:
            orig = original_results[orig_name]
        else:
            print(f"Warning: {orig_name} not in original results")
            continue
            
        if imp_name in improved_results:
            imp = improved_results[imp_name]
        elif model_name in improved_results:
            imp = improved_results[model_name]
        else:
            print(f"Warning: {model_name} not in improved results")
            continue
        
        # Calculate improvements
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        print(f"\n📊 {model_name}")
        print("-" * 60)
        
        for metric in metrics:
            if metric in orig and metric in imp:
                orig_val = orig[metric]
                imp_val = imp[metric]
                diff = imp_val - orig_val
                pct_change = (diff / orig_val * 100) if orig_val > 0 else 0
                
                # Determine if improvement or degradation
                if diff > 0:
                    symbol = "✅"
                    color = "green"
                elif diff < 0:
                    symbol = "❌"
                    color = "red"
                else:
                    symbol = "➖"
                    color = "gray"
                
                print(f"  {metric:12} | Original: {orig_val:.4f} | Improved: {imp_val:.4f} | "
                      f"Δ {diff:+.4f} ({pct_change:+.1f}%) {symbol}")
                
                comparison_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Original': orig_val,
                    'Improved': imp_val,
                    'Change': diff,
                    'Pct_Change': pct_change
                })
        
        # Show confusion matrices
        if 'confusion_matrix' in orig and 'confusion_matrix' in imp:
            orig_cm = orig['confusion_matrix']
            imp_cm = imp['confusion_matrix']
            
            print(f"\n  Confusion Matrix Comparison:")
            print(f"  {'':20} | Original        | Improved")
            print(f"  {'-'*20} | {'-'*15} | {'-'*15}")
            print(f"  {'True Negatives':20} | {orig_cm[0][0]:>6,}         | {imp_cm[0][0]:>6,}")
            print(f"  {'False Positives':20} | {orig_cm[0][1]:>6,}         | {imp_cm[0][1]:>6,}")
            print(f"  {'False Negatives':20} | {orig_cm[1][0]:>6,}         | {imp_cm[1][0]:>6,}  {'⚠️ MINIMIZE' if orig_cm[1][0] > 500 else ''}")
            print(f"  {'True Positives':20} | {orig_cm[1][1]:>6,}         | {imp_cm[1][1]:>6,}  {'✅ MAXIMIZE' if imp_cm[1][1] > orig_cm[1][1] else ''}")
    
    # Create visualization
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Performance: Original vs Improved', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 3
            col = idx % 3
            
            metric_data = df[df['Metric'] == metric]
            
            if not metric_data.empty:
                x = range(len(metric_data))
                width = 0.35
                
                axes[row, col].bar([i - width/2 for i in x], metric_data['Original'], 
                                  width, label='Original', alpha=0.8, color='#FF6B6B')
                axes[row, col].bar([i + width/2 for i in x], metric_data['Improved'], 
                                  width, label='Improved', alpha=0.8, color='#4ECDC4')
                
                axes[row, col].set_ylabel('Score')
                axes[row, col].set_title(metric.replace('_', ' ').title())
                axes[row, col].set_xticks(x)
                axes[row, col].set_xticklabels([m[:15] for m in metric_data['Model']], rotation=15, ha='right')
                axes[row, col].legend()
                axes[row, col].set_ylim(0, 1.0)
                axes[row, col].grid(axis='y', alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        output_path = Path("outputs/analysis")
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / 'original_vs_improved_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n\n✅ Visualization saved to: {output_path / 'original_vs_improved_comparison.png'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Calculate averages
    if comparison_data:
        avg_changes = df.groupby('Metric')['Change'].mean()
        
        print("\n📈 Average Changes by Metric:")
        for metric, change in avg_changes.items():
            direction = "⬆️" if change > 0 else "⬇️" if change < 0 else "➖"
            print(f"  {metric:12}: {change:+.4f} {direction}")
        
        # Highlight key improvements
        print("\n🎯 Key Takeaways:")
        
        # Check LSTM recall improvement
        lstm_recall = df[(df['Model'] == 'LSTM Classifier') & (df['Metric'] == 'recall')]
        if not lstm_recall.empty:
            orig_recall = lstm_recall['Original'].values[0]
            imp_recall = lstm_recall['Improved'].values[0]
            
            if imp_recall > orig_recall:
                improvement = (imp_recall - orig_recall) / orig_recall * 100
                print(f"  ✅ LSTM Recall improved from {orig_recall:.1%} to {imp_recall:.1%} (+{improvement:.1f}%)")
                if imp_recall > 0.5:
                    print(f"     🎉 LSTM is now actually detecting delays!")
            else:
                print(f"  ⚠️  LSTM Recall needs more work: {imp_recall:.1%}")
        
        # Check overall F1 improvements
        f1_improvements = df[df['Metric'] == 'f1_score']['Change'].mean()
        if f1_improvements > 0:
            print(f"  ✅ Average F1-Score improved by {f1_improvements:.4f}")
        
        # Best model
        best_f1 = df[df['Metric'] == 'f1_score'].nlargest(1, 'Improved')
        if not best_f1.empty:
            print(f"  🏆 Best Model: {best_f1['Model'].values[0]} (F1: {best_f1['Improved'].values[0]:.4f})")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    compare_results()

