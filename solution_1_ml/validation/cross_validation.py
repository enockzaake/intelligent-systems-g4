"""
Cross-Validation Module
Performs k-fold cross-validation with route-aware grouping and confidence intervals
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import DataPreprocessor
from models import LogisticRegressionModel, RandomForestModel


class CrossValidator:
    """
    Performs k-fold cross-validation with proper route grouping
    """
    
    def __init__(self, data_path, output_dir="outputs/cross_validation"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = DataPreprocessor(data_path)
        self.results = {}
    
    def cross_validate_model(self, model_class, model_params, X, y, groups, n_splits=5):
        """
        Perform k-fold cross-validation ensuring routes stay together
        
        Args:
            model_class: Model class to instantiate
            model_params: Parameters for model
            X: Feature matrix
            y: Target vector
            groups: Route IDs for grouping
            n_splits: Number of folds
        
        Returns:
            Dictionary with metrics arrays and statistics
        """
        group_kfold = GroupKFold(n_splits=n_splits)
        
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': []
        }
        
        fold_num = 1
        
        for train_idx, test_idx in group_kfold.split(X, y, groups):
            print(f"  Fold {fold_num}/{n_splits}...")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = model_class(**model_params)
            model.train(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Get probabilities
            y_proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                if proba is not None and len(proba.shape) > 1:
                    y_proba = proba[:, 1]
            
            # Calculate metrics
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            scores['f1_score'].append(f1_score(y_test, y_pred, zero_division=0))
            
            if y_proba is not None:
                scores['roc_auc'].append(roc_auc_score(y_test, y_proba))
            
            fold_num += 1
        
        # Calculate statistics
        statistics = {}
        for metric, values in scores.items():
            if len(values) > 0:
                statistics[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': [float(v) for v in values],
                    'ci_95_lower': float(np.percentile(values, 2.5)),
                    'ci_95_upper': float(np.percentile(values, 97.5))
                }
        
        return statistics
    
    def run_cross_validation(self, n_splits=5):
        """
        Run cross-validation for all models
        """
        print(f"\n{'='*80}")
        print(f"K-FOLD CROSS-VALIDATION (k={n_splits})")
        print(f"{'='*80}")
        
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        df = self.preprocessor.load_data()
        df = self.preprocessor.create_features(df)
        df = self.preprocessor.encode_categorical(df, fit=True)
        
        # Prepare features
        X, y, feature_cols = self.preprocessor.prepare_features(
            df, 'delayed_flag', fit_scaler=True
        )
        
        # Get route groups
        groups = df['route_id'].values
        
        print(f"Total samples: {len(X):,}")
        print(f"Total routes: {len(np.unique(groups)):,}")
        print(f"Features: {len(feature_cols)}")
        
        # Models to test
        models = {
            'Logistic Regression': {
                'class': LogisticRegressionModel,
                'params': {'max_iter': 1000, 'random_state': 42}
            },
            'Random Forest': {
                'class': RandomForestModel,
                'params': {
                    'task': 'classification',
                    'n_estimators': 200,
                    'max_depth': 20,
                    'random_state': 42
                }
            }
        }
        
        # Cross-validate each model
        for model_name, model_config in models.items():
            print(f"\n{'='*80}")
            print(f"Cross-validating: {model_name}")
            print(f"{'='*80}")
            
            statistics = self.cross_validate_model(
                model_config['class'],
                model_config['params'],
                X, y, groups, n_splits
            )
            
            self.results[model_name] = statistics
            
            # Print results
            print(f"\nResults for {model_name}:")
            print("-" * 60)
            for metric, stats in statistics.items():
                print(f"  {metric}:")
                print(f"    Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"    95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]")
                print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Save results
        self.save_results()
        
        # Create visualizations
        self.create_visualizations()
        
        return self.results
    
    def save_results(self):
        """Save cross-validation results"""
        results_file = self.output_dir / "cross_validation_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Cross-validation results saved to: {results_file}")
        
        # Create detailed report
        report_file = self.output_dir / "cross_validation_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CROSS-VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write("K-Fold Cross-Validation with Route-Aware Grouping\n")
            f.write("Ensures no route appears in both train and test sets\n\n")
            
            for model_name, statistics in self.results.items():
                f.write(f"\n{model_name}:\n")
                f.write("="*60 + "\n\n")
                
                for metric, stats in statistics.items():
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  95% Confidence Interval: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]\n")
                    f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                    f.write(f"  Individual folds: {', '.join([f'{v:.4f}' for v in stats['values']])}\n\n")
        
        print(f"✅ Cross-validation report saved to: {report_file}")
    
    def create_visualizations(self):
        """Create visualization of cross-validation results"""
        print("\nCreating visualizations...")
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Validation Results with 95% Confidence Intervals', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            models = list(self.results.keys())
            means = [self.results[m][metric]['mean'] for m in models]
            stds = [self.results[m][metric]['std'] for m in models]
            ci_lowers = [self.results[m][metric]['ci_95_lower'] for m in models]
            ci_uppers = [self.results[m][metric]['ci_95_upper'] for m in models]
            
            x_pos = np.arange(len(models))
            
            # Plot bars with error bars
            bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
                          color=['blue', 'green'], edgecolor='black', linewidth=1.5)
            
            # Add confidence interval lines
            for i, (lower, upper) in enumerate(zip(ci_lowers, ci_uppers)):
                ax.plot([i-0.2, i+0.2], [lower, lower], 'r-', linewidth=2)
                ax.plot([i-0.2, i+0.2], [upper, upper], 'r-', linewidth=2)
                ax.plot([i, i], [lower, upper], 'r--', linewidth=1.5, alpha=0.5)
            
            # Add value labels
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                       f'{mean:.3f}±{std:.3f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} (with 95% CI in red)', 
                        fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        viz_file = self.output_dir / "cross_validation_results.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {viz_file}")
        
        plt.close()
        
        # Box plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Validation Score Distributions', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            data_to_plot = []
            labels = []
            
            for model_name in self.results.keys():
                data_to_plot.append(self.results[model_name][metric]['values'])
                labels.append(model_name)
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)
            
            # Color boxes
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution Across Folds', 
                        fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(labels, rotation=15, ha='right')
        
        plt.tight_layout()
        
        boxplot_file = self.output_dir / "cross_validation_boxplots.png"
        plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Box plots saved to: {boxplot_file}")
        
        plt.close()


def main():
    """Run cross-validation"""
    data_path = Path("data/synthetic_delivery_data.csv")
    
    if not data_path.exists():
        data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    validator = CrossValidator(str(data_path))
    results = validator.run_cross_validation(n_splits=5)
    
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION COMPLETE")
    print(f"{'='*80}")
    print("\nResults show model performance with confidence intervals")
    print("Route-aware grouping prevents data leakage between folds")


if __name__ == "__main__":
    main()

