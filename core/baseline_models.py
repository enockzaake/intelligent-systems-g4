"""
Baseline Models Module
Implements simple baseline models for comparison
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

from data_preprocessing import DataPreprocessor


class MajorityClassBaseline:
    """Always predicts the majority class"""
    
    def __init__(self):
        self.majority_class = None
    
    def train(self, X, y):
        """Learn the majority class"""
        self.majority_class = np.bincount(y).argmax()
        return self
    
    def predict(self, X):
        """Predict majority class for all samples"""
        return np.full(len(X), self.majority_class)
    
    def predict_proba(self, X):
        """Return uniform probabilities"""
        if self.majority_class == 0:
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
        else:
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


class RouteMeanBaseline:
    """Predicts based on historical route mean delay rate"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.route_means = {}
        self.global_mean = 0.5
    
    def train(self, X, y, route_ids):
        """Learn mean delay rate per route"""
        df = pd.DataFrame({'route_id': route_ids, 'delayed': y})
        self.route_means = df.groupby('route_id')['delayed'].mean().to_dict()
        self.global_mean = y.mean()
        return self
    
    def predict_proba_for_routes(self, route_ids):
        """Get delay probabilities for routes"""
        probs = []
        for route_id in route_ids:
            prob = self.route_means.get(route_id, self.global_mean)
            probs.append(prob)
        return np.array(probs)
    
    def predict(self, X, route_ids):
        """Predict based on route mean"""
        probs = self.predict_proba_for_routes(route_ids)
        return (probs > self.threshold).astype(int)


class RuleBasedBaseline:
    """Simple rule-based predictor using domain knowledge"""
    
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.thresholds = {
            'distance_deviation': 0.2,
            'time_window_length': 45,
            'prev_stop_delay': 5,
            'hour_of_arrival_rush': [7, 8, 17, 18],
            'stop_position_norm': 0.7
        }
    
    def train(self, X, y):
        """No training needed for rule-based"""
        return self
    
    def get_feature_index(self, feature_name):
        """Get index of feature in X matrix"""
        try:
            return self.feature_columns.index(feature_name)
        except ValueError:
            return None
    
    def predict(self, X):
        """Apply rules to predict delays"""
        predictions = []
        
        # Get feature indices
        distance_dev_idx = self.get_feature_index('distance_deviation')
        time_window_idx = self.get_feature_index('time_window_length')
        prev_delay_idx = self.get_feature_index('prev_stop_delay')
        hour_idx = self.get_feature_index('hour_of_arrival')
        position_idx = self.get_feature_index('stop_position_norm')
        
        for i in range(len(X)):
            score = 0
            
            # Rule 1: High distance deviation
            if distance_dev_idx is not None and X[i, distance_dev_idx] > self.thresholds['distance_deviation']:
                score += 1
            
            # Rule 2: Tight time window
            if time_window_idx is not None and X[i, time_window_idx] < self.thresholds['time_window_length']:
                score += 1
            
            # Rule 3: Previous stop was delayed
            if prev_delay_idx is not None and X[i, prev_delay_idx] > self.thresholds['prev_stop_delay']:
                score += 1
            
            # Rule 4: Rush hour
            if hour_idx is not None and X[i, hour_idx] in self.thresholds['hour_of_arrival_rush']:
                score += 1
            
            # Rule 5: Late in route
            if position_idx is not None and X[i, position_idx] > self.thresholds['stop_position_norm']:
                score += 1
            
            # Predict delay if score >= 2
            predictions.append(1 if score >= 2 else 0)
        
        return np.array(predictions)


class BaselineComparison:
    """
    Compares all baseline models with ML models
    """
    
    def __init__(self, data_path, output_dir="outputs/baseline_comparison"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = DataPreprocessor(data_path)
        self.results = {}
    
    def evaluate_baseline(self, baseline_name, baseline_model, X_train, y_train, 
                         X_test, y_test, route_ids_train=None, route_ids_test=None):
        """
        Evaluate a baseline model
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING: {baseline_name}")
        print(f"{'='*80}")
        
        # Train
        if isinstance(baseline_model, RouteMeanBaseline):
            baseline_model.train(X_train, y_train, route_ids_train)
            y_pred = baseline_model.predict(X_test, route_ids_test)
        else:
            baseline_model.train(X_train, y_train)
            y_pred = baseline_model.predict(X_test)
        
        # Evaluate
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'complexity': 'Trivial' if isinstance(baseline_model, MajorityClassBaseline) else 'Simple'
        }
        
        # Print results
        print(f"\nResults:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        return metrics
    
    def run_baseline_comparison(self):
        """
        Run complete baseline comparison
        """
        print(f"\n{'='*80}")
        print("BASELINE MODEL COMPARISON")
        print(f"{'='*80}")
        
        # Load and preprocess data
        print("\nLoading data...")
        df = self.preprocessor.load_data()
        df = self.preprocessor.create_features(df)
        df = self.preprocessor.encode_categorical(df, fit=True)
        
        # Split data
        from data_preprocessing import DataPreprocessor
        train_df, test_df = self.preprocessor.split_by_routes(df, test_size=0.2, random_state=42)
        
        # Prepare features
        X_train, y_train, feature_cols = self.preprocessor.prepare_features(
            train_df, 'delayed_flag', fit_scaler=True
        )
        X_test, y_test, _ = self.preprocessor.prepare_features(
            test_df, 'delayed_flag', fit_scaler=False
        )
        
        # Get route IDs
        route_ids_train = train_df['route_id'].values
        route_ids_test = test_df['route_id'].values
        
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Train delay rate: {y_train.mean():.2%}")
        print(f"Test delay rate: {y_test.mean():.2%}")
        
        # Test baselines
        baselines = {
            'Majority Class': MajorityClassBaseline(),
            'Route Mean': RouteMeanBaseline(threshold=0.5),
            'Rule-Based': RuleBasedBaseline(feature_cols)
        }
        
        for baseline_name, baseline_model in baselines.items():
            metrics = self.evaluate_baseline(
                baseline_name, baseline_model,
                X_train, y_train, X_test, y_test,
                route_ids_train, route_ids_test
            )
            self.results[baseline_name] = metrics
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save baseline comparison results"""
        results_file = self.output_dir / "baseline_comparison_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Baseline results saved to: {results_file}")
        
        # Create report
        report_file = self.output_dir / "baseline_comparison_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BASELINE MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            f.write("Comparing simple baseline models with ML models\n")
            f.write("Baselines establish minimum performance requirements\n\n")
            
            # Sort by F1-score
            sorted_results = sorted(
                self.results.items(),
                key=lambda x: x[1]['f1_score'],
                reverse=True
            )
            
            f.write("RESULTS (sorted by F1-score):\n")
            f.write("-" * 80 + "\n\n")
            
            for model_name, metrics in sorted_results:
                f.write(f"{model_name}:\n")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric:15s}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric:15s}: {value}\n")
                f.write("\n")
        
        print(f"✅ Baseline report saved to: {report_file}")
    
    def compare_with_ml_models(self, ml_results):
        """
        Compare baseline results with ML model results
        """
        print(f"\n{'='*80}")
        print("BASELINE VS ML MODEL COMPARISON")
        print(f"{'='*80}\n")
        
        # Combine all results
        all_results = {**self.results, **ml_results}
        
        # Create comparison table
        comparison = []
        
        for model_name, metrics in all_results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'Complexity': metrics.get('complexity', 'Medium/High')
            })
        
        # Sort by F1-score
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("\nComparison Table:")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("\n")
        
        # Calculate improvements
        best_baseline_f1 = max([m['f1_score'] for m in self.results.values()])
        
        print("ML Model Improvements Over Best Baseline:")
        print("-"*60)
        
        for model_name, metrics in ml_results.items():
            ml_f1 = metrics.get('f1_score', 0)
            improvement = ((ml_f1 - best_baseline_f1) / best_baseline_f1) * 100
            
            print(f"  {model_name}: +{improvement:.1f}% improvement")
        
        # Save comparison
        comparison_file = self.output_dir / "baseline_vs_ml_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"\n✅ Comparison saved to: {comparison_file}")
        
        return comparison_df


def main():
    """Run baseline comparison"""
    data_path = Path("data/synthetic_delivery_data.csv")
    
    if not data_path.exists():
        data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    comparison = BaselineComparison(str(data_path))
    results = comparison.run_baseline_comparison()
    
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON COMPLETE")
    print(f"{'='*80}")
    print("\nBaseline models establish minimum performance requirements")
    print("ML models must significantly outperform these simple approaches")


if __name__ == "__main__":
    main()

