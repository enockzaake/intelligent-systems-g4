"""
Temporal Validation Module
Validates models using time-based train-test splits to ensure future prediction capability
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from datetime import datetime

from data_preprocessing import DataPreprocessor
from models import LogisticRegressionModel, RandomForestModel, LSTMModel


class TemporalValidator:
    """
    Performs temporal validation to test model performance on future data
    """
    
    def __init__(self, data_path, output_dir="outputs/temporal_validation"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = DataPreprocessor(data_path)
        self.results = {}
    
    def temporal_train_test_split(self, df, test_weeks=4):
        """
        Split data temporally: train on older data, test on recent data
        
        Args:
            df: DataFrame with week_id column
            test_weeks: Number of weeks to use for testing
        
        Returns:
            train_df, test_df
        """
        print(f"\n{'='*80}")
        print("TEMPORAL TRAIN-TEST SPLIT")
        print(f"{'='*80}")
        
        # Sort by week
        df = df.sort_values('week_id').reset_index(drop=True)
        
        # Get unique weeks
        unique_weeks = sorted(df['week_id'].unique())
        print(f"\nTotal weeks in dataset: {len(unique_weeks)}")
        print(f"Week range: {unique_weeks[0]} to {unique_weeks[-1]}")
        
        # Calculate split point
        split_week = unique_weeks[-test_weeks] if len(unique_weeks) > test_weeks else unique_weeks[len(unique_weeks)//2]
        
        # Split data
        train_df = df[df['week_id'] < split_week].copy()
        test_df = df[df['week_id'] >= split_week].copy()
        
        print(f"\nSplit week: {split_week}")
        print(f"Training data: weeks {train_df['week_id'].min()} to {train_df['week_id'].max()}")
        print(f"Test data: weeks {test_df['week_id'].min()} to {test_df['week_id'].max()}")
        print(f"\nTraining samples: {len(train_df):,}")
        print(f"Test samples: {len(test_df):,}")
        print(f"Train delay rate: {train_df['delayed_flag'].mean():.2%}")
        print(f"Test delay rate: {test_df['delayed_flag'].mean():.2%}")
        
        return train_df, test_df
    
    def evaluate_model_temporal(self, model_name, model, X_train, y_train, X_test, y_test):
        """
        Train and evaluate model with temporal split
        """
        print(f"\n{'='*80}")
        print(f"TEMPORAL VALIDATION: {model_name}")
        print(f"{'='*80}")
        
        # Train
        print("\nTraining on historical data...")
        model.train(X_train, y_train)
        
        # Predict on future data
        print("Predicting on future data...")
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            if proba is not None and len(proba.shape) > 1:
                y_proba = proba[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        # Print results
        print("\nTemporal Validation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def run_full_temporal_validation(self, test_weeks=4):
        """
        Run complete temporal validation for all models
        """
        print(f"\n{'='*80}")
        print("FULL TEMPORAL VALIDATION")
        print(f"{'='*80}")
        
        # Load and preprocess data
        print("\nLoading data...")
        df = self.preprocessor.load_data()
        df = self.preprocessor.create_features(df)
        df = self.preprocessor.encode_categorical(df, fit=True)
        
        # Temporal split
        train_df, test_df = self.temporal_train_test_split(df, test_weeks)
        
        # Prepare features
        print("\nPreparing features...")
        X_train, y_train, feature_cols = self.preprocessor.prepare_features(
            train_df, 'delayed_flag', fit_scaler=True
        )
        X_test, y_test, _ = self.preprocessor.prepare_features(
            test_df, 'delayed_flag', fit_scaler=False
        )
        
        # Test models
        models_to_test = {
            'Logistic Regression': LogisticRegressionModel(max_iter=1000, random_state=42),
            'Random Forest': RandomForestModel(
                task='classification',
                n_estimators=200,
                max_depth=20,
                random_state=42
            )
        }
        
        # Evaluate each model
        for model_name, model in models_to_test.items():
            metrics = self.evaluate_model_temporal(
                model_name, model, X_train, y_train, X_test, y_test
            )
            self.results[model_name] = metrics
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save temporal validation results"""
        results_file = self.output_dir / "temporal_validation_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Temporal validation results saved to: {results_file}")
        
        # Create summary report
        report_file = self.output_dir / "temporal_validation_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TEMPORAL VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write("This validation tests model performance on FUTURE data\n")
            f.write("Training on historical weeks, testing on recent weeks\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"\n{model_name}:\n")
                f.write("-" * 40 + "\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric:15s}: {value:.4f}\n")
        
        print(f"✅ Temporal validation report saved to: {report_file}")
    
    def compare_random_vs_temporal(self, random_results):
        """
        Compare random split results vs temporal split results
        """
        print(f"\n{'='*80}")
        print("RANDOM VS TEMPORAL SPLIT COMPARISON")
        print(f"{'='*80}\n")
        
        comparison = []
        
        for model_name in self.results.keys():
            if model_name in random_results:
                temporal = self.results[model_name]
                random = random_results[model_name]
                
                print(f"\n{model_name}:")
                print("-" * 60)
                
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metric in temporal and metric in random:
                        temp_val = temporal[metric]
                        rand_val = random[metric]
                        diff = temp_val - rand_val
                        
                        print(f"  {metric:15s}: Random={rand_val:.4f}, Temporal={temp_val:.4f}, Diff={diff:+.4f}")
                        
                        comparison.append({
                            'model': model_name,
                            'metric': metric,
                            'random': rand_val,
                            'temporal': temp_val,
                            'difference': diff
                        })
        
        # Save comparison
        comparison_df = pd.DataFrame(comparison)
        comparison_file = self.output_dir / "random_vs_temporal_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"\n✅ Comparison saved to: {comparison_file}")
        
        return comparison_df


def main():
    """Run temporal validation"""
    data_path = Path("data/synthetic_delivery_data.csv")
    
    if not data_path.exists():
        data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    validator = TemporalValidator(str(data_path))
    results = validator.run_full_temporal_validation(test_weeks=4)
    
    print(f"\n{'='*80}")
    print("TEMPORAL VALIDATION COMPLETE")
    print(f"{'='*80}")
    print("\nResults show how models perform on FUTURE data")
    print("This is the most realistic evaluation for time-series prediction")


if __name__ == "__main__":
    main()

