"""
Statistical Significance Testing Module
Performs McNemar's test and other statistical tests to compare models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2
from statsmodels.stats.contingency_tables import mcnemar
import json

from data_preprocessing import DataPreprocessor
from models import LogisticRegressionModel, RandomForestModel


def mcnemar_test(y_true, y_pred1, y_pred2, model1_name="Model 1", model2_name="Model 2"):
    """
    Perform McNemar's test to compare two classifiers
    
    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
    
    Returns:
        Dictionary with test results
    """
    # Create contingency table
    # [[both_correct, model1_wrong_model2_correct],
    #  [model1_correct_model2_wrong, both_wrong]]
    
    both_correct = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    model1_only = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    model2_only = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    both_wrong = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
    
    # McNemar's test focuses on disagreements
    contingency_table = [[both_correct, model2_only],
                        [model1_only, both_wrong]]
    
    # Perform test
    try:
        result = mcnemar(contingency_table, exact=False, correction=True)
        p_value = result.pvalue
        statistic = result.statistic
    except Exception as e:
        print(f"Warning: McNemar test failed: {e}")
        # Fallback calculation
        if model1_only + model2_only > 0:
            statistic = ((abs(model1_only - model2_only) - 1)**2) / (model1_only + model2_only)
            p_value = 1 - chi2.cdf(statistic, 1)
        else:
            statistic = 0
            p_value = 1.0
    
    # Interpret results
    if p_value < 0.001:
        significance = "Highly Significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "Very Significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "Significant (p < 0.05)"
    else:
        significance = "Not Significant (p >= 0.05)"
    
    # Determine which model is better
    if model2_only > model1_only:
        better_model = model2_name
        advantage = model2_only - model1_only
    elif model1_only > model2_only:
        better_model = model1_name
        advantage = model1_only - model2_only
    else:
        better_model = "Tie"
        advantage = 0
    
    return {
        'test_name': "McNemar's Test",
        'model1': model1_name,
        'model2': model2_name,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significance': significance,
        'better_model': better_model,
        'advantage': int(advantage),
        'contingency_table': {
            'both_correct': int(both_correct),
            'model1_correct_only': int(model1_only),
            'model2_correct_only': int(model2_only),
            'both_wrong': int(both_wrong)
        }
    }


def paired_ttest_metrics(scores1, scores2, metric_name="Metric"):
    """
    Perform paired t-test on cross-validation scores
    
    Args:
        scores1: Array of scores from model 1
        scores2: Array of scores from model 2
        metric_name: Name of the metric being compared
    
    Returns:
        Dictionary with test results
    """
    from scipy.stats import ttest_rel
    
    # Perform paired t-test
    t_stat, p_value = ttest_rel(scores1, scores2)
    
    # Calculate effect size (Cohen's d)
    mean_diff = np.mean(scores1) - np.mean(scores2)
    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
    cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # Interpret effect size
    if abs(cohen_d) < 0.2:
        effect_interpretation = "Negligible"
    elif abs(cohen_d) < 0.5:
        effect_interpretation = "Small"
    elif abs(cohen_d) < 0.8:
        effect_interpretation = "Medium"
    else:
        effect_interpretation = "Large"
    
    # Interpret significance
    if p_value < 0.001:
        significance = "Highly Significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "Very Significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "Significant (p < 0.05)"
    else:
        significance = "Not Significant (p >= 0.05)"
    
    return {
        'test_name': "Paired T-Test",
        'metric': metric_name,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significance': significance,
        'mean_difference': float(mean_diff),
        'cohen_d': float(cohen_d),
        'effect_size': effect_interpretation
    }


class StatisticalComparison:
    """
    Performs comprehensive statistical comparison of models
    """
    
    def __init__(self, data_path, output_dir="outputs/statistical_tests"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = DataPreprocessor(data_path)
        self.results = {}
    
    def run_mcnemar_tests(self):
        """
        Run McNemar's test for all model pairs
        """
        print(f"\n{'='*80}")
        print("MCNEMAR'S TEST - PAIRWISE MODEL COMPARISON")
        print(f"{'='*80}")
        
        # Load and preprocess data
        print("\nLoading data...")
        df = self.preprocessor.load_data()
        df = self.preprocessor.create_features(df)
        df = self.preprocessor.encode_categorical(df, fit=True)
        
        # Split data
        train_df, test_df = self.preprocessor.split_by_routes(df, test_size=0.2, random_state=42)
        
        # Prepare features
        X_train, y_train, feature_cols = self.preprocessor.prepare_features(
            train_df, 'delayed_flag', fit_scaler=True
        )
        X_test, y_test, _ = self.preprocessor.prepare_features(
            test_df, 'delayed_flag', fit_scaler=False
        )
        
        print(f"Test samples: {len(X_test):,}")
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegressionModel(max_iter=1000, random_state=42),
            'Random Forest': RandomForestModel(
                task='classification',
                n_estimators=200,
                max_depth=20,
                random_state=42
            )
        }
        
        predictions = {}
        
        print("\nTraining models...")
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            model.train(X_train, y_train)
            predictions[model_name] = model.predict(X_test)
        
        # Perform pairwise comparisons
        print("\nPerforming pairwise comparisons...")
        
        model_names = list(models.keys())
        mcnemar_results = []
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1_name = model_names[i]
                model2_name = model_names[j]
                
                print(f"\n  Comparing: {model1_name} vs {model2_name}")
                
                result = mcnemar_test(
                    y_test,
                    predictions[model1_name],
                    predictions[model2_name],
                    model1_name,
                    model2_name
                )
                
                mcnemar_results.append(result)
                
                print(f"    Statistic: {result['statistic']:.4f}")
                print(f"    P-value: {result['p_value']:.6f}")
                print(f"    {result['significance']}")
                print(f"    Better model: {result['better_model']}")
        
        self.results['mcnemar_tests'] = mcnemar_results
        
        return mcnemar_results
    
    def run_cv_comparisons(self, cv_results):
        """
        Run paired t-tests on cross-validation results
        
        Args:
            cv_results: Dictionary with cross-validation results
        """
        print(f"\n{'='*80}")
        print("PAIRED T-TESTS - CROSS-VALIDATION SCORES")
        print(f"{'='*80}")
        
        model_names = list(cv_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        ttest_results = []
        
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            print("-" * 60)
            
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1_name = model_names[i]
                    model2_name = model_names[j]
                    
                    scores1 = cv_results[model1_name][metric]['values']
                    scores2 = cv_results[model2_name][metric]['values']
                    
                    result = paired_ttest_metrics(
                        scores1, scores2,
                        f"{metric} ({model1_name} vs {model2_name})"
                    )
                    
                    result['model1'] = model1_name
                    result['model2'] = model2_name
                    
                    ttest_results.append(result)
                    
                    print(f"  {model1_name} vs {model2_name}:")
                    print(f"    T-statistic: {result['t_statistic']:.4f}")
                    print(f"    P-value: {result['p_value']:.6f}")
                    print(f"    {result['significance']}")
                    print(f"    Effect size (Cohen's d): {result['cohen_d']:.4f} ({result['effect_size']})")
        
        self.results['ttest_results'] = ttest_results
        
        return ttest_results
    
    def save_results(self):
        """Save all statistical test results"""
        results_file = self.output_dir / "statistical_tests_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Statistical test results saved to: {results_file}")
        
        # Create detailed report
        report_file = self.output_dir / "statistical_tests_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STATISTICAL SIGNIFICANCE TESTING REPORT\n")
            f.write("="*80 + "\n\n")
            
            # McNemar's tests
            if 'mcnemar_tests' in self.results:
                f.write("\n" + "="*80 + "\n")
                f.write("MCNEMAR'S TEST RESULTS\n")
                f.write("="*80 + "\n\n")
                f.write("Tests whether two classifiers have significantly different error rates\n\n")
                
                for result in self.results['mcnemar_tests']:
                    f.write(f"{result['model1']} vs {result['model2']}:\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"  Test Statistic: {result['statistic']:.4f}\n")
                    f.write(f"  P-value: {result['p_value']:.6f}\n")
                    f.write(f"  Significance: {result['significance']}\n")
                    f.write(f"  Better Model: {result['better_model']}\n")
                    f.write(f"  Advantage: {result['advantage']} additional correct predictions\n\n")
                    f.write("  Contingency Table:\n")
                    f.write(f"    Both correct: {result['contingency_table']['both_correct']}\n")
                    f.write(f"    {result['model1']} correct only: {result['contingency_table']['model1_correct_only']}\n")
                    f.write(f"    {result['model2']} correct only: {result['contingency_table']['model2_correct_only']}\n")
                    f.write(f"    Both wrong: {result['contingency_table']['both_wrong']}\n\n")
            
            # T-test results
            if 'ttest_results' in self.results:
                f.write("\n" + "="*80 + "\n")
                f.write("PAIRED T-TEST RESULTS\n")
                f.write("="*80 + "\n\n")
                f.write("Tests whether cross-validation scores differ significantly\n\n")
                
                for result in self.results['ttest_results']:
                    f.write(f"{result['metric']}:\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"  T-statistic: {result['t_statistic']:.4f}\n")
                    f.write(f"  P-value: {result['p_value']:.6f}\n")
                    f.write(f"  Significance: {result['significance']}\n")
                    f.write(f"  Mean Difference: {result['mean_difference']:.4f}\n")
                    f.write(f"  Cohen's d: {result['cohen_d']:.4f}\n")
                    f.write(f"  Effect Size: {result['effect_size']}\n\n")
        
        print(f"✅ Statistical test report saved to: {report_file}")


def main():
    """Run statistical tests"""
    data_path = Path("data/synthetic_delivery_data.csv")
    
    if not data_path.exists():
        data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    comparison = StatisticalComparison(str(data_path))
    comparison.run_mcnemar_tests()
    comparison.save_results()
    
    print(f"\n{'='*80}")
    print("STATISTICAL TESTING COMPLETE")
    print(f"{'='*80}")
    print("\nResults show whether model differences are statistically significant")
    print("McNemar's test compares classifier disagreements")


if __name__ == "__main__":
    main()

