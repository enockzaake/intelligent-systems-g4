"""
Comprehensive Validation Script
Runs all validation methods: temporal, cross-validation, baselines, and statistical tests
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Import validation modules
from temporal_validation import TemporalValidator
from cross_validation import CrossValidator
from baseline_models import BaselineComparison
from statistical_tests import StatisticalComparison


def run_comprehensive_validation(data_path, output_dir="outputs/comprehensive_validation"):
    """
    Run all validation methods and generate comprehensive report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("="*80)
    print(f"\nData: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'data_path': str(data_path),
        'timestamp': datetime.now().isoformat(),
        'validations': {}
    }
    
    # 1. Baseline Comparison
    print("\n" + "="*80)
    print("STEP 1: BASELINE MODEL COMPARISON")
    print("="*80)
    print("Establishing minimum performance requirements...")
    
    try:
        baseline_comp = BaselineComparison(
            str(data_path),
            output_dir=output_dir / "baseline_comparison"
        )
        baseline_results = baseline_comp.run_baseline_comparison()
        results['validations']['baseline'] = {
            'status': 'success',
            'results': baseline_results
        }
        print("✅ Baseline comparison complete")
    except Exception as e:
        print(f"❌ Baseline comparison failed: {e}")
        results['validations']['baseline'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # 2. Cross-Validation
    print("\n" + "="*80)
    print("STEP 2: K-FOLD CROSS-VALIDATION")
    print("="*80)
    print("Testing model stability with multiple train-test splits...")
    
    try:
        cv = CrossValidator(
            str(data_path),
            output_dir=output_dir / "cross_validation"
        )
        cv_results = cv.run_cross_validation(n_splits=5)
        results['validations']['cross_validation'] = {
            'status': 'success',
            'results': cv_results
        }
        print("✅ Cross-validation complete")
    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
        results['validations']['cross_validation'] = {
            'status': 'failed',
            'error': str(e)
        }
        cv_results = None
    
    # 3. Temporal Validation
    print("\n" + "="*80)
    print("STEP 3: TEMPORAL VALIDATION")
    print("="*80)
    print("Testing model performance on future data...")
    
    try:
        temporal = TemporalValidator(
            str(data_path),
            output_dir=output_dir / "temporal_validation"
        )
        temporal_results = temporal.run_full_temporal_validation(test_weeks=4)
        results['validations']['temporal'] = {
            'status': 'success',
            'results': temporal_results
        }
        print("✅ Temporal validation complete")
    except Exception as e:
        print(f"❌ Temporal validation failed: {e}")
        results['validations']['temporal'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # 4. Statistical Significance Tests
    print("\n" + "="*80)
    print("STEP 4: STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    print("Testing whether model differences are statistically significant...")
    
    try:
        stat_tests = StatisticalComparison(
            str(data_path),
            output_dir=output_dir / "statistical_tests"
        )
        stat_tests.run_mcnemar_tests()
        
        if cv_results:
            stat_tests.run_cv_comparisons(cv_results)
        
        stat_tests.save_results()
        results['validations']['statistical_tests'] = {
            'status': 'success',
            'results': stat_tests.results
        }
        print("✅ Statistical testing complete")
    except Exception as e:
        print(f"❌ Statistical testing failed: {e}")
        results['validations']['statistical_tests'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Save comprehensive results
    results_file = output_dir / "comprehensive_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Comprehensive results saved to: {results_file}")
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    return results


def generate_summary_report(results, output_dir):
    """
    Generate a comprehensive summary report
    """
    report_file = output_dir / "VALIDATION_SUMMARY_REPORT.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE VALIDATION SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {results['timestamp']}\n")
        f.write(f"Data: {results['data_path']}\n\n")
        
        # Executive Summary
        f.write("="*80 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        successful = sum(1 for v in results['validations'].values() if v['status'] == 'success')
        total = len(results['validations'])
        
        f.write(f"Validations Completed: {successful}/{total}\n\n")
        
        # Validation Status
        f.write("Validation Status:\n")
        for validation_name, validation_data in results['validations'].items():
            status_icon = "✅" if validation_data['status'] == 'success' else "❌"
            f.write(f"  {status_icon} {validation_name.replace('_', ' ').title()}: {validation_data['status'].upper()}\n")
        
        f.write("\n")
        
        # Baseline Comparison Summary
        if 'baseline' in results['validations'] and results['validations']['baseline']['status'] == 'success':
            f.write("\n" + "="*80 + "\n")
            f.write("BASELINE COMPARISON SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            baseline_results = results['validations']['baseline']['results']
            
            # Sort by F1-score
            sorted_baselines = sorted(
                baseline_results.items(),
                key=lambda x: x[1]['f1_score'],
                reverse=True
            )
            
            f.write("Baseline Models (sorted by F1-score):\n\n")
            for model_name, metrics in sorted_baselines:
                f.write(f"  {model_name}:\n")
                f.write(f"    Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall:    {metrics['recall']:.4f}\n")
                f.write(f"    F1-Score:  {metrics['f1_score']:.4f}\n\n")
            
            best_baseline = sorted_baselines[0]
            f.write(f"Best Baseline: {best_baseline[0]} (F1: {best_baseline[1]['f1_score']:.4f})\n")
            f.write("ML models must significantly outperform this baseline.\n")
        
        # Cross-Validation Summary
        if 'cross_validation' in results['validations'] and results['validations']['cross_validation']['status'] == 'success':
            f.write("\n" + "="*80 + "\n")
            f.write("CROSS-VALIDATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            cv_results = results['validations']['cross_validation']['results']
            
            f.write("Model Performance with Confidence Intervals:\n\n")
            for model_name, statistics in cv_results.items():
                f.write(f"  {model_name}:\n")
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metric in statistics:
                        stats = statistics[metric]
                        f.write(f"    {metric}:\n")
                        f.write(f"      Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                        f.write(f"      95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]\n")
                f.write("\n")
        
        # Temporal Validation Summary
        if 'temporal' in results['validations'] and results['validations']['temporal']['status'] == 'success':
            f.write("\n" + "="*80 + "\n")
            f.write("TEMPORAL VALIDATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write("Performance on Future Data (Most Realistic Test):\n\n")
            
            temporal_results = results['validations']['temporal']['results']
            
            for model_name, metrics in temporal_results.items():
                f.write(f"  {model_name}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.4f}\n")
                f.write("\n")
        
        # Statistical Tests Summary
        if 'statistical_tests' in results['validations'] and results['validations']['statistical_tests']['status'] == 'success':
            f.write("\n" + "="*80 + "\n")
            f.write("STATISTICAL SIGNIFICANCE SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            stat_results = results['validations']['statistical_tests']['results']
            
            if 'mcnemar_tests' in stat_results:
                f.write("McNemar's Test Results:\n\n")
                for result in stat_results['mcnemar_tests']:
                    f.write(f"  {result['model1']} vs {result['model2']}:\n")
                    f.write(f"    P-value: {result['p_value']:.6f}\n")
                    f.write(f"    {result['significance']}\n")
                    f.write(f"    Better Model: {result['better_model']}\n\n")
        
        # Key Findings
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS & RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. BASELINE COMPARISON:\n")
        f.write("   - Establishes minimum performance requirements\n")
        f.write("   - ML models must significantly outperform simple baselines\n\n")
        
        f.write("2. CROSS-VALIDATION:\n")
        f.write("   - Provides confidence intervals for model performance\n")
        f.write("   - Route-aware splitting prevents data leakage\n")
        f.write("   - Assesses model stability across different data splits\n\n")
        
        f.write("3. TEMPORAL VALIDATION:\n")
        f.write("   - Most realistic evaluation for time-series prediction\n")
        f.write("   - Tests model performance on future/unseen time periods\n")
        f.write("   - Critical for production deployment confidence\n\n")
        
        f.write("4. STATISTICAL SIGNIFICANCE:\n")
        f.write("   - McNemar's test confirms model differences are not due to chance\n")
        f.write("   - P-values < 0.05 indicate significant differences\n")
        f.write("   - Effect sizes (Cohen's d) quantify practical importance\n\n")
        
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        f.write("This comprehensive validation provides multiple lines of evidence\n")
        f.write("for model performance and statistical significance.\n\n")
        f.write("For production deployment, consider:\n")
        f.write("- Model with best temporal validation performance\n")
        f.write("- Smallest confidence intervals (most stable)\n")
        f.write("- Statistically significant improvements over baselines\n")
        f.write("- Balance between performance and complexity\n")
    
    print(f"✅ Summary report saved to: {report_file}")


def main():
    """
    Main execution function
    """
    # Check for data file
    data_path = Path("data/synthetic_delivery_data.csv")
    
    if not data_path.exists():
        data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found")
        print("Please ensure data exists at:")
        print("  - data/synthetic_delivery_data.csv")
        print("  - OR data/cleaned_delivery_data.csv")
        sys.exit(1)
    
    # Run comprehensive validation
    results = run_comprehensive_validation(data_path)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION COMPLETE")
    print("="*80)
    print("\nAll validation results saved to: outputs/comprehensive_validation/")
    print("\nKey files:")
    print("  - VALIDATION_SUMMARY_REPORT.txt  (Executive summary)")
    print("  - comprehensive_validation_results.json  (Complete results)")
    print("  - */  (Individual validation folders)")
    print("\n" + "="*80)
    
    return results


if __name__ == "__main__":
    main()

