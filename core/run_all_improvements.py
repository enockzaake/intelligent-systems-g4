"""
Master Script - Run All Improvements
Executes the complete improvement pipeline
"""
import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    """Run a Python script and report status"""
    print("\n" + "=" * 80)
    print(f"▶️  {description}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed:.1f}s")
        return True
    
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False
    
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        return False


def main():
    """Execute complete improvement pipeline"""
    
    print("\n" + "=" * 80)
    print("🚀 COMPLETE MODEL IMPROVEMENT PIPELINE")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Check data quality")
    print("  2. Train improved models (~15 minutes)")
    print("  3. Train precision-focused LSTM (~15 minutes)")
    print("  4. Test ensemble model")
    print("  5. Generate visualizations")
    print("  6. Compare results")
    print("\n⏱️  Total estimated time: ~35-40 minutes")
    
    response = input("\n▶️  Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    results = {}
    
    # Step 1: Data check
    results['data_check'] = run_script(
        "quick_data_check.py",
        "Step 1/6: Quick Data Quality Check"
    )
    
    if not results['data_check']:
        print("\n⚠️  Data check failed. Please verify data file exists.")
        return
    
    # Step 2: Train improved models
    results['train_improved'] = run_script(
        "train_improved.py",
        "Step 2/6: Training Improved Models"
    )
    
    if not results['train_improved']:
        print("\n⚠️  Training failed. Check error messages above.")
        return
    
    # Step 3: Train precision-focused LSTM
    print("\n" + "=" * 80)
    print("Step 3/6: Training Precision-Focused LSTM")
    print("=" * 80)
    print("\nThis creates an alternative LSTM with better precision.")
    print("You can skip this if you only need the standard improved LSTM.")
    
    skip_precision = input("\n▶️  Train precision-focused LSTM? (y/n): ")
    if skip_precision.lower() == 'y':
        results['train_precision'] = run_script(
            "train_lstm_precision_focused.py",
            "Training Precision-Focused LSTM"
        )
    else:
        print("⏭️  Skipped precision-focused LSTM training")
        results['train_precision'] = True
    
    # Step 4: Test ensemble
    results['ensemble'] = run_script(
        "ensemble_model.py",
        "Step 4/6: Testing Ensemble Model"
    )
    
    # Step 5: Generate visualizations
    results['visualizations'] = run_script(
        "visualize_improvements.py",
        "Step 5/6: Generating Visualizations"
    )
    
    # Step 6: Compare results
    results['compare'] = run_script(
        "compare_results.py",
        "Step 6/6: Comparing Results"
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 PIPELINE SUMMARY")
    print("=" * 80)
    
    for step, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {step:20}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All steps completed successfully!")
        print("\n📁 Output Locations:")
        print("  • Models: outputs_improved/models/")
        print("  • Results: outputs_improved/results/")
        print("  • Visualizations: outputs/analysis/")
        print("\n📚 Next Steps:")
        print("  • Review visualizations in outputs/analysis/")
        print("  • Check evaluation report: outputs_improved/results/evaluation_report.txt")
        print("  • Try predictions: python predict_improved.py")
        print("  • Read complete guide: COMPLETE_GUIDE.md")
    else:
        print("\n⚠️  Some steps failed. Please check errors above.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

