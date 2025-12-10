"""
Test Synthetic Data Pipeline
Generate data, train models, and evaluate results
"""
import subprocess
import sys
from pathlib import Path
import shutil


def run_command(cmd, description):
    """Run a command and report status"""
    print("\n" + "=" * 80)
    print(f"▶️  {description}")
    print("=" * 80)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        return False


def main():
    """Run complete synthetic data test pipeline"""
    
    print("\n" + "=" * 80)
    print("🧪 SYNTHETIC DATA TEST PIPELINE")
    print("=" * 80)
    
    print("\nThis will:")
    print("  1. Generate synthetic data")
    print("  2. Compare with real data")
    print("  3. Train models on synthetic data")
    print("  4. Compare results")
    print("\n⏱️  Estimated time: 25-30 minutes")
    
    response = input("\n▶️  Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Generate synthetic data
    if not run_command(
        f"{sys.executable} generate_synthetic_data.py",
        "Step 1/4: Generating Synthetic Data"
    ):
        return
    
    # Step 2: Compare datasets
    if not run_command(
        f"{sys.executable} compare_datasets.py",
        "Step 2/4: Comparing Real vs Synthetic Data"
    ):
        print("⚠️  Comparison failed, but continuing...")
    
    # Step 3: Backup real data and use synthetic
    print("\n" + "=" * 80)
    print("Step 3/4: Training Models on Synthetic Data")
    print("=" * 80)
    
    real_data = Path("data/cleaned_delivery_data.csv")
    synthetic_data = Path("data/synthetic_delivery_data.csv")
    backup_data = Path("data/cleaned_delivery_data_BACKUP.csv")
    
    # Backup real data
    if real_data.exists() and not backup_data.exists():
        print(f"\n📋 Backing up real data to: {backup_data}")
        shutil.copy(real_data, backup_data)
    
    # Replace with synthetic
    if synthetic_data.exists():
        print(f"📋 Temporarily using synthetic data for training...")
        shutil.copy(synthetic_data, real_data)
        
        # Create new output directory for synthetic results
        synth_output = Path("outputs_synthetic")
        if synth_output.exists():
            print(f"⚠️  Removing existing synthetic outputs...")
            shutil.rmtree(synth_output)
        
        # Modify train_improved to use different output
        print("\n🏋️  Training models on synthetic data...")
        
        if not run_command(
            f"{sys.executable} train_improved.py",
            "Training Improved Models on Synthetic Data"
        ):
            print("❌ Training failed")
            # Restore real data
            if backup_data.exists():
                shutil.copy(backup_data, real_data)
            return
        
        # Rename outputs
        if Path("outputs_improved").exists():
            shutil.move("outputs_improved", synth_output)
            print(f"✅ Results saved to: {synth_output}")
    
    # Restore real data
    if backup_data.exists():
        print(f"\n📋 Restoring real data...")
        shutil.copy(backup_data, real_data)
    
    # Step 4: Compare results
    print("\n" + "=" * 80)
    print("Step 4/4: Analyzing Results")
    print("=" * 80)
    
    print("\n📊 Results Summary:")
    
    # Check if we have results
    if Path("outputs_synthetic/results/evaluation_report.txt").exists():
        print("\n✅ Synthetic Data Results:")
        with open("outputs_synthetic/results/evaluation_report.txt", 'r') as f:
            lines = f.readlines()
            # Print first 50 lines
            for line in lines[:50]:
                print("  " + line.rstrip())
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 SYNTHETIC DATA TEST COMPLETE")
    print("=" * 80)
    
    print("\n📁 Output Locations:")
    print("  • Synthetic data: data/synthetic_delivery_data.csv")
    print("  • Real data backup: data/cleaned_delivery_data_BACKUP.csv")
    print("  • Synthetic results: outputs_synthetic/")
    print("  • Real results: outputs_improved/")
    print("  • Comparison chart: outputs/analysis/real_vs_synthetic_comparison.png")
    
    print("\n📊 Compare Results:")
    print("  • Synthetic: outputs_synthetic/results/evaluation_report.txt")
    print("  • Real: outputs_improved/results/evaluation_report.txt")
    
    print("\n🎯 Next Steps:")
    print("  1. Review comparison chart")
    print("  2. Check if synthetic data models perform better")
    print("  3. If better, use synthetic data for production")
    print("  4. If not, adjust generation parameters in generate_synthetic_data.py")


if __name__ == "__main__":
    main()

