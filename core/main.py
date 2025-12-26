import sys
import argparse
from pathlib import Path

from train import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description='Train models on delivery dataset')
    parser.add_argument('--dataset', type=str, 
                       default='data/improved_delivery_data.csv',
                       help='Path to training dataset CSV file')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("AI-DRIVEN ROUTE OPTIMIZATION AND DELAY PREDICTION SYSTEM")
    print("=" * 80)
    
    # Use specified dataset or fallback to defaults
    data_path = Path(args.dataset)
    if not data_path.exists():
        # Try fallback options
        fallback_paths = [
            Path("data/improved_delivery_data.csv"),
            Path("data/synthetic_delivery_data.csv"),
            Path("data/cleaned_delivery_data.csv")
        ]
        
        for fallback in fallback_paths:
            if fallback.exists():
                data_path = fallback
                print(f"\n⚠️  Specified dataset not found. Using: {data_path}")
                break
        else:
            print(f"\n❌ Error: Data file not found at {args.dataset}")
            print("Please ensure the dataset file exists or generate it first:")
            print("  python generate_improved_dataset.py")
        sys.exit(1)
    else:
        print(f"\n📁 Using dataset: {data_path}")
    
    output_dir = Path("outputs")
    
    trainer = ModelTrainer(str(data_path), output_dir=str(output_dir))
    
    trainer.preprocess_data()
    
    trainer.get_summary_statistics()
    
    # Train with optimized parameters
    trainer.train_all_models(
        lstm_epochs=100,
        lstm_batch_size=128,
        lstm_sequence_length=5
    )
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel outputs available in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
