import sys
from pathlib import Path

from train import ModelTrainer


def main():
    print("\n" + "=" * 80)
    print("AI-DRIVEN ROUTE OPTIMIZATION AND DELAY PREDICTION SYSTEM")
    print("=" * 80)
    
    # Use synthetic data by default (better for training)
    data_path = Path("data/synthetic_delivery_data.csv")
    if not data_path.exists():
        data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"\nError: Data file not found at {data_path}")
        print("Please ensure the cleaned data file exists.")
        sys.exit(1)
    
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
