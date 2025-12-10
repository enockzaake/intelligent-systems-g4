"""
Improved Model Training Script
Addresses class imbalance, low variance, and poor LSTM performance
"""
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from models import LogisticRegressionModel, RandomForestModel, LSTMModel
from evaluate import ModelEvaluator
from config import Config


class ImprovedModelTrainer:
    def __init__(self, data_path, output_dir="outputs_improved"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor_dir = self.output_dir / "preprocessor"
        self.preprocessor_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = DataPreprocessor(data_path)
        self.evaluator = ModelEvaluator(results_dir=self.results_dir)
        
        self.models = {}
        self.data = None
        self.class_weights = None
    
    def preprocess_data(self):
        print("=" * 80)
        print("PREPROCESSING DATA (IMPROVED)")
        print("=" * 80)
        
        self.data = self.preprocessor.process_full_pipeline()
        
        # Calculate class weights for imbalanced data
        y_train = self.data['classification']['y_train']
        n_samples = len(y_train)
        n_classes = 2
        n_class_0 = np.sum(y_train == 0)
        n_class_1 = np.sum(y_train == 1)
        
        # Balanced class weights
        weight_0 = n_samples / (n_classes * n_class_0)
        weight_1 = n_samples / (n_classes * n_class_1)
        
        self.class_weights = {0: weight_0, 1: weight_1}
        
        print(f"\nTraining samples (classification): {len(self.data['classification']['X_train'])}")
        print(f"Test samples (classification): {len(self.data['classification']['X_test'])}")
        print(f"Number of features: {self.data['classification']['X_train'].shape[1]}")
        
        print(f"\n--- Class Distribution ---")
        print(f"Class 0 (On-time): {n_class_0:,} ({n_class_0/n_samples:.2%})")
        print(f"Class 1 (Delayed): {n_class_1:,} ({n_class_1/n_samples:.2%})")
        print(f"\n--- Computed Class Weights ---")
        print(f"Weight for class 0: {weight_0:.4f}")
        print(f"Weight for class 1: {weight_1:.4f}")
        
        self.preprocessor.save_preprocessor(self.preprocessor_dir)
        print(f"\nPreprocessor saved to {self.preprocessor_dir}")
    
    def train_logistic_regression_improved(self):
        print("\n" + "=" * 80)
        print("TRAINING LOGISTIC REGRESSION (Classification) - IMPROVED")
        print("=" * 80)
        
        # Use class weights to handle imbalance
        model = LogisticRegressionModel(
            max_iter=1000, 
            random_state=42,
            class_weight=self.class_weights
        )
        
        X_train = self.data['classification']['X_train']
        y_train = self.data['classification']['y_train']
        X_test = self.data['classification']['X_test']
        y_test = self.data['classification']['y_test']
        
        print("Training model with class weights...")
        model.train(X_train, y_train)
        
        print("Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        
        print("\nMetrics:")
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                print(f"  {metric}: {value:.4f}")
        
        # Show confusion matrix
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0][0]:,}  |  FP: {cm[0][1]:,}")
            print(f"  FN: {cm[1][0]:,}  |  TP: {cm[1][1]:,}")
        
        model_path = self.models_dir / "logistic_regression_improved.pkl"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        self.models["Logistic Regression (Improved)"] = model
        self.evaluator.add_result("Logistic Regression (Improved)", "classification", metrics)
        
        self.evaluator.plot_feature_importance(
            model, 
            self.data['feature_columns'], 
            "Logistic Regression (Improved)"
        )
        
        return model, metrics
    
    def train_random_forest_classifier_improved(self):
        print("\n" + "=" * 80)
        print("TRAINING RANDOM FOREST CLASSIFIER - IMPROVED")
        print("=" * 80)
        
        model = RandomForestModel(
            task="classification",
            n_estimators=200,  # Increased from 100
            max_depth=20,       # Increased from 15
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=self.class_weights,
            random_state=42
        )
        
        X_train = self.data['classification']['X_train']
        y_train = self.data['classification']['y_train']
        X_test = self.data['classification']['X_test']
        y_test = self.data['classification']['y_test']
        
        print("Training model with improved hyperparameters and class weights...")
        model.train(X_train, y_train)
        
        print("Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        
        print("\nMetrics:")
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                print(f"  {metric}: {value:.4f}")
        
        # Show confusion matrix
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0][0]:,}  |  FP: {cm[0][1]:,}")
            print(f"  FN: {cm[1][0]:,}  |  TP: {cm[1][1]:,}")
        
        model_path = self.models_dir / "random_forest_classifier_improved.pkl"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        self.models["Random Forest Classifier (Improved)"] = model
        self.evaluator.add_result("Random Forest Classifier (Improved)", "classification", metrics)
        
        self.evaluator.plot_feature_importance(
            model,
            self.data['feature_columns'],
            "Random Forest Classifier (Improved)"
        )
        
        return model, metrics
    
    def train_lstm_classifier_improved(self, sequence_length=5, epochs=100, batch_size=128):
        print("\n" + "=" * 80)
        print("TRAINING LSTM CLASSIFIER - IMPROVED")
        print("=" * 80)
        
        print(f"Preparing sequences (sequence_length={sequence_length})...")
        
        train_sequences, train_targets_clf, _ = self.preprocessor.prepare_lstm_sequences(
            self.data['train_df'], sequence_length
        )
        test_sequences, test_targets_clf, _ = self.preprocessor.prepare_lstm_sequences(
            self.data['test_df'], sequence_length
        )
        
        print(f"Train sequences: {train_sequences.shape}")
        print(f"Test sequences: {test_sequences.shape}")
        
        input_size = train_sequences.shape[2]
        
        # Calculate class weights for PyTorch (as a ratio)
        pos_weight = self.class_weights[1] / self.class_weights[0]
        
        print(f"\nClass imbalance ratio: {pos_weight:.4f}")
        print(f"This will be used in BCEWithLogitsLoss for weighted training.")
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=128,      # Increased from 64
            num_layers=3,         # Increased from 2
            dropout=0.3,          # Increased from 0.2
            task="classification",
            learning_rate=0.005,  # Increased from 0.001
            pos_weight=pos_weight  # Add class weight
        )
        
        print("\nTraining model with improved architecture and class weights...")
        print(f"Architecture: 3 layers, 128 hidden units")
        print(f"Dropout: 0.3, Learning rate: 0.005")
        
        history = model.train(
            train_sequences, train_targets_clf,
            X_val=test_sequences, y_val=test_targets_clf,
            epochs=epochs, batch_size=batch_size, verbose=True
        )
        
        print("\nEvaluating model...")
        metrics = model.evaluate(test_sequences, test_targets_clf, batch_size=batch_size)
        
        print("\nMetrics:")
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                print(f"  {metric}: {value:.4f}")
        
        # Show confusion matrix
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0][0]:,}  |  FP: {cm[0][1]:,}")
            print(f"  FN: {cm[1][0]:,}  |  TP: {cm[1][1]:,}")
        
        model_path = self.models_dir / "lstm_classifier_improved.pth"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        self.models["LSTM Classifier (Improved)"] = model
        self.evaluator.add_result("LSTM Classifier (Improved)", "classification", metrics)
        
        return model, metrics, history
    
    def train_classification_models(self, lstm_epochs=100, lstm_batch_size=128, lstm_sequence_length=5):
        """Train only classification models (skip regression since it performs poorly)"""
        print("\n" + "=" * 80)
        print("TRAINING CLASSIFICATION MODELS (IMPROVED)")
        print("=" * 80)
        
        if self.data is None:
            self.preprocess_data()
        
        print("\n📊 FOCUS: Classification models only")
        print("📝 NOTE: Skipping regression due to extremely low target variance")
        print("         (Max delay is <0.3 minutes, making regression unfeasible)")
        
        self.train_logistic_regression_improved()
        self.train_random_forest_classifier_improved()
        self.train_lstm_classifier_improved(
            sequence_length=lstm_sequence_length,
            epochs=lstm_epochs,
            batch_size=lstm_batch_size
        )
        
        print("\n" + "=" * 80)
        print("GENERATING EVALUATION REPORTS")
        print("=" * 80)
        
        self.evaluator.save_results()
        self.evaluator.create_comparison_report()
        self.evaluator.create_comparison_table()
        self.evaluator.plot_model_comparison()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print(f"  - Models: {self.models_dir}")
        print(f"  - Results: {self.results_dir}")
        print(f"  - Preprocessor: {self.preprocessor_dir}")
        
        self.print_improvement_summary()
    
    def print_improvement_summary(self):
        """Print a summary of improvements made"""
        print("\n" + "=" * 80)
        print("IMPROVEMENTS IMPLEMENTED")
        print("=" * 80)
        
        improvements = [
            "",
            "✅ 1. Class Weights Added",
            "   - Logistic Regression: Using balanced class weights",
            "   - Random Forest: Using balanced class weights",
            "   - LSTM: Using pos_weight in loss function",
            "",
            "✅ 2. Improved Hyperparameters",
            "   - Random Forest: Increased to 200 trees, depth 20",
            "   - LSTM: Increased to 3 layers, 128 hidden units",
            "   - LSTM: Increased dropout to 0.3",
            "   - LSTM: Increased learning rate to 0.005",
            "",
            "✅ 3. Better Sequence Modeling",
            "   - Reduced sequence length from 10 to 5",
            "   - Increased batch size to 128",
            "   - More epochs (100) with better early stopping potential",
            "",
            "✅ 4. Focus on Classification",
            "   - Regression models skipped (unfeasible with current data)",
            "   - All efforts on improving delay prediction (binary)",
            "",
            "📈 EXPECTED IMPROVEMENTS:",
            "   - Better recall (catching more delayed deliveries)",
            "   - More balanced precision/recall trade-off",
            "   - LSTM should perform much better than 5% recall",
            "   - Overall better F1-scores",
            ""
        ]
        
        for line in improvements:
            print(line)
    
    def get_summary_statistics(self):
        if self.data is None:
            print("No data processed yet. Run preprocess_data() first.")
            return
        
        train_df = self.data['train_df']
        test_df = self.data['test_df']
        
        print("\n" + "=" * 80)
        print("DATA SUMMARY STATISTICS")
        print("=" * 80)
        
        print(f"\nTotal routes in train: {train_df['route_id'].nunique()}")
        print(f"Total routes in test: {test_df['route_id'].nunique()}")
        print(f"Total stops in train: {len(train_df)}")
        print(f"Total stops in test: {len(test_df)}")
        
        print(f"\nDelay rate (train): {train_df['delayed_flag'].mean():.2%}")
        print(f"Delay rate (test): {test_df['delayed_flag'].mean():.2%}")
        
        print(f"\nAverage delay (train): {train_df['delay_minutes'].mean():.4f} minutes ({train_df['delay_minutes'].mean()*60:.2f} seconds)")
        print(f"Average delay (test): {test_df['delay_minutes'].mean():.4f} minutes ({test_df['delay_minutes'].mean()*60:.2f} seconds)")
        
        print(f"\nMax delay (train): {train_df['delay_minutes'].max():.4f} minutes ({train_df['delay_minutes'].max()*60:.2f} seconds)")
        print(f"Max delay (test): {test_df['delay_minutes'].max():.4f} minutes ({test_df['delay_minutes'].max()*60:.2f} seconds)")


def main():
    print("\n" + "=" * 80)
    print("AI-DRIVEN ROUTE OPTIMIZATION - IMPROVED TRAINING")
    print("=" * 80)
    
    data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"\nError: Data file not found at {data_path}")
        print("Please ensure the cleaned data file exists.")
        return
    
    trainer = ImprovedModelTrainer(str(data_path), output_dir="outputs_improved")
    
    trainer.preprocess_data()
    trainer.get_summary_statistics()
    
    # Train only classification models with improved parameters
    trainer.train_classification_models(
        lstm_epochs=100,
        lstm_batch_size=128,
        lstm_sequence_length=5
    )
    
    print("\n" + "=" * 80)
    print("IMPROVED MODEL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

