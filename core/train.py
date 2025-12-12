import numpy as np
from pathlib import Path
import json
from datetime import datetime

from data_preprocessing import DataPreprocessor
from models import LogisticRegressionModel, RandomForestModel, LSTMModel
from evaluate import ModelEvaluator
from config import Config


class ModelTrainer:
    def __init__(self, data_path, output_dir="outputs"):
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
    
    def preprocess_data(self):
        print("=" * 80)
        print("PREPROCESSING DATA")
        print("=" * 80)
        
        self.data = self.preprocessor.process_full_pipeline()
        
        print(f"\nTraining samples (classification): {len(self.data['classification']['X_train'])}")
        print(f"Test samples (classification): {len(self.data['classification']['X_test'])}")
        print(f"Number of features: {self.data['classification']['X_train'].shape[1]}")
        print(f"Feature columns: {len(self.data['feature_columns'])}")
        
        self.preprocessor.save_preprocessor(self.preprocessor_dir)
        print(f"\nPreprocessor saved to {self.preprocessor_dir}")
    
    def train_logistic_regression(self):
        print("\n" + "=" * 80)
        print("TRAINING LOGISTIC REGRESSION (Classification)")
        print("=" * 80)
        
        model = LogisticRegressionModel(max_iter=1000, random_state=42)
        
        X_train = self.data['classification']['X_train']
        y_train = self.data['classification']['y_train']
        X_test = self.data['classification']['X_test']
        y_test = self.data['classification']['y_test']
        
        print("Training model...")
        model.train(X_train, y_train)
        
        print("Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        
        print("\nMetrics:")
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                print(f"  {metric}: {value:.4f}")
        
        model_path = self.models_dir / "logistic_regression.pkl"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        self.models["Logistic Regression"] = model
        self.evaluator.add_result("Logistic Regression", "classification", metrics)
        
        self.evaluator.plot_feature_importance(
            model, 
            self.data['feature_columns'], 
            "Logistic Regression"
        )
        
        return model, metrics
    
    def train_random_forest_classifier(self):
        print("\n" + "=" * 80)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("=" * 80)
        
        model = RandomForestModel(
            task="classification",
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        
        X_train = self.data['classification']['X_train']
        y_train = self.data['classification']['y_train']
        X_test = self.data['classification']['X_test']
        y_test = self.data['classification']['y_test']
        
        print("Training model...")
        model.train(X_train, y_train)
        
        print("Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        
        print("\nMetrics:")
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                print(f"  {metric}: {value:.4f}")
        
        model_path = self.models_dir / "random_forest_classifier.pkl"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        self.models["Random Forest Classifier"] = model
        self.evaluator.add_result("Random Forest Classifier", "classification", metrics)
        
        self.evaluator.plot_feature_importance(
            model,
            self.data['feature_columns'],
            "Random Forest Classifier"
        )
        
        return model, metrics
    
    def train_random_forest_regressor(self):
        print("\n" + "=" * 80)
        print("TRAINING RANDOM FOREST REGRESSOR")
        print("=" * 80)
        
        model = RandomForestModel(
            task="regression",
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        
        X_train = self.data['regression']['X_train']
        y_train = self.data['regression']['y_train']
        X_test = self.data['regression']['X_test']
        y_test = self.data['regression']['y_test']
        
        print("Training model...")
        model.train(X_train, y_train)
        
        print("Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        model_path = self.models_dir / "random_forest_regressor.pkl"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        self.models["Random Forest Regressor"] = model
        self.evaluator.add_result("Random Forest Regressor", "regression", metrics)
        
        self.evaluator.plot_feature_importance(
            model,
            self.data['feature_columns'],
            "Random Forest Regressor"
        )
        
        return model, metrics
    
    def train_lstm_classifier(self, sequence_length=5, epochs=100, batch_size=128):
        print("\n" + "=" * 80)
        print("TRAINING LSTM CLASSIFIER")
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
        
        # Check if sequences were generated
        if len(train_sequences) == 0 or len(test_sequences) == 0:
            print("\n⚠️  No sequences generated - skipping LSTM training")
            print("   This can happen if routes don't have multiple stops")
            print("   LSTM requires sequential data within routes")
            return None, None, None
        
        input_size = train_sequences.shape[2]
        
        # Calculate pos_weight for class imbalance
        y_train = self.data['classification']['y_train']
        n_class_0 = np.sum(y_train == 0)
        n_class_1 = np.sum(y_train == 1)
        pos_weight = (len(y_train) / (2 * n_class_1)) / (len(y_train) / (2 * n_class_0))
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            task="classification",
            learning_rate=0.005,
            pos_weight=pos_weight
        )
        
        print("\nTraining model...")
        history = model.train(
            train_sequences, train_targets_clf,
            X_val=test_sequences, y_val=test_targets_clf,
            epochs=epochs, batch_size=batch_size, verbose=True
        )
        
        print("\nEvaluating model...")
        metrics = model.evaluate(test_sequences, test_targets_clf, batch_size=batch_size)
        
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        model_path = self.models_dir / "lstm_classifier.pth"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        self.models["LSTM Classifier"] = model
        self.evaluator.add_result("LSTM Classifier", "classification", metrics)
        
        return model, metrics, history
    
    def train_lstm_regressor(self, sequence_length=10, epochs=50, batch_size=64):
        print("\n" + "=" * 80)
        print("TRAINING LSTM REGRESSOR")
        print("=" * 80)
        
        print(f"Preparing sequences (sequence_length={sequence_length})...")
        
        train_sequences, _, train_targets_reg = self.preprocessor.prepare_lstm_sequences(
            self.data['train_df'], sequence_length
        )
        test_sequences, _, test_targets_reg = self.preprocessor.prepare_lstm_sequences(
            self.data['test_df'], sequence_length
        )
        
        print(f"Train sequences: {train_sequences.shape}")
        print(f"Test sequences: {test_sequences.shape}")
        
        input_size = train_sequences.shape[2]
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            task="regression",
            learning_rate=0.001
        )
        
        print("\nTraining model...")
        history = model.train(
            train_sequences, train_targets_reg,
            X_val=test_sequences, y_val=test_targets_reg,
            epochs=epochs, batch_size=batch_size, verbose=True
        )
        
        print("\nEvaluating model...")
        metrics = model.evaluate(test_sequences, test_targets_reg, batch_size=batch_size)
        
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        model_path = self.models_dir / "lstm_regressor.pth"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        self.models["LSTM Regressor"] = model
        self.evaluator.add_result("LSTM Regressor", "regression", metrics)
        
        return model, metrics, history
    
    def train_all_models(self, lstm_epochs=100, lstm_batch_size=128, lstm_sequence_length=5):
        print("\n" + "=" * 80)
        print("TRAINING ALL MODELS (OPTIMIZED)")
        print("=" * 80)
        
        if self.data is None:
            self.preprocess_data()
        
        # Calculate class weights
        y_train = self.data['classification']['y_train']
        n_samples = len(y_train)
        n_class_0 = np.sum(y_train == 0)
        n_class_1 = np.sum(y_train == 1)
        weight_0 = n_samples / (2 * n_class_0)
        weight_1 = n_samples / (2 * n_class_1)
        pos_weight = weight_1 / weight_0
        
        print(f"\n📊 Class Distribution:")
        print(f"  Class 0: {n_class_0:,} ({n_class_0/n_samples:.2%})")
        print(f"  Class 1: {n_class_1:,} ({n_class_1/n_samples:.2%})")
        print(f"  Weight ratio: {pos_weight:.2f}")
        
        # Train classification models with class weights
        self.train_logistic_regression()
        self.train_random_forest_classifier()
        
        # Train LSTM with improved settings
        print("\n🚀 Training LSTM with optimized parameters...")
        self.train_lstm_classifier(
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
        
        print(f"\nAverage delay (train): {train_df['delay_minutes'].mean():.2f} minutes")
        print(f"Average delay (test): {test_df['delay_minutes'].mean():.2f} minutes")
        
        print(f"\nMax delay (train): {train_df['delay_minutes'].max():.2f} minutes")
        print(f"Max delay (test): {test_df['delay_minutes'].max():.2f} minutes")


def main():
    data_path = "data/cleaned_delivery_data.csv"
    
    trainer = ModelTrainer(data_path, output_dir="outputs")
    
    trainer.preprocess_data()
    trainer.get_summary_statistics()
    
    trainer.train_all_models(
        lstm_epochs=50,
        lstm_batch_size=64,
        lstm_sequence_length=10
    )


if __name__ == "__main__":
    main()

