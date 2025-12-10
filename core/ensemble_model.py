"""
Ensemble Model - Combines Random Forest and LSTM for optimal performance
"""
import numpy as np
import pickle
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

class EnsembleClassifier:
    """
    Ensemble classifier that combines Random Forest and LSTM predictions
    """
    
    def __init__(self, rf_model=None, lstm_model=None, strategy='voting', threshold=0.5):
        """
        Initialize ensemble model
        
        Args:
            rf_model: Trained Random Forest model
            lstm_model: Trained LSTM model
            strategy: 'voting' (majority vote), 'weighted' (weighted average), or 'max_recall' (if either predicts delay)
            threshold: Decision threshold for probability-based strategies
        """
        self.rf_model = rf_model
        self.lstm_model = lstm_model
        self.strategy = strategy
        self.threshold = threshold
        self.trained = (rf_model is not None and lstm_model is not None)
        
        # Weights for weighted strategy (can be tuned based on validation performance)
        self.rf_weight = 0.7  # Random Forest is more reliable (higher F1)
        self.lstm_weight = 0.3
    
    def predict(self, X_rf, X_lstm):
        """
        Make predictions using ensemble strategy
        
        Args:
            X_rf: Features for Random Forest (2D array)
            X_lstm: Sequences for LSTM (3D array)
        
        Returns:
            Ensemble predictions
        """
        if not self.trained:
            raise ValueError("Models not loaded. Call load() first.")
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X_rf)
        lstm_pred = self.lstm_model.predict(X_lstm)
        
        if self.strategy == 'voting':
            # Majority vote (if at least one predicts delay for max_recall)
            return np.maximum(rf_pred, lstm_pred)
        
        elif self.strategy == 'max_recall':
            # If either model predicts delay, predict delay (prioritize recall)
            return np.logical_or(rf_pred, lstm_pred).astype(int)
        
        elif self.strategy == 'weighted':
            # Weighted average of probabilities
            rf_proba = self.rf_model.predict_proba(X_rf)[:, 1]
            lstm_proba = self.lstm_model.predict_proba(X_lstm)
            
            weighted_proba = (self.rf_weight * rf_proba + self.lstm_weight * lstm_proba)
            return (weighted_proba > self.threshold).astype(int)
        
        elif self.strategy == 'conservative':
            # Both models must agree for positive prediction (prioritize precision)
            return np.logical_and(rf_pred, lstm_pred).astype(int)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def predict_proba(self, X_rf, X_lstm):
        """
        Predict probabilities using weighted average
        
        Args:
            X_rf: Features for Random Forest
            X_lstm: Sequences for LSTM
        
        Returns:
            Probability of delay
        """
        if not self.trained:
            raise ValueError("Models not loaded. Call load() first.")
        
        rf_proba = self.rf_model.predict_proba(X_rf)[:, 1]
        lstm_proba = self.lstm_model.predict_proba(X_lstm)
        
        return self.rf_weight * rf_proba + self.lstm_weight * lstm_proba
    
    def evaluate(self, X_rf, X_lstm, y_test):
        """
        Evaluate ensemble performance
        
        Args:
            X_rf: Test features for Random Forest
            X_lstm: Test sequences for LSTM
            y_test: True labels
        
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X_rf, X_lstm)
        probabilities = self.predict_proba(X_rf, X_lstm)
        
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
        }
        
        return metrics
    
    def load_models(self, rf_path, lstm_path, preprocessor_dir):
        """
        Load trained models from disk
        
        Args:
            rf_path: Path to Random Forest model
            lstm_path: Path to LSTM model
            preprocessor_dir: Directory with preprocessor files
        """
        # Load Random Forest
        with open(rf_path, 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Load LSTM
        from models.lstm_model import LSTMModel
        from data_preprocessing import DataPreprocessor
        
        # Get input size from preprocessor
        preprocessor = DataPreprocessor(data_path="")
        preprocessor.load_preprocessor(preprocessor_dir)
        input_size = len(preprocessor.feature_columns)
        
        # Create LSTM model with same architecture
        self.lstm_model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            task="classification",
            learning_rate=0.005
        )
        self.lstm_model.load(lstm_path)
        
        self.trained = True
    
    def compare_strategies(self, X_rf, X_lstm, y_test):
        """
        Compare all ensemble strategies
        
        Args:
            X_rf: Test features for Random Forest
            X_lstm: Test sequences for LSTM  
            y_test: True labels
        
        Returns:
            Dictionary with results for each strategy
        """
        strategies = ['voting', 'weighted', 'max_recall', 'conservative']
        results = {}
        
        for strategy in strategies:
            original_strategy = self.strategy
            self.strategy = strategy
            
            metrics = self.evaluate(X_rf, X_lstm, y_test)
            results[strategy] = metrics
            
            self.strategy = original_strategy
        
        return results


def test_ensemble():
    """Test the ensemble model on the test set"""
    
    print("\n" + "=" * 80)
    print("ENSEMBLE MODEL EVALUATION")
    print("=" * 80)
    
    # Load data
    from data_preprocessing import DataPreprocessor
    
    data_path = "data/cleaned_delivery_data.csv"
    preprocessor = DataPreprocessor(data_path)
    data = preprocessor.process_full_pipeline()
    
    # Prepare test data
    X_test_rf = data['classification']['X_test']
    y_test = data['classification']['y_test']
    
    # Prepare LSTM sequences
    test_sequences, test_targets, _ = preprocessor.prepare_lstm_sequences(
        data['test_df'], sequence_length=5
    )
    
    # Load models
    ensemble = EnsembleClassifier()
    ensemble.load_models(
        rf_path="outputs_improved/models/random_forest_classifier_improved.pkl",
        lstm_path="outputs_improved/models/lstm_classifier_improved.pth",
        preprocessor_dir="outputs_improved/preprocessor"
    )
    
    print("\n📊 Testing Different Ensemble Strategies:")
    print("=" * 80)
    
    # Compare all strategies
    results = ensemble.compare_strategies(X_test_rf, test_sequences, y_test)
    
    # Display results
    for strategy, metrics in results.items():
        print(f"\n🔹 Strategy: {strategy.upper()}")
        print("-" * 60)
        
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                print(f"  {metric:12}: {value:.4f}")
        
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            print(f"\n  Confusion Matrix:")
            print(f"    TN: {cm[0][0]:>6,}  |  FP: {cm[0][1]:>6,}")
            print(f"    FN: {cm[1][0]:>6,}  |  TP: {cm[1][1]:>6,}")
    
    # Find best strategy
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    best_f1_strategy = max(results.items(), key=lambda x: x[1]['f1_score'])
    best_recall_strategy = max(results.items(), key=lambda x: x[1]['recall'])
    best_precision_strategy = max(results.items(), key=lambda x: x[1]['precision'])
    
    print(f"\n🏆 Best F1-Score: {best_f1_strategy[0]} (F1: {best_f1_strategy[1]['f1_score']:.4f})")
    print(f"🎯 Best Recall: {best_recall_strategy[0]} (Recall: {best_recall_strategy[1]['recall']:.4f})")
    print(f"🔍 Best Precision: {best_precision_strategy[0]} (Precision: {best_precision_strategy[1]['precision']:.4f})")
    
    print("\n💡 Use Cases:")
    print("  • For balanced performance: Use 'weighted' or 'voting'")
    print("  • To minimize missed delays: Use 'max_recall'")
    print("  • To minimize false alarms: Use 'conservative'")
    
    # Compare with individual models
    print("\n" + "=" * 80)
    print("COMPARISON WITH INDIVIDUAL MODELS")
    print("=" * 80)
    
    # Get individual model performance
    rf_pred = ensemble.rf_model.predict(X_test_rf)
    rf_f1 = f1_score(y_test, rf_pred)
    
    lstm_pred = ensemble.lstm_model.predict(test_sequences)
    lstm_f1 = f1_score(y_test, lstm_pred)
    
    print(f"\n  Random Forest F1:     {rf_f1:.4f}")
    print(f"  LSTM F1:              {lstm_f1:.4f}")
    print(f"  Ensemble (weighted):  {results['weighted']['f1_score']:.4f}")
    
    if results['weighted']['f1_score'] > max(rf_f1, lstm_f1):
        print(f"\n  ✅ Ensemble improves over individual models!")
    else:
        print(f"\n  ℹ️  Random Forest alone may be sufficient")


if __name__ == "__main__":
    test_ensemble()

