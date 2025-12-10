"""
LSTM Training with Precision Focus
Adjusts hyperparameters to improve precision while maintaining reasonable recall
"""
import numpy as np
from pathlib import Path
from data_preprocessing import DataPreprocessor
from models import LSTMModel
from evaluate import ModelEvaluator
import warnings
warnings.filterwarnings('ignore')


def train_precision_focused_lstm():
    """Train LSTM with focus on improving precision"""
    
    print("\n" + "=" * 80)
    print("LSTM CLASSIFIER - PRECISION FOCUSED TRAINING")
    print("=" * 80)
    
    # Load data
    data_path = "data/cleaned_delivery_data.csv"
    preprocessor = DataPreprocessor(data_path)
    data = preprocessor.process_full_pipeline()
    
    # Calculate class weights (reduce pos_weight for better precision)
    y_train = data['classification']['y_train']
    n_samples = len(y_train)
    n_class_0 = np.sum(y_train == 0)
    n_class_1 = np.sum(y_train == 1)
    
    weight_0 = n_samples / (2 * n_class_0)
    weight_1 = n_samples / (2 * n_class_1)
    
    # Reduce pos_weight for better precision (use 60% of full weight)
    pos_weight = (weight_1 / weight_0) * 0.6
    
    print(f"\n📊 Training Configuration:")
    print(f"  Class 0 (On-time): {n_class_0:,} ({n_class_0/n_samples:.2%})")
    print(f"  Class 1 (Delayed): {n_class_1:,} ({n_class_1/n_samples:.2%})")
    print(f"  Adjusted pos_weight: {pos_weight:.4f} (reduced for precision)")
    
    # Prepare sequences
    print(f"\nPreparing sequences (sequence_length=5)...")
    train_sequences, train_targets_clf, _ = preprocessor.prepare_lstm_sequences(
        data['train_df'], sequence_length=5
    )
    test_sequences, test_targets_clf, _ = preprocessor.prepare_lstm_sequences(
        data['test_df'], sequence_length=5
    )
    
    print(f"Train sequences: {train_sequences.shape}")
    print(f"Test sequences: {test_sequences.shape}")
    
    input_size = train_sequences.shape[2]
    
    # Create model with adjusted hyperparameters
    model = LSTMModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        dropout=0.4,  # Increased from 0.3 for better regularization
        task="classification",
        learning_rate=0.003,  # Reduced from 0.005 for more stable training
        pos_weight=pos_weight
    )
    
    print("\n🎯 Training Strategy:")
    print("  • Reduced pos_weight to balance precision/recall")
    print("  • Increased dropout (0.4) for better generalization")
    print("  • Reduced learning rate (0.003) for stability")
    print("  • Will use higher prediction threshold (0.6)")
    
    # Train model
    print("\n🚀 Training model...")
    history = model.train(
        train_sequences, train_targets_clf,
        X_val=test_sequences, y_val=test_targets_clf,
        epochs=100, batch_size=128, verbose=True
    )
    
    # Evaluate with standard threshold
    print("\n📊 Evaluation with threshold=0.5 (standard):")
    metrics_05 = model.evaluate(test_sequences, test_targets_clf, batch_size=128)
    
    for metric, value in metrics_05.items():
        if metric != "confusion_matrix":
            print(f"  {metric:12}: {value:.4f}")
    
    # Evaluate with higher threshold for better precision
    print("\n📊 Evaluation with threshold=0.6 (precision-focused):")
    
    # Get probabilities
    probabilities = model.predict_proba(test_sequences)
    predictions_06 = (probabilities > 0.6).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    metrics_06 = {
        "accuracy": accuracy_score(test_targets_clf, predictions_06),
        "precision": precision_score(test_targets_clf, predictions_06, zero_division=0),
        "recall": recall_score(test_targets_clf, predictions_06, zero_division=0),
        "f1_score": f1_score(test_targets_clf, predictions_06, zero_division=0),
        "roc_auc": roc_auc_score(test_targets_clf, probabilities),
        "confusion_matrix": confusion_matrix(test_targets_clf, predictions_06)
    }
    
    for metric, value in metrics_06.items():
        if metric != "confusion_matrix":
            print(f"  {metric:12}: {value:.4f}")
    
    cm = metrics_06["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0][0]:>6,}  |  FP: {cm[0][1]:>6,}")
    print(f"    FN: {cm[1][0]:>6,}  |  TP: {cm[1][1]:>6,}")
    
    # Try multiple thresholds
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    
    for threshold in thresholds:
        preds = (probabilities > threshold).astype(int)
        precision = precision_score(test_targets_clf, preds, zero_division=0)
        recall = recall_score(test_targets_clf, preds, zero_division=0)
        f1 = f1_score(test_targets_clf, preds, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        print(f"\nThreshold = {threshold:.1f}")
        print(f"  Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}")
    
    # Find best threshold
    best_f1 = max(results, key=lambda x: x['f1_score'])
    best_balance = max(results, key=lambda x: min(x['precision'], x['recall']))
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\n🏆 Best F1-Score: threshold={best_f1['threshold']:.1f} (F1={best_f1['f1_score']:.4f})")
    print(f"⚖️  Best Balance: threshold={best_balance['threshold']:.1f}")
    
    # Save model
    output_dir = Path("outputs_improved/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "lstm_classifier_precision_focused.pth"
    model.save(model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    # Save recommended threshold
    import json
    config_path = output_dir / "lstm_precision_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'recommended_threshold': best_f1['threshold'],
            'pos_weight': pos_weight,
            'metrics_at_threshold': {
                'precision': best_f1['precision'],
                'recall': best_f1['recall'],
                'f1_score': best_f1['f1_score']
            }
        }, f, indent=2)
    
    print(f"✅ Configuration saved to {config_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    train_precision_focused_lstm()

