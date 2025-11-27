import sys
from pathlib import Path
import pandas as pd
import numpy as np

from data_preprocessing import DataPreprocessor
from models import LogisticRegressionModel, RandomForestModel, LSTMModel
from evaluate import ModelEvaluator
import utils


def test_data_preprocessing():
    utils.print_section_header("Testing Data Preprocessing")
    
    data_path = "data/cleaned_delivery_data.csv"
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        return False
    
    preprocessor = DataPreprocessor(data_path)
    
    print("Loading data...")
    df = preprocessor.load_data()
    print(f"Loaded {len(df)} records")
    
    print("\nCreating features...")
    df_features = preprocessor.create_features(df)
    print(f"Created features. Shape: {df_features.shape}")
    
    print("\nEncoding categorical variables...")
    df_encoded = preprocessor.encode_categorical(df_features, fit=True)
    print(f"Encoded. Shape: {df_encoded.shape}")
    
    print("\nSplitting by routes...")
    train_df, test_df = preprocessor.split_by_routes(df_encoded)
    print(f"Train routes: {train_df['route_id'].nunique()}, Test routes: {test_df['route_id'].nunique()}")
    
    print("\nPreparing features for modeling...")
    X_train, y_train, feature_cols = preprocessor.prepare_features(
        train_df, "delayed_flag", fit_scaler=True
    )
    X_test, y_test, _ = preprocessor.prepare_features(
        test_df, "delayed_flag", fit_scaler=False
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    print("\nPreparing LSTM sequences...")
    sequences, targets_clf, targets_reg = preprocessor.prepare_lstm_sequences(train_df, sequence_length=5)
    print(f"Sequences shape: {sequences.shape}")
    print(f"Targets (clf) shape: {targets_clf.shape}")
    print(f"Targets (reg) shape: {targets_reg.shape}")
    
    print("\n✓ Data preprocessing tests passed!")
    return True


def test_baseline_models():
    utils.print_section_header("Testing Baseline Models")
    
    data_path = "data/cleaned_delivery_data.csv"
    preprocessor = DataPreprocessor(data_path)
    data = preprocessor.process_full_pipeline()
    
    X_train_clf = data['classification']['X_train'][:1000]
    y_train_clf = data['classification']['y_train'][:1000]
    X_test_clf = data['classification']['X_test'][:200]
    y_test_clf = data['classification']['y_test'][:200]
    
    X_train_reg = data['regression']['X_train'][:1000]
    y_train_reg = data['regression']['y_train'][:1000]
    X_test_reg = data['regression']['X_test'][:200]
    y_test_reg = data['regression']['y_test'][:200]
    
    print("\n1. Testing Logistic Regression...")
    lr_model = LogisticRegressionModel(max_iter=500)
    lr_model.train(X_train_clf, y_train_clf)
    lr_metrics = lr_model.evaluate(X_test_clf, y_test_clf)
    print(f"   Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"   F1-Score: {lr_metrics['f1_score']:.4f}")
    
    print("\n2. Testing Random Forest Classifier...")
    rf_clf = RandomForestModel(task="classification", n_estimators=50)
    rf_clf.train(X_train_clf, y_train_clf)
    rf_clf_metrics = rf_clf.evaluate(X_test_clf, y_test_clf)
    print(f"   Accuracy: {rf_clf_metrics['accuracy']:.4f}")
    print(f"   F1-Score: {rf_clf_metrics['f1_score']:.4f}")
    
    print("\n3. Testing Random Forest Regressor...")
    rf_reg = RandomForestModel(task="regression", n_estimators=50)
    rf_reg.train(X_train_reg, y_train_reg)
    rf_reg_metrics = rf_reg.evaluate(X_test_reg, y_test_reg)
    print(f"   MAE: {rf_reg_metrics['mae']:.4f}")
    print(f"   RMSE: {rf_reg_metrics['rmse']:.4f}")
    print(f"   R²: {rf_reg_metrics['r2']:.4f}")
    
    print("\n✓ Baseline models tests passed!")
    return True


def test_lstm_model():
    utils.print_section_header("Testing LSTM Model")
    
    data_path = "data/cleaned_delivery_data.csv"
    preprocessor = DataPreprocessor(data_path)
    preprocessor.process_full_pipeline()
    
    train_df = preprocessor.load_data().head(500)
    train_df = preprocessor.create_features(train_df)
    train_df = preprocessor.encode_categorical(train_df, fit=True)
    
    print("\nPreparing sequences...")
    sequences, targets_clf, targets_reg = preprocessor.prepare_lstm_sequences(
        train_df, sequence_length=5
    )
    
    train_size = int(0.8 * len(sequences))
    X_train = sequences[:train_size]
    X_test = sequences[train_size:]
    y_train_clf = targets_clf[:train_size]
    y_test_clf = targets_clf[train_size:]
    y_train_reg = targets_reg[:train_size]
    y_test_reg = targets_reg[train_size:]
    
    input_size = sequences.shape[2]
    
    print("\n1. Testing LSTM Classifier...")
    lstm_clf = LSTMModel(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        task="classification",
        learning_rate=0.01
    )
    lstm_clf.train(X_train, y_train_clf, epochs=5, batch_size=32, verbose=False)
    lstm_clf_metrics = lstm_clf.evaluate(X_test, y_test_clf)
    print(f"   Accuracy: {lstm_clf_metrics['accuracy']:.4f}")
    print(f"   F1-Score: {lstm_clf_metrics['f1_score']:.4f}")
    
    print("\n2. Testing LSTM Regressor...")
    lstm_reg = LSTMModel(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        task="regression",
        learning_rate=0.01
    )
    lstm_reg.train(X_train, y_train_reg, epochs=5, batch_size=32, verbose=False)
    lstm_reg_metrics = lstm_reg.evaluate(X_test, y_test_reg)
    print(f"   MAE: {lstm_reg_metrics['mae']:.4f}")
    print(f"   RMSE: {lstm_reg_metrics['rmse']:.4f}")
    
    print("\n✓ LSTM models tests passed!")
    return True


def test_evaluator():
    utils.print_section_header("Testing Model Evaluator")
    
    evaluator = ModelEvaluator(results_dir="test_results")
    
    print("\nAdding sample results...")
    evaluator.add_result("Model A", "classification", {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85,
        "roc_auc": 0.90
    })
    
    evaluator.add_result("Model B", "classification", {
        "accuracy": 0.87,
        "precision": 0.85,
        "recall": 0.86,
        "f1_score": 0.855,
        "roc_auc": 0.91
    })
    
    evaluator.add_result("Model A", "regression", {
        "mae": 5.2,
        "mse": 45.3,
        "rmse": 6.73,
        "r2": 0.75
    })
    
    print("\nCreating comparison report...")
    evaluator.create_comparison_report()
    
    print("\nCreating comparison table...")
    evaluator.create_comparison_table()
    
    print("\n✓ Evaluator tests passed!")
    
    import shutil
    if Path("test_results").exists():
        shutil.rmtree("test_results")
    
    return True


def test_save_load():
    utils.print_section_header("Testing Model Save/Load")
    
    data_path = "data/cleaned_delivery_data.csv"
    preprocessor = DataPreprocessor(data_path)
    data = preprocessor.process_full_pipeline()
    
    X_train = data['classification']['X_train'][:500]
    y_train = data['classification']['y_train'][:500]
    X_test = data['classification']['X_test'][:100]
    y_test = data['classification']['y_test'][:100]
    
    test_dir = Path("test_models")
    test_dir.mkdir(exist_ok=True)
    
    print("\n1. Testing Logistic Regression save/load...")
    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    pred_before = model.predict(X_test)
    
    model.save(test_dir / "test_lr.pkl")
    
    loaded_model = LogisticRegressionModel()
    loaded_model.load(test_dir / "test_lr.pkl")
    pred_after = loaded_model.predict(X_test)
    
    assert np.array_equal(pred_before, pred_after), "Predictions don't match!"
    print("   ✓ Logistic Regression save/load successful")
    
    print("\n2. Testing Random Forest save/load...")
    rf_model = RandomForestModel(task="classification", n_estimators=10)
    rf_model.train(X_train, y_train)
    pred_before = rf_model.predict(X_test)
    
    rf_model.save(test_dir / "test_rf.pkl")
    
    loaded_rf = RandomForestModel(task="classification")
    loaded_rf.load(test_dir / "test_rf.pkl")
    pred_after = loaded_rf.predict(X_test)
    
    assert np.array_equal(pred_before, pred_after), "Predictions don't match!"
    print("   ✓ Random Forest save/load successful")
    
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    print("\n✓ Save/load tests passed!")
    return True


def run_all_tests():
    utils.print_section_header("RUNNING ALL TESTS", width=100)
    
    tests = [
        ("Data Preprocessing", test_data_preprocessing),
        ("Baseline Models", test_baseline_models),
        ("LSTM Models", test_lstm_model),
        ("Model Evaluator", test_evaluator),
        ("Save/Load", test_save_load),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*100}")
            print(f"Running: {test_name}")
            print(f"{'='*100}")
            
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = "FAILED"
    
    utils.print_section_header("TEST RESULTS SUMMARY", width=100)
    
    for test_name, status in results.items():
        status_symbol = "✓" if status == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {status}")
    
    all_passed = all(status == "PASSED" for status in results.values())
    
    if all_passed:
        print("\n" + "="*100)
        print("ALL TESTS PASSED!".center(100))
        print("="*100)
    else:
        print("\n" + "="*100)
        print("SOME TESTS FAILED".center(100))
        print("="*100)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

