import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import torch

from data_preprocessing import DataPreprocessor
from models import LogisticRegressionModel, RandomForestModel, LSTMModel


class ModelPredictor:
    def __init__(self, models_dir="outputs/models", preprocessor_dir="outputs/preprocessor"):
        self.models_dir = Path(models_dir)
        self.preprocessor_dir = Path(preprocessor_dir)
        
        self.preprocessor = DataPreprocessor("")
        self.preprocessor.load_preprocessor(self.preprocessor_dir)
        
        self.models = {}
    
    def load_model(self, model_name):
        if model_name == "logistic_regression":
            model = LogisticRegressionModel()
            model.load(self.models_dir / "logistic_regression.pkl")
            self.models["logistic_regression"] = model
        
        elif model_name == "random_forest_classifier":
            model = RandomForestModel(task="classification")
            model.load(self.models_dir / "random_forest_classifier.pkl")
            self.models["random_forest_classifier"] = model
        
        elif model_name == "random_forest_regressor":
            model = RandomForestModel(task="regression")
            model.load(self.models_dir / "random_forest_regressor.pkl")
            self.models["random_forest_regressor"] = model
        
        elif model_name == "lstm_classifier":
            checkpoint = torch.load(
                self.models_dir / "lstm_classifier.pth",
                map_location=torch.device("cpu")
            )
            model = LSTMModel(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
                task="classification",
                learning_rate=checkpoint["learning_rate"]
            )
            model.load(self.models_dir / "lstm_classifier.pth")
            self.models["lstm_classifier"] = model
        
        elif model_name == "lstm_regressor":
            checkpoint = torch.load(
                self.models_dir / "lstm_regressor.pth",
                map_location=torch.device("cpu")
            )
            model = LSTMModel(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
                task="regression",
                learning_rate=checkpoint["learning_rate"]
            )
            model.load(self.models_dir / "lstm_regressor.pth")
            self.models["lstm_regressor"] = model
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        print(f"Loaded model: {model_name}")
        return model
    
    def load_all_models(self):
        model_names = [
            "logistic_regression",
            "random_forest_classifier",
            "random_forest_regressor",
            "lstm_classifier",
            "lstm_regressor"
        ]
        
        for name in model_names:
            try:
                self.load_model(name)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
    
    def preprocess_data(self, df):
        df = self.preprocessor.create_features(df)
        df = self.preprocessor.encode_categorical(df, fit=False)
        return df
    
    def predict_classification(self, df, model_name="random_forest_classifier"):
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        df_processed = self.preprocess_data(df)
        X, _, _ = self.preprocessor.prepare_features(
            df_processed, "delayed_flag", fit_scaler=False
        )
        
        predictions = model.predict(X)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            if probabilities is not None and len(probabilities.shape) > 1:
                probabilities = probabilities[:, 1]
            return predictions, probabilities
        
        return predictions, None
    
    def predict_regression(self, df, model_name="random_forest_regressor"):
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        df_processed = self.preprocess_data(df)
        X, _, _ = self.preprocessor.prepare_features(
            df_processed, "delay_minutes", fit_scaler=False
        )
        
        predictions = model.predict(X)
        return predictions
    
    def predict_route_delays(self, df):
        df_processed = self.preprocess_data(df)
        
        clf_predictions, clf_probabilities = self.predict_classification(
            df, model_name="random_forest_classifier"
        )
        
        reg_predictions = self.predict_regression(
            df, model_name="random_forest_regressor"
        )
        
        results = df_processed[["route_id", "stop_id", "driver_id"]].copy()
        results["delayed_flag_pred"] = clf_predictions
        results["delay_probability"] = clf_probabilities
        results["delay_minutes_pred"] = reg_predictions
        
        return results
    
    def predict_and_aggregate_routes(self, df):
        predictions = self.predict_route_delays(df)
        
        route_aggregates = predictions.groupby("route_id").agg({
            "delayed_flag_pred": "mean",
            "delay_probability": "mean",
            "delay_minutes_pred": ["sum", "mean", "max"]
        }).reset_index()
        
        route_aggregates.columns = [
            "route_id",
            "route_delay_rate",
            "avg_delay_probability",
            "total_delay_minutes",
            "avg_delay_minutes",
            "max_delay_minutes"
        ]
        
        return route_aggregates


def main():
    predictor = ModelPredictor(
        models_dir="outputs/models",
        preprocessor_dir="outputs/preprocessor"
    )
    
    predictor.load_all_models()
    
    print("\nAll models loaded successfully!")
    print(f"Available models: {list(predictor.models.keys())}")
    
    sample_data_path = Path("data/cleaned_delivery_data.csv")
    if sample_data_path.exists():
        df = pd.read_csv(sample_data_path)
        sample_df = df.head(100)
        
        print("\nMaking predictions on sample data...")
        predictions = predictor.predict_route_delays(sample_df)
        
        print("\nSample predictions:")
        print(predictions.head(10))
        
        route_aggregates = predictor.predict_and_aggregate_routes(sample_df)
        print("\nRoute-level aggregates:")
        print(route_aggregates.head(5))


if __name__ == "__main__":
    main()

