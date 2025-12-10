"""
Improved Prediction Module
Uses the improved models for delay prediction
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import torch
import json

from data_preprocessing import DataPreprocessor
from models import LogisticRegressionModel, RandomForestModel, LSTMModel


class ImprovedModelPredictor:
    """Enhanced predictor using improved models"""
    
    def __init__(self, models_dir="outputs_improved/models", 
                 preprocessor_dir="outputs_improved/preprocessor",
                 use_ensemble=False):
        self.models_dir = Path(models_dir)
        self.preprocessor_dir = Path(preprocessor_dir)
        self.use_ensemble = use_ensemble
        
        self.preprocessor = DataPreprocessor("")
        self.preprocessor.load_preprocessor(self.preprocessor_dir)
        
        self.models = {}
        self.lstm_config = self._load_lstm_config()
    
    def _load_lstm_config(self):
        """Load LSTM configuration if available"""
        config_path = self.models_dir / "lstm_precision_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {'recommended_threshold': 0.5}
    
    def load_model(self, model_name, model_version="improved"):
        """
        Load a specific model
        
        Args:
            model_name: Name of the model to load
            model_version: 'improved' or 'precision_focused' for LSTM
        """
        if model_name == "random_forest_classifier":
            model = RandomForestModel(task="classification")
            model.load(self.models_dir / "random_forest_classifier_improved.pkl")
            self.models["random_forest"] = model
            print(f"✅ Loaded Random Forest Classifier (Improved)")
        
        elif model_name == "logistic_regression":
            model = LogisticRegressionModel()
            model.load(self.models_dir / "logistic_regression_improved.pkl")
            self.models["logistic_regression"] = model
            print(f"✅ Loaded Logistic Regression (Improved)")
        
        elif model_name == "lstm_classifier":
            # Choose version
            if model_version == "precision_focused":
                model_path = self.models_dir / "lstm_classifier_precision_focused.pth"
                if not model_path.exists():
                    print(f"⚠️  Precision-focused LSTM not found, using improved version")
                    model_path = self.models_dir / "lstm_classifier_improved.pth"
            else:
                model_path = self.models_dir / "lstm_classifier_improved.pth"
            
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            model = LSTMModel(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
                task="classification",
                learning_rate=checkpoint["learning_rate"],
                pos_weight=checkpoint.get("pos_weight", None)
            )
            model.load(model_path)
            self.models["lstm"] = model
            print(f"✅ Loaded LSTM Classifier ({model_version})")
        
        elif model_name == "ensemble":
            # Load both RF and LSTM for ensemble
            self.load_model("random_forest_classifier")
            self.load_model("lstm_classifier")
            
            from ensemble_model import EnsembleClassifier
            self.models["ensemble"] = EnsembleClassifier(
                rf_model=self.models["random_forest"],
                lstm_model=self.models["lstm"],
                strategy='weighted',  # Default strategy
                threshold=0.5
            )
            print(f"✅ Loaded Ensemble Model (RF + LSTM)")
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        return self.models.get(model_name.replace("_classifier", "").replace("_improved", ""))
    
    def load_best_model(self):
        """Load the best performing model (Random Forest Improved)"""
        return self.load_model("random_forest_classifier")
    
    def preprocess_data(self, df):
        """Preprocess input data"""
        df = self.preprocessor.create_features(df)
        df = self.preprocessor.encode_categorical(df, fit=False)
        return df
    
    def predict(self, df, model_name="random_forest", return_probabilities=True):
        """
        Make predictions
        
        Args:
            df: DataFrame with delivery data
            model_name: 'random_forest', 'logistic_regression', 'lstm', or 'ensemble'
            return_probabilities: Whether to return probability scores
        
        Returns:
            predictions, probabilities (optional)
        """
        # Load model if not already loaded
        model_key = model_name.replace("_classifier", "")
        if model_key not in self.models:
            self.load_model(f"{model_name}_classifier" if model_name != "ensemble" else "ensemble")
        
        model = self.models[model_key]
        df_processed = self.preprocess_data(df)
        
        # Handle ensemble separately
        if model_name == "ensemble":
            # Prepare data for both models
            X_rf, _, _ = self.preprocessor.prepare_features(
                df_processed, "delayed_flag", fit_scaler=False
            )
            X_lstm, _, _ = self.preprocessor.prepare_lstm_sequences(
                df_processed, sequence_length=5
            )
            
            predictions = model.predict(X_rf, X_lstm)
            if return_probabilities:
                probabilities = model.predict_proba(X_rf, X_lstm)
                return predictions, probabilities
            return predictions, None
        
        # Handle LSTM
        elif model_name == "lstm":
            X_lstm, _, _ = self.preprocessor.prepare_lstm_sequences(
                df_processed, sequence_length=5
            )
            
            # Use recommended threshold if available
            threshold = self.lstm_config.get('recommended_threshold', 0.5)
            probabilities = model.predict_proba(X_lstm)
            predictions = (probabilities > threshold).astype(int)
            
            if return_probabilities:
                return predictions, probabilities
            return predictions, None
        
        # Handle traditional ML models
        else:
            X, _, _ = self.preprocessor.prepare_features(
                df_processed, "delayed_flag", fit_scaler=False
            )
            
            predictions = model.predict(X)
            
            if return_probabilities:
                probabilities = model.predict_proba(X)
                if probabilities is not None and len(probabilities.shape) > 1:
                    probabilities = probabilities[:, 1]
                return predictions, probabilities
            
            return predictions, None
    
    def classify_risk(self, probability):
        """Classify delay risk based on probability"""
        if probability > 0.7:
            return "HIGH"
        elif probability > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def predict_with_confidence(self, df, model_name="random_forest"):
        """
        Make predictions with confidence levels and risk assessment
        
        Returns:
            DataFrame with predictions, probabilities, risk levels, and confidence
        """
        predictions, probabilities = self.predict(df, model_name, return_probabilities=True)
        
        df_processed = self.preprocess_data(df)
        
        results = pd.DataFrame({
            'route_id': df_processed['route_id'].values if 'route_id' in df_processed else range(len(predictions)),
            'stop_id': df_processed['stop_id'].values if 'stop_id' in df_processed else range(len(predictions)),
            'delayed_prediction': predictions,
            'delay_probability': probabilities if probabilities is not None else predictions,
        })
        
        # Add risk classification
        if probabilities is not None:
            results['risk_level'] = [self.classify_risk(p) for p in probabilities]
            
            # Add confidence (distance from decision boundary)
            results['confidence'] = np.abs(probabilities - 0.5) * 2  # 0 to 1 scale
        else:
            results['risk_level'] = 'UNKNOWN'
            results['confidence'] = 0.0
        
        # Add recommended actions
        results['recommended_action'] = results.apply(self._get_recommended_action, axis=1)
        
        return results
    
    def _get_recommended_action(self, row):
        """Get recommended action based on prediction"""
        if row['risk_level'] == 'HIGH':
            return "⚠️ Alert driver, adjust route, add buffer time"
        elif row['risk_level'] == 'MEDIUM':
            return "👀 Monitor closely, be ready to intervene"
        else:
            return "✅ Continue as planned"
    
    def get_feature_importance(self, model_name="random_forest"):
        """Get feature importance from the model"""
        model_key = model_name.replace("_classifier", "")
        if model_key not in self.models:
            return None
        
        model = self.models[model_key]
        if hasattr(model, 'get_feature_importance'):
            if self.preprocessor.feature_columns is not None:
                importance = model.get_feature_importance(self.preprocessor.feature_columns)
                if importance is not None:
                    return dict(importance)
        return None
    
    def explain_prediction(self, df_row, model_name="random_forest", top_n=5):
        """
        Explain a single prediction with contributing factors
        
        Args:
            df_row: Single row DataFrame or Series
            model_name: Model to use
            top_n: Number of top features to show
        
        Returns:
            Dictionary with prediction explanation
        """
        # Convert Series to DataFrame if needed
        if isinstance(df_row, pd.Series):
            df_row = df_row.to_frame().T
        
        # Make prediction
        prediction, probability = self.predict(df_row, model_name, return_probabilities=True)
        
        # Get feature importance
        feature_importance = self.get_feature_importance(model_name)
        
        # Process the row
        df_processed = self.preprocess_data(df_row)
        
        # Get top contributing features
        contributing_factors = []
        if feature_importance:
            top_features = list(feature_importance.keys())[:top_n]
            
            for feature in top_features:
                if feature in df_processed.columns:
                    value = df_processed[feature].iloc[0]
                    importance = feature_importance[feature]
                    contributing_factors.append({
                        "feature": feature,
                        "value": float(value) if not pd.isna(value) else 0.0,
                        "importance": float(importance)
                    })
        
        explanation = {
            "prediction": "Delayed" if prediction[0] == 1 else "On-time",
            "probability": float(probability[0]) if probability is not None else None,
            "risk_level": self.classify_risk(probability[0]) if probability is not None else "UNKNOWN",
            "confidence": float(np.abs(probability[0] - 0.5) * 2) if probability is not None else 0.0,
            "top_contributing_factors": contributing_factors,
            "recommended_action": self._get_recommended_action(pd.Series({
                'risk_level': self.classify_risk(probability[0]) if probability is not None else "UNKNOWN"
            }))
        }
        
        return explanation
    
    def batch_predict_routes(self, df, model_name="random_forest"):
        """
        Predict delays for multiple routes with aggregation
        
        Returns:
            Stop-level predictions and route-level aggregates
        """
        # Get stop-level predictions
        stop_predictions = self.predict_with_confidence(df, model_name)
        
        # Aggregate by route
        route_aggregates = stop_predictions.groupby('route_id').agg({
            'delayed_prediction': 'mean',
            'delay_probability': 'mean',
            'confidence': 'mean'
        }).reset_index()
        
        route_aggregates.columns = [
            'route_id',
            'delay_rate',
            'avg_delay_probability',
            'avg_confidence'
        ]
        
        # Classify route risk
        route_aggregates['route_risk'] = route_aggregates['avg_delay_probability'].apply(self.classify_risk)
        
        # Count high-risk stops per route
        high_risk_stops = stop_predictions[stop_predictions['risk_level'] == 'HIGH'].groupby('route_id').size()
        route_aggregates['high_risk_stops'] = route_aggregates['route_id'].map(high_risk_stops).fillna(0).astype(int)
        
        return stop_predictions, route_aggregates
    
    def compare_models(self, df):
        """
        Compare predictions from all available models
        
        Returns:
            DataFrame with predictions from each model
        """
        # Load all models
        self.load_model("random_forest_classifier")
        self.load_model("logistic_regression")
        self.load_model("lstm_classifier")
        
        comparison = pd.DataFrame()
        
        # Get predictions from each model
        for model_name in ['random_forest', 'logistic_regression', 'lstm']:
            preds, probs = self.predict(df, model_name, return_probabilities=True)
            comparison[f'{model_name}_pred'] = preds
            if probs is not None:
                comparison[f'{model_name}_prob'] = probs
        
        # Add consensus prediction (majority vote)
        comparison['consensus_pred'] = (
            comparison[['random_forest_pred', 'logistic_regression_pred', 'lstm_pred']].sum(axis=1) >= 2
        ).astype(int)
        
        # Add average probability
        prob_cols = [col for col in comparison.columns if col.endswith('_prob')]
        if prob_cols:
            comparison['avg_probability'] = comparison[prob_cols].mean(axis=1)
        
        return comparison


def demo_predictions():
    """Demonstrate the improved predictor"""
    
    print("\n" + "=" * 80)
    print("IMPROVED MODEL PREDICTION DEMO")
    print("=" * 80)
    
    # Load data
    data_path = Path("data/cleaned_delivery_data.csv")
    if not data_path.exists():
        print("❌ Data file not found")
        return
    
    df = pd.read_csv(data_path)
    sample_df = df.head(20)
    
    # Initialize predictor
    predictor = ImprovedModelPredictor()
    
    # Load best model
    print("\n📦 Loading best model (Random Forest Improved)...")
    predictor.load_best_model()
    
    # Make predictions
    print("\n🔮 Making predictions...")
    results = predictor.predict_with_confidence(sample_df, model_name="random_forest")
    
    print("\n📊 Sample Predictions:")
    print(results[['route_id', 'stop_id', 'delayed_prediction', 
                  'delay_probability', 'risk_level', 'recommended_action']].head(10))
    
    # Explain a single prediction
    print("\n" + "=" * 80)
    print("PREDICTION EXPLANATION EXAMPLE")
    print("=" * 80)
    
    explanation = predictor.explain_prediction(sample_df.iloc[0], model_name="random_forest")
    
    print(f"\n🎯 Prediction: {explanation['prediction']}")
    print(f"📊 Probability: {explanation['probability']:.2%}")
    print(f"⚠️  Risk Level: {explanation['risk_level']}")
    print(f"✅ Confidence: {explanation['confidence']:.2%}")
    print(f"💡 Action: {explanation['recommended_action']}")
    
    print(f"\n🔍 Top Contributing Factors:")
    for i, factor in enumerate(explanation['top_contributing_factors'], 1):
        print(f"  {i}. {factor['feature']}: {factor['value']:.4f} (importance: {factor['importance']:.4f})")
    
    # Route-level aggregation
    print("\n" + "=" * 80)
    print("ROUTE-LEVEL ANALYSIS")
    print("=" * 80)
    
    stop_preds, route_agg = predictor.batch_predict_routes(sample_df, model_name="random_forest")
    
    print("\n📍 Route Aggregates:")
    print(route_agg.head())
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo_predictions()

