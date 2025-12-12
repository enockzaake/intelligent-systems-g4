"""
Simple Model Testing Interface
Test trained models and run simulations
"""
import pandas as pd
import numpy as np
from pathlib import Path
from predict import ModelPredictor


def test_models():
    """Test all trained models"""
    
    print("\n" + "=" * 80)
    print("MODEL TESTING & SIMULATION")
    print("=" * 80)
    
    # Check if models exist
    models_dir = Path("outputs/models")
    if not models_dir.exists() or not any(models_dir.glob("*.pkl")):
        print("\n❌ No trained models found!")
        print("   Please train models first: python main.py")
        return
    
    # Load test data
    data_path = Path("data/synthetic_delivery_data.csv")
    if not data_path.exists():
        data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print("\n❌ No data found!")
        return
    
    print(f"\n📂 Loading data from: {data_path.name}")
    df = pd.read_csv(data_path)
    
    # Use a sample for testing
    sample_size = min(1000, len(df))
    test_df = df.sample(n=sample_size, random_state=42)
    
    print(f"📊 Testing on {sample_size} samples...")
    
    # Initialize predictor
    predictor = ModelPredictor()
    
    # Test each model
    models_to_test = [
        ("random_forest_classifier", "Random Forest"),
        ("logistic_regression", "Logistic Regression"),
        ("lstm_classifier", "LSTM")
    ]
    
    results = {}
    
    for model_name, display_name in models_to_test:
        try:
            print(f"\n{'='*60}")
            print(f"Testing: {display_name}")
            print('='*60)
            
            # Load model
            predictor.load_model(model_name)
            
            # Make predictions
            predictions, probabilities = predictor.predict_classification(
                test_df, model_name=model_name
            )
            
            # Calculate statistics
            delay_rate = predictions.mean()
            
            if probabilities is not None:
                avg_confidence = probabilities.mean()
                high_risk = (probabilities > 0.7).sum()
                medium_risk = ((probabilities > 0.4) & (probabilities <= 0.7)).sum()
                low_risk = (probabilities <= 0.4).sum()
            else:
                avg_confidence = None
                high_risk = medium_risk = low_risk = 0
            
            results[display_name] = {
                'predictions': predictions,
                'probabilities': probabilities,
                'delay_rate': delay_rate,
                'avg_confidence': avg_confidence,
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk
            }
            
            print(f"✅ Predictions made: {len(predictions)}")
            print(f"📊 Predicted delay rate: {delay_rate:.2%}")
            if avg_confidence:
                print(f"🎯 Average confidence: {avg_confidence:.2%}")
                print(f"   High risk: {high_risk} ({high_risk/len(predictions):.1%})")
                print(f"   Medium risk: {medium_risk} ({medium_risk/len(predictions):.1%})")
                print(f"   Low risk: {low_risk} ({low_risk/len(predictions):.1%})")
            
        except Exception as e:
            print(f"❌ Failed to test {display_name}: {e}")
            results[display_name] = None
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    comparison_data = []
    for model_name, result in results.items():
        if result:
            comparison_data.append({
                'Model': model_name,
                'Predicted Delays': f"{result['delay_rate']:.1%}",
                'High Risk': result['high_risk'],
                'Medium Risk': result['medium_risk'],
                'Low Risk': result['low_risk']
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        print("\n" + comp_df.to_string(index=False))
    
    # Simulation: What-if scenarios
    print("\n" + "=" * 80)
    print("SIMULATION: WHAT-IF SCENARIOS")
    print("=" * 80)
    
    print("\n📊 Scenario 1: Morning Rush Hour")
    morning_routes = test_df[test_df['arrived_time'].str[:2].astype(int).between(7, 9)]
    if len(morning_routes) > 0:
        predictor.load_model("random_forest_classifier")
        morning_preds, morning_probs = predictor.predict_classification(morning_routes)
        print(f"   Samples: {len(morning_routes)}")
        print(f"   Expected delays: {morning_preds.mean():.1%}")
        print(f"   Avg risk: {morning_probs.mean():.1%}" if morning_probs is not None else "")
    
    print("\n📊 Scenario 2: Long Distance Deliveries")
    long_distance = test_df[test_df['distancep'] > test_df['distancep'].quantile(0.75)]
    if len(long_distance) > 0:
        predictor.load_model("random_forest_classifier")
        distance_preds, distance_probs = predictor.predict_classification(long_distance)
        print(f"   Samples: {len(long_distance)}")
        print(f"   Expected delays: {distance_preds.mean():.1%}")
        print(f"   Avg risk: {distance_probs.mean():.1%}" if distance_probs is not None else "")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    
    print("\n📁 Results available in: outputs/results/")
    print("🌐 Launch dashboard: cd dashboard && npm run dev")


if __name__ == "__main__":
    test_models()

