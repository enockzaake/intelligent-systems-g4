"""
Inference module for DL Route Optimizer
Provides easy-to-use prediction functions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dl_route_optimizer import DLRouteOptimizer
from typing import Dict, List, Optional
import json


class DLRoutePredictor:
    """
    Easy-to-use predictor for DL route optimization.
    """
    
    def __init__(self, model_path: str = "outputs_v2/dl_models/best_model.pt"):
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize optimizer and load model
        self.optimizer = DLRouteOptimizer(
            feature_dim=14,
            embedding_dim=128,
            num_heads=8,
            num_layers=3,
            max_stops=50
        )
        
        self.optimizer.load_model(self.model_path)
        print(f"✓ DL Route Predictor loaded from {model_path}")
    
    def predict_single_route(self, route_df: pd.DataFrame) -> Dict:
        """
        Predict optimal sequence for a single route.
        
        Args:
            route_df: DataFrame with columns:
                - stop_id, driver_id, country, day_of_week
                - indexp, indexa (planned and actual sequences)
                - distancep, distancea
                - earliest_time, latest_time
                - depot, delivery
                - delay_flag, delay_minutes (optional)
        
        Returns:
            Dictionary with prediction results
        """
        result = self.optimizer.predict_route_sequence(route_df)
        
        if result is None:
            return {
                'success': False,
                'error': 'Failed to generate prediction'
            }
        
        # Compute metrics
        metrics = self._compute_metrics(result, route_df)
        result.update(metrics)
        result['success'] = True
        
        return result
    
    def predict_multiple_routes(self, df: pd.DataFrame, route_ids: Optional[List] = None) -> List[Dict]:
        """
        Predict optimal sequences for multiple routes.
        
        Args:
            df: DataFrame with route data
            route_ids: List of route IDs to predict (if None, predicts all)
        
        Returns:
            List of prediction dictionaries
        """
        if route_ids is None:
            route_ids = df['route_id'].unique()
        
        results = []
        
        for route_id in route_ids:
            route_df = df[df['route_id'] == route_id]
            result = self.predict_single_route(route_df)
            results.append(result)
        
        return results
    
    def _compute_metrics(self, prediction: Dict, route_df: pd.DataFrame) -> Dict:
        """Compute evaluation metrics for prediction."""
        pred_seq = np.array(prediction['predicted_sequence'])
        actual_seq = np.array(prediction['actual_sequence'])
        planned_seq = np.array(prediction['planned_sequence'])
        
        # Sequence accuracy (exact match rate)
        sequence_accuracy = np.mean(pred_seq == actual_seq)
        
        # Kendall's Tau correlation
        from scipy.stats import kendalltau
        tau_pred_actual, _ = kendalltau(pred_seq, actual_seq)
        tau_planned_actual, _ = kendalltau(planned_seq, actual_seq)
        
        # Distance comparison (if available)
        route_df_sorted = route_df.sort_values('indexp').reset_index(drop=True)
        
        metrics = {
            'sequence_accuracy': float(sequence_accuracy),
            'kendall_tau_predicted': float(tau_pred_actual),
            'kendall_tau_planned': float(tau_planned_actual),
            'improvement_over_planned': float(tau_pred_actual - tau_planned_actual),
            'planned_total_distance': float(route_df['distancep'].sum()),
            'actual_total_distance': float(route_df['distancea'].sum())
        }
        
        return metrics
    
    def evaluate_on_test_set(self, test_df: pd.DataFrame, sample_size: int = 100) -> Dict:
        """
        Evaluate model performance on test set.
        
        Args:
            test_df: Test dataset
            sample_size: Number of routes to evaluate (set to None for all)
        
        Returns:
            Dictionary with aggregate metrics
        """
        route_ids = test_df['route_id'].unique()
        
        if sample_size and len(route_ids) > sample_size:
            route_ids = np.random.choice(route_ids, sample_size, replace=False)
        
        print(f"\nEvaluating on {len(route_ids)} routes...")
        
        results = self.predict_multiple_routes(test_df, route_ids)
        
        # Aggregate metrics
        successful = [r for r in results if r.get('success', False)]
        
        if len(successful) == 0:
            return {
                'error': 'No successful predictions',
                'num_routes': len(route_ids)
            }
        
        metrics = {
            'num_routes_evaluated': len(successful),
            'avg_sequence_accuracy': np.mean([r['sequence_accuracy'] for r in successful]),
            'avg_kendall_tau_predicted': np.mean([r['kendall_tau_predicted'] for r in successful]),
            'avg_kendall_tau_planned': np.mean([r['kendall_tau_planned'] for r in successful]),
            'avg_improvement': np.mean([r['improvement_over_planned'] for r in successful]),
            'routes_better_than_planned': sum(1 for r in successful if r['improvement_over_planned'] > 0),
            'routes_worse_than_planned': sum(1 for r in successful if r['improvement_over_planned'] < 0)
        }
        
        metrics['pct_better_than_planned'] = metrics['routes_better_than_planned'] / len(successful)
        
        return metrics
    
    def create_route_visualization_data(self, prediction: Dict, route_df: pd.DataFrame) -> Dict:
        """
        Create data structure for visualization in UI.
        
        Returns:
            Dictionary with data ready for frontend visualization
        """
        route_df_sorted = route_df.sort_values('indexp').reset_index(drop=True)
        
        stops = []
        for idx, row in route_df_sorted.iterrows():
            stop = {
                'stop_id': row['stop_id'],
                'planned_position': int(row['indexp']),
                'actual_position': int(row['indexa']),
                'predicted_position': int(prediction['predicted_sequence'][idx]),
                'is_depot': int(row.get('depot', 0)) == 1,
                'is_delivery': int(row.get('delivery', 0)) == 1,
                'distance_planned': float(row.get('distancep', 0)),
                'distance_actual': float(row.get('distancea', 0)),
                'earliest_time': str(row.get('earliest_time', '')),
                'latest_time': str(row.get('latest_time', '')),
                'confidence': float(prediction['confidence_scores'][idx])
            }
            stops.append(stop)
        
        viz_data = {
            'route_id': prediction['route_id'],
            'num_stops': prediction['num_stops'],
            'stops': stops,
            'metrics': {
                'sequence_accuracy': prediction.get('sequence_accuracy', 0),
                'kendall_tau_predicted': prediction.get('kendall_tau_predicted', 0),
                'kendall_tau_planned': prediction.get('kendall_tau_planned', 0),
                'improvement': prediction.get('improvement_over_planned', 0)
            },
            'sequences': {
                'planned': prediction['planned_sequence'],
                'actual': prediction['actual_sequence'],
                'predicted': prediction['predicted_sequence']
            }
        }
        
        return viz_data


def main():
    """Example usage of DL Route Predictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DL Route Prediction')
    parser.add_argument('--model', type=str, default='outputs_v2/dl_models/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='data/synthetic_delivery_data.csv',
                       help='Path to test data')
    parser.add_argument('--route-id', type=int, default=None,
                       help='Specific route ID to predict')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run full evaluation on test set')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of routes to evaluate')
    parser.add_argument('--output', type=str, default='outputs_v2/predictions',
                       help='Output directory for predictions')
    
    args = parser.parse_args()
    
    # Load predictor
    print("Loading DL Route Predictor...")
    predictor = DLRoutePredictor(args.model)
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} stops from {df['route_id'].nunique():,} routes\n")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.evaluate:
        # Full evaluation
        print("=" * 80)
        print("EVALUATING MODEL ON TEST SET")
        print("=" * 80)
        
        metrics = predictor.evaluate_on_test_set(df, args.sample_size)
        
        print("\nEVALUATION RESULTS:")
        print("-" * 80)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Save results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to: {output_dir / 'evaluation_results.json'}")
    
    elif args.route_id is not None:
        # Predict single route
        print(f"Predicting route {args.route_id}...")
        route_df = df[df['route_id'] == args.route_id]
        
        if len(route_df) == 0:
            print(f"Error: Route {args.route_id} not found")
            return
        
        result = predictor.predict_single_route(route_df)
        
        if result['success']:
            print("\n" + "=" * 80)
            print(f"PREDICTION FOR ROUTE {args.route_id}")
            print("=" * 80)
            print(f"Number of stops: {result['num_stops']}")
            print(f"\nSequence Accuracy: {result['sequence_accuracy']:.2%}")
            print(f"Kendall Tau (Predicted): {result['kendall_tau_predicted']:.4f}")
            print(f"Kendall Tau (Planned): {result['kendall_tau_planned']:.4f}")
            print(f"Improvement over Planned: {result['improvement_over_planned']:.4f}")
            print(f"\nPlanned Sequence:   {result['planned_sequence']}")
            print(f"Actual Sequence:    {result['actual_sequence']}")
            print(f"Predicted Sequence: {result['predicted_sequence']}")
            
            # Save visualization data
            viz_data = predictor.create_route_visualization_data(result, route_df)
            with open(output_dir / f'route_{args.route_id}_prediction.json', 'w') as f:
                json.dump(viz_data, f, indent=2)
            
            print(f"\nVisualization data saved to: {output_dir / f'route_{args.route_id}_prediction.json'}")
        else:
            print(f"Prediction failed: {result.get('error', 'Unknown error')}")
    
    else:
        # Predict first 10 routes as example
        print("Predicting first 10 routes...")
        route_ids = df['route_id'].unique()[:10]
        results = predictor.predict_multiple_routes(df, route_ids)
        
        print("\n" + "=" * 80)
        print("PREDICTIONS SUMMARY")
        print("=" * 80)
        
        for result in results:
            if result['success']:
                print(f"\nRoute {result['route_id']}:")
                print(f"  Sequence Accuracy: {result['sequence_accuracy']:.2%}")
                print(f"  Improvement: {result['improvement_over_planned']:+.4f}")


if __name__ == '__main__':
    main()

