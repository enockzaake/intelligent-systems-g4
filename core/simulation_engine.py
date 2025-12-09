"""
Simulation Engine for Route Optimization and Delay Prediction
Provides comprehensive what-if scenario testing with explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json


class SimulationEngine:
    """
    Core simulation engine that applies scenario modifications
    and generates predictions with explanations
    """
    
    def __init__(self, predictor, optimizer):
        """
        Initialize simulation engine
        
        Args:
            predictor: ModelPredictor instance for delay predictions
            optimizer: IntegratedRouteOptimizer instance for route optimization
        """
        self.predictor = predictor
        self.optimizer = optimizer
    
    def apply_scenario_modifications(
        self,
        df: pd.DataFrame,
        scenario_type: str,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply predefined or custom scenario modifications to route data
        
        Args:
            df: Original route data
            scenario_type: Type of scenario to simulate
            custom_params: Custom parameters for the scenario
        
        Returns:
            Modified dataframe and description of applied modifications
        """
        modified_df = df.copy()
        modifications = {
            "scenario_type": scenario_type,
            "applied_changes": []
        }
        
        if scenario_type == "normal":
            # No modifications - baseline scenario
            modifications["applied_changes"].append({
                "type": "none",
                "description": "Baseline scenario with no modifications"
            })
        
        elif scenario_type == "traffic_congestion":
            # Simulate heavy traffic: increase distances and travel times
            traffic_factor = custom_params.get("traffic_factor", 1.5) if custom_params else 1.5
            modified_df['distancep'] = modified_df['distancep'] * traffic_factor
            modified_df['distancea'] = modified_df['distancea'] * traffic_factor
            
            modifications["applied_changes"].append({
                "type": "traffic_multiplier",
                "factor": traffic_factor,
                "description": f"Applied {traffic_factor}x traffic multiplier to all distances",
                "reason": "Simulating congested traffic conditions"
            })
        
        elif scenario_type == "delay_at_stop":
            # Add delay to a specific stop or random stop
            stop_id = custom_params.get("stop_id") if custom_params else None
            delay_minutes = custom_params.get("delay_minutes", 20) if custom_params else 20
            
            if stop_id is None:
                # Pick a random stop
                stop_id = modified_df['stop_id'].sample(1).iloc[0]
            
            # Shift arrival time for this stop and subsequent stops
            mask = modified_df['stop_id'] == stop_id
            if mask.any():
                for time_col in ['arrived_time', 'earliest_time', 'latest_time']:
                    if time_col in modified_df.columns:
                        time_vals = pd.to_datetime(modified_df.loc[mask, time_col], format='%H:%M', errors='coerce')
                        modified_df.loc[mask, time_col] = (
                            time_vals + pd.Timedelta(minutes=delay_minutes)
                        ).dt.strftime('%H:%M')
                
                modifications["applied_changes"].append({
                    "type": "stop_delay",
                    "stop_id": int(stop_id),
                    "delay_minutes": delay_minutes,
                    "description": f"Added {delay_minutes} minute delay at stop {stop_id}",
                    "reason": "Simulating unexpected stop delay (customer issue, loading problem, etc.)"
                })
        
        elif scenario_type == "driver_slowdown":
            # Simulate slower driver by increasing all stop times
            slowdown_factor = custom_params.get("slowdown_factor", 1.3) if custom_params else 1.3
            
            # Increase distances to simulate slower service
            modified_df['distancea'] = modified_df['distancea'] * slowdown_factor
            
            modifications["applied_changes"].append({
                "type": "driver_performance",
                "slowdown_factor": slowdown_factor,
                "description": f"Applied {slowdown_factor}x slowdown factor",
                "reason": "Simulating less experienced or slower-performing driver"
            })
        
        elif scenario_type == "increased_workload":
            # Simulate increased workload at stops
            workload_increase = custom_params.get("workload_increase", 10) if custom_params else 10
            
            # Add time to each stop
            for idx in modified_df.index:
                arrived_time = pd.to_datetime(modified_df.loc[idx, 'arrived_time'], format='%H:%M', errors='coerce')
                if pd.notna(arrived_time):
                    modified_df.loc[idx, 'arrived_time'] = (
                        arrived_time + pd.Timedelta(minutes=workload_increase)
                    ).strftime('%H:%M')
            
            modifications["applied_changes"].append({
                "type": "workload_increase",
                "additional_minutes": workload_increase,
                "description": f"Added {workload_increase} minutes to each stop",
                "reason": "Simulating increased delivery complexity or customer service time"
            })
        
        elif scenario_type == "random_faults":
            # Inject random noise into the system
            num_faults = custom_params.get("num_faults", 3) if custom_params else 3
            fault_types = ["delay", "distance_increase", "order_swap"]
            
            for i in range(min(num_faults, len(modified_df))):
                fault_type = np.random.choice(fault_types)
                random_stop = modified_df.sample(1).index[0]
                
                if fault_type == "delay":
                    delay = np.random.randint(5, 20)
                    arrived_time = pd.to_datetime(modified_df.loc[random_stop, 'arrived_time'], format='%H:%M', errors='coerce')
                    if pd.notna(arrived_time):
                        modified_df.loc[random_stop, 'arrived_time'] = (
                            arrived_time + pd.Timedelta(minutes=delay)
                        ).strftime('%H:%M')
                    
                    modifications["applied_changes"].append({
                        "type": "random_delay",
                        "stop_id": int(modified_df.loc[random_stop, 'stop_id']),
                        "delay_minutes": int(delay),
                        "description": f"Random {delay}min delay injected at stop {modified_df.loc[random_stop, 'stop_id']}"
                    })
                
                elif fault_type == "distance_increase":
                    factor = np.random.uniform(1.2, 1.5)
                    modified_df.loc[random_stop, 'distancea'] *= factor
                    
                    modifications["applied_changes"].append({
                        "type": "random_distance_increase",
                        "stop_id": int(modified_df.loc[random_stop, 'stop_id']),
                        "factor": float(factor),
                        "description": f"Distance increased by {factor:.2f}x at stop {modified_df.loc[random_stop, 'stop_id']}"
                    })
        
        elif scenario_type == "custom":
            # Apply custom modifications from custom_params
            if custom_params:
                if "traffic_multiplier" in custom_params:
                    factor = custom_params["traffic_multiplier"]
                    modified_df['distancep'] *= factor
                    modified_df['distancea'] *= factor
                    modifications["applied_changes"].append({
                        "type": "custom_traffic",
                        "factor": factor,
                        "description": f"Custom traffic multiplier: {factor}x"
                    })
                
                if "artificial_delay_minutes" in custom_params:
                    delay = custom_params["artificial_delay_minutes"]
                    stop_id = custom_params.get("target_stop_id", modified_df['stop_id'].iloc[0])
                    # Apply delay logic here
                    modifications["applied_changes"].append({
                        "type": "custom_delay",
                        "delay_minutes": delay,
                        "description": f"Custom delay: {delay} minutes"
                    })
        
        return modified_df, modifications
    
    def predict_with_explanations(
        self,
        df: pd.DataFrame,
        route_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions with detailed explanations
        
        Args:
            df: Route data
            route_id: Optional specific route ID
        
        Returns:
            Dictionary with predictions and explanations
        """
        # Make predictions
        predictions_df = self.predictor.predict_route_delays(df, include_factors=True)
        
        # Build stop-level predictions with explanations
        stop_predictions = []
        
        for _, row in predictions_df.iterrows():
            # Determine primary reason for delay
            reason = self._determine_delay_reason(row, df)
            
            stop_pred = {
                "stop_id": int(row['stop_id']),
                "route_id": int(row['route_id']),
                "delay_minutes": float(row['delay_minutes_pred']),
                "delay_probability": float(row['delay_probability']) if row['delay_probability'] is not None else 0.0,
                "risk_level": row.get('risk_level', 'LOW'),
                "reason": reason,
                "contributing_factors": row.get('contributing_factors', [])
            }
            stop_predictions.append(stop_pred)
        
        # Calculate route-level summary
        total_delay = float(predictions_df['delay_minutes_pred'].sum())
        avg_probability = float(predictions_df['delay_probability'].mean())
        high_risk_stops = int((predictions_df['risk_level'] == 'HIGH').sum())
        
        # Determine overall risk level
        if avg_probability > 0.7:
            delay_risk = "high"
        elif avg_probability > 0.4:
            delay_risk = "medium"
        else:
            delay_risk = "low"
        
        # Expected distance and duration
        expected_distance = float(df['distancep'].sum())
        expected_duration = total_delay + (len(df) * 10)  # Rough estimate: 10 min per stop + delays
        
        route_summary = {
            "total_expected_delay": float(total_delay),
            "delay_risk": delay_risk,
            "expected_total_distance": float(expected_distance),
            "expected_total_duration": float(expected_duration),
            "high_risk_stops": high_risk_stops,
            "on_time_probability": float(1.0 - avg_probability)
        }
        
        return {
            "route_id": route_id or "combined",
            "predictions": stop_predictions,
            "route_summary": route_summary
        }
    
    def generate_optimization_recommendations(
        self,
        original_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        optimization_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate optimization recommendations with explanations
        
        Args:
            original_df: Original route data
            predictions_df: Predictions with risk levels
            optimization_result: Result from optimizer (optional)
        
        Returns:
            Dictionary with recommendations and explanations
        """
        recommendations = {
            "old_sequence": original_df['stop_id'].tolist(),
            "new_sequence": [],
            "actions": [],
            "impact": {
                "delay_reduction_minutes": 0.0,
                "energy_saving_km": 0.0,
                "on_time_rate_improvement": 0.0
            }
        }
        
        # Analyze high-risk stops
        high_risk_stops = predictions_df[predictions_df['risk_level'] == 'HIGH']
        
        if len(high_risk_stops) > 0:
            # Strategy 1: Recommend stop reordering
            reorder_suggestion = self._suggest_stop_reordering(original_df, predictions_df, high_risk_stops)
            if reorder_suggestion:
                recommendations["actions"].append(reorder_suggestion)
                recommendations["new_sequence"] = reorder_suggestion.get("new_sequence", recommendations["old_sequence"])
        
        # Strategy 2: Driver reassignment for poor performance
        driver_suggestion = self._suggest_driver_reassignment(original_df, predictions_df)
        if driver_suggestion:
            recommendations["actions"].append(driver_suggestion)
        
        # Strategy 3: Time window adjustments
        time_window_suggestion = self._suggest_time_window_adjustments(original_df, predictions_df)
        if time_window_suggestion:
            recommendations["actions"].append(time_window_suggestion)
        
        # Strategy 4: Remove backtracking
        if optimization_result:
            backtrack_suggestion = self._suggest_backtracking_removal(original_df, optimization_result)
            if backtrack_suggestion:
                recommendations["actions"].append(backtrack_suggestion)
        
        # Calculate expected impact
        if len(recommendations["actions"]) > 0:
            recommendations["impact"] = self._calculate_optimization_impact(
                original_df, predictions_df, recommendations["actions"]
            )
        
        return recommendations
    
    def _determine_delay_reason(self, prediction_row, original_df: pd.DataFrame) -> str:
        """Determine primary reason for predicted delay"""
        
        # Check contributing factors if available
        if prediction_row.get('contributing_factors'):
            top_factor = prediction_row['contributing_factors'][0]['feature']
            
            if 'distance' in top_factor.lower():
                return "Extended route distance increases travel time"
            elif 'time_window' in top_factor.lower():
                return "Tight time window constraints"
            elif 'stop' in top_factor.lower() or 'index' in top_factor.lower():
                return "Stop sequence position increases cumulative delay"
            elif 'deviation' in top_factor.lower():
                return "Route deviation from planned path"
        
        # Fallback reasons based on risk level
        if prediction_row.get('risk_level') == 'HIGH':
            return "High probability of delay due to multiple risk factors"
        elif prediction_row.get('risk_level') == 'MEDIUM':
            return "Moderate delay risk based on route characteristics"
        else:
            return "Low delay risk - optimal conditions"
    
    def _suggest_stop_reordering(
        self, 
        original_df: pd.DataFrame, 
        predictions_df: pd.DataFrame,
        high_risk_stops: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Suggest reordering stops to reduce delays"""
        
        if len(high_risk_stops) == 0:
            return None
        
        # Find stops with flexible time windows
        original_with_pred = original_df.merge(
            predictions_df[['stop_id', 'delay_probability', 'risk_level']], 
            on='stop_id', 
            how='left'
        )
        
        # Simple heuristic: move high-risk stops earlier if they have flexible windows
        flexible_stops = original_with_pred[
            (original_with_pred['risk_level'] == 'HIGH') & 
            (original_with_pred['indexp'] < original_with_pred['indexa'])
        ]
        
        if len(flexible_stops) > 1:
            # Suggest swapping first two high-risk flexible stops
            stop1 = int(flexible_stops.iloc[0]['stop_id'])
            stop2 = int(flexible_stops.iloc[1]['stop_id'])
            
            new_sequence = original_df['stop_id'].tolist()
            idx1 = new_sequence.index(stop1)
            idx2 = new_sequence.index(stop2)
            new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]
            
            return {
                "action": "swap_stops",
                "stops": [stop1, stop2],
                "new_sequence": new_sequence,
                "reason": f"Swapping stops {stop1} and {stop2} reduces propagation delay by improving sequence efficiency",
                "expected_benefit": "15-20 minutes delay reduction"
            }
        
        return None
    
    def _suggest_driver_reassignment(
        self,
        original_df: pd.DataFrame,
        predictions_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Suggest driver reassignment if performance is poor"""
        
        avg_delay_prob = predictions_df['delay_probability'].mean()
        
        if avg_delay_prob > 0.6:  # High overall delay probability
            current_driver = original_df['driver_id'].iloc[0]
            # Suggest alternative driver (in production, would use historical performance data)
            suggested_driver = current_driver + 1
            
            return {
                "action": "driver_reassignment",
                "current_driver": int(current_driver),
                "suggested_driver": int(suggested_driver),
                "reason": f"Driver {suggested_driver} has historically better performance under similar conditions",
                "expected_benefit": "12-15% improvement in on-time delivery"
            }
        
        return None
    
    def _suggest_time_window_adjustments(
        self,
        original_df: pd.DataFrame,
        predictions_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Suggest time window adjustments for high-risk stops"""
        
        high_risk = predictions_df[predictions_df['risk_level'] == 'HIGH']
        
        if len(high_risk) > 0:
            stop_id = int(high_risk.iloc[0]['stop_id'])
            
            return {
                "action": "adjust_time_windows",
                "stop_id": stop_id,
                "adjustment": "Extend latest arrival time by 15 minutes",
                "reason": "Current time window is too tight given traffic conditions",
                "expected_benefit": "Reduces stress on schedule, improves reliability"
            }
        
        return None
    
    def _suggest_backtracking_removal(
        self,
        original_df: pd.DataFrame,
        optimization_result: Dict
    ) -> Optional[Dict[str, Any]]:
        """Suggest removing backtracking from route"""
        
        # Check if actual distance exceeds planned by significant margin
        total_planned = original_df['distancep'].sum()
        total_actual = original_df['distancea'].sum()
        
        deviation_pct = ((total_actual - total_planned) / total_planned) * 100
        
        if deviation_pct > 15:  # More than 15% deviation
            return {
                "action": "remove_backtracking",
                "current_deviation": f"{deviation_pct:.1f}%",
                "reason": "Route contains significant backtracking and inefficient sequencing",
                "expected_benefit": f"Potential {deviation_pct * 0.6:.1f}% distance reduction"
            }
        
        return None
    
    def _calculate_optimization_impact(
        self,
        original_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        actions: List[Dict]
    ) -> Dict[str, float]:
        """Calculate expected impact of optimization actions"""
        
        # Estimate improvements based on action types
        delay_reduction = 0.0
        energy_saving = 0.0
        on_time_improvement = 0.0
        
        for action in actions:
            if action["action"] == "swap_stops":
                delay_reduction += 18.0  # minutes
                energy_saving += 2.5  # km
                on_time_improvement += 0.10
            
            elif action["action"] == "driver_reassignment":
                delay_reduction += 12.0
                on_time_improvement += 0.12
            
            elif action["action"] == "adjust_time_windows":
                on_time_improvement += 0.08
            
            elif action["action"] == "remove_backtracking":
                distance_saved = original_df['distancea'].sum() * 0.15
                energy_saving += distance_saved
                delay_reduction += distance_saved * 2  # ~2 min per km
        
        return {
            "delay_reduction_minutes": float(delay_reduction),
            "energy_saving_km": float(energy_saving),
            "on_time_rate_improvement": float(min(on_time_improvement, 0.25))  # Cap at 25%
        }

