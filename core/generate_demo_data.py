"""
Demonstration Data Generator
Creates realistic delivery data optimized for showcasing different scenarios:
- Normal conditions (on-time deliveries)
- Traffic congestion (moderate delays)
- Severe delays (major incidents)
- Route optimization scenarios
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path


class DemoDataGenerator:
    """
    Generate demonstration data with realistic delay patterns
    Optimized for showcasing ML predictions and route optimization
    """
    
    def __init__(self, seed=42):
        """Initialize generator with random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        
        # Configuration
        self.countries = ['Netherlands', 'Spain', 'Italy', 'Germany', 'UK', 'France', 'USA']
        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Driver IDs (format: D####)
        self.driver_ids = [f"D{i:04d}" for i in range(1, 101)]
        
        # Scenario-based delay distributions
        self.scenario_delays = {
            'normal': {'mean': 2.0, 'std': 3.0, 'max': 10.0, 'delay_rate': 0.15},  # 15% delay rate
            'traffic': {'mean': 8.0, 'std': 6.0, 'max': 25.0, 'delay_rate': 0.40},  # 40% delay rate
            'severe': {'mean': 20.0, 'std': 15.0, 'max': 60.0, 'delay_rate': 0.70},  # 70% delay rate
            'optimized': {'mean': 0.5, 'std': 1.5, 'max': 5.0, 'delay_rate': 0.05},  # 5% delay rate
        }
    
    def generate_route(self, route_id, num_stops=12, scenario='normal', driver_id=None, country=None, day=None):
        """
        Generate a single route with specified scenario
        
        Args:
            route_id: Route identifier
            num_stops: Number of stops in route
            scenario: 'normal', 'traffic', 'severe', or 'optimized'
            driver_id: Specific driver ID (optional)
            country: Specific country (optional)
            day: Specific day of week (optional)
        """
        if driver_id is None:
            driver_id = random.choice(self.driver_ids)
        if country is None:
            country = random.choice(self.countries)
        if day is None:
            day = random.choice(self.days_of_week)
        
        # Get delay parameters for scenario
        delay_params = self.scenario_delays[scenario]
        
        # Generate route
        stops = []
        cumulative_distance_p = 0
        cumulative_distance_a = 0
        current_time = datetime(2024, 1, 1, 7, 0, 0)  # Start at 7 AM
        
        for stop_idx in range(num_stops):
            # Determine if this is depot or delivery
            is_depot = (stop_idx == 0)
            is_delivery = not is_depot
            
            # Generate distances
            if is_depot:
                distance_p = 0
                distance_a = 0
            else:
                # Distance to next stop (5-20 km)
                segment_distance = np.random.uniform(5, 20)
                cumulative_distance_p += segment_distance
                
                # Actual distance may differ (route deviation)
                deviation_factor = np.random.uniform(0.85, 1.15)  # ±15% deviation
                cumulative_distance_a = cumulative_distance_p * deviation_factor
            
            # Time windows
            service_time = np.random.uniform(5, 15)  # 5-15 minutes per stop
            travel_time = cumulative_distance_p / 50.0 * 60  # Assume 50 km/h average speed
            
            earliest_time = current_time + timedelta(minutes=travel_time)
            time_window_length = np.random.choice([30, 60, 90], p=[0.3, 0.5, 0.2])  # 30/60/90 min windows
            latest_time = earliest_time + timedelta(minutes=time_window_length)
            
            # Generate delay based on scenario
            will_delay = np.random.random() < delay_params['delay_rate']
            
            if will_delay:
                # Generate delay from scenario distribution
                delay_minutes = np.random.normal(delay_params['mean'], delay_params['std'])
                delay_minutes = max(0, min(delay_minutes, delay_params['max']))
            else:
                # On-time or early (negative delay)
                delay_minutes = np.random.uniform(-5, 0)
            
            # Actual arrival time
            arrived_time = latest_time + timedelta(minutes=delay_minutes)
            
            # Convert delay to normalized format (fraction of day)
            actual_arrival_delay = delay_minutes / (24 * 60)  # Convert to days
            
            # Stop IDs
            stop_id = f"S{route_id}_{stop_idx}"
            
            # Planned vs actual index (may differ due to reordering)
            indexp = stop_idx
            if scenario == 'optimized' and stop_idx > 0:
                # Optimized routes may have different ordering
                indexa = indexp + np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
                indexa = max(0, min(indexa, num_stops - 1))
            else:
                indexa = indexp + np.random.choice([-2, -1, 0, 1, 2], p=[0.05, 0.1, 0.7, 0.1, 0.05])
                indexa = max(0, min(indexa, num_stops - 1))
            
            stop_data = {
                'route_id': route_id,
                'stop_id': stop_id,
                'driver_id': driver_id,
                'address_id': stop_idx,
                'week_id': 1,
                'country': country,
                'day_of_week': day,
                'indexp': indexp,
                'indexa': indexa,
                'arrived_time': arrived_time.strftime('%H:%M:%S'),
                'earliest_time': earliest_time.strftime('%H:%M:%S'),
                'latest_time': latest_time.strftime('%H:%M:%S'),
                'distancep': round(cumulative_distance_p, 2),
                'distancea': round(cumulative_distance_a, 2),
                'depot': 1 if is_depot else 0,
                'delivery': 1 if is_delivery else 0,
                'actual_arrival_delay': actual_arrival_delay,
                'delay_flag': 1 if delay_minutes > 0 else 0,
                'delay_minutes': round(max(0, delay_minutes), 2)  # Clamp negative to 0
            }
            
            stops.append(stop_data)
            
            # Update current time for next stop
            current_time = arrived_time + timedelta(minutes=service_time)
        
        return stops
    
    def generate_dataset(self, num_routes=200, scenarios_distribution=None):
        """
        Generate complete dataset with mixed scenarios
        
        Args:
            num_routes: Total number of routes to generate
            scenarios_distribution: Dict mapping scenario to proportion (default: balanced)
        """
        if scenarios_distribution is None:
            scenarios_distribution = {
                'normal': 0.50,      # 50% normal
                'traffic': 0.30,     # 30% traffic
                'severe': 0.15,      # 15% severe
                'optimized': 0.05    # 5% optimized
            }
        
        all_stops = []
        route_id = 1
        
        # Generate routes according to distribution
        scenario_list = []
        for scenario, proportion in scenarios_distribution.items():
            count = int(num_routes * proportion)
            scenario_list.extend([scenario] * count)
        
        # Fill remaining routes with normal
        while len(scenario_list) < num_routes:
            scenario_list.append('normal')
        
        # Shuffle to mix scenarios
        random.shuffle(scenario_list)
        
        for scenario in scenario_list[:num_routes]:
            # Vary number of stops per route (8-15 stops)
            num_stops = np.random.randint(8, 16)
            
            route_stops = self.generate_route(
                route_id=route_id,
                num_stops=num_stops,
                scenario=scenario
            )
            
            all_stops.extend(route_stops)
            route_id += 1
        
        df = pd.DataFrame(all_stops)
        
        # Sort by route_id and indexp
        df = df.sort_values(['route_id', 'indexp']).reset_index(drop=True)
        
        return df


def main():
    """Generate demonstration dataset"""
    generator = DemoDataGenerator(seed=42)
    
    print("Generating demonstration dataset...")
    print("=" * 60)
    
    # Generate dataset with good mix of scenarios
    df = generator.generate_dataset(
        num_routes=200,
        scenarios_distribution={
            'normal': 0.50,      # 50% normal conditions
            'traffic': 0.30,     # 30% traffic congestion
            'severe': 0.15,      # 15% severe delays
            'optimized': 0.05    # 5% optimized routes
        }
    )
    
    # Save to file
    output_path = Path(__file__).parent / "data" / "demo_delivery_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset generated successfully!")
    print(f"Total routes: {df['route_id'].nunique()}")
    print(f"Total stops: {len(df)}")
    print(f"Average stops per route: {len(df) / df['route_id'].nunique():.1f}")
    print(f"\nDelay statistics:")
    print(f"  Delay rate: {df['delay_flag'].mean() * 100:.1f}%")
    print(f"  Average delay (when delayed): {df[df['delay_minutes'] > 0]['delay_minutes'].mean():.1f} min")
    print(f"  Max delay: {df['delay_minutes'].max():.1f} min")
    print(f"  Min delay: {df['delay_minutes'].min():.1f} min")
    print(f"\nSaved to: {output_path}")
    
    # Show sample routes
    print("\n" + "=" * 60)
    print("Sample Routes:")
    print("=" * 60)
    for route_id in df['route_id'].unique()[:3]:
        route_df = df[df['route_id'] == route_id]
        delays = route_df[route_df['delay_minutes'] > 0]
        print(f"\nRoute {route_id}:")
        print(f"  Driver: {route_df['driver_id'].iloc[0]}")
        print(f"  Country: {route_df['country'].iloc[0]}")
        print(f"  Day: {route_df['day_of_week'].iloc[0]}")
        print(f"  Stops: {len(route_df)}")
        print(f"  Total distance: {route_df['distancep'].max():.1f} km")
        print(f"  Delays: {len(delays)} stops")
        if len(delays) > 0:
            print(f"  Avg delay: {delays['delay_minutes'].mean():.1f} min")
            print(f"  Max delay: {delays['delay_minutes'].max():.1f} min")


if __name__ == "__main__":
    main()

