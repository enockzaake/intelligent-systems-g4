"""
Synthetic Delivery Data Generator
Creates realistic delivery data with learnable patterns and proper variance
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path


class SyntheticDeliveryDataGenerator:
    """
    Generate synthetic delivery data with realistic patterns and variance
    """
    
    def __init__(self, seed=42):
        """Initialize generator with random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        
        # Configuration
        self.countries = ['USA', 'Germany', 'France', 'UK', 'Spain', 'Italy', 'Netherlands']
        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Driver experience levels (affects delay probability)
        self.driver_experience = {
            'novice': {'skill': 0.6, 'ratio': 0.2},      # 20% novice drivers
            'intermediate': {'skill': 0.8, 'ratio': 0.5}, # 50% intermediate
            'expert': {'skill': 0.95, 'ratio': 0.3}       # 30% expert
        }
        
        # Time windows and patterns
        self.time_windows = {
            'tight': {'length': 30, 'delay_prob': 0.25},    # 30-min window, 25% delay rate
            'normal': {'length': 60, 'delay_prob': 0.10},   # 60-min window, 10% delay rate
            'flexible': {'length': 120, 'delay_prob': 0.05} # 120-min window, 5% delay rate
        }
        
        # Traffic patterns by time of day
        self.traffic_patterns = {
            'morning_rush': (7, 9, 1.5),    # 7-9 AM, 1.5x delay factor
            'midday': (9, 16, 1.0),          # 9 AM - 4 PM, normal
            'evening_rush': (16, 19, 1.8),   # 4-7 PM, 1.8x delay factor
            'evening': (19, 22, 1.1),        # 7-10 PM, slight increase
            'night': (22, 7, 0.8)            # 10 PM - 7 AM, easier
        }
        
        # Weather impact (random weather conditions)
        self.weather_impact = {
            'clear': {'prob': 0.6, 'delay_factor': 1.0},
            'rain': {'prob': 0.2, 'delay_factor': 1.4},
            'heavy_rain': {'prob': 0.1, 'delay_factor': 2.0},
            'snow': {'prob': 0.05, 'delay_factor': 2.5},
            'fog': {'prob': 0.05, 'delay_factor': 1.3}
        }
    
    def _get_driver_skill(self):
        """Assign driver skill based on experience distribution"""
        rand = np.random.random()
        cumulative = 0
        for exp_level, config in self.driver_experience.items():
            cumulative += config['ratio']
            if rand <= cumulative:
                return config['skill']
        return 0.8  # Default to intermediate
    
    def _get_traffic_factor(self, hour):
        """Get traffic delay factor based on time of day"""
        for pattern, (start, end, factor) in self.traffic_patterns.items():
            if start < end:
                if start <= hour < end:
                    return factor
            else:  # Night time wraps around
                if hour >= start or hour < end:
                    return factor
        return 1.0
    
    def _get_weather_condition(self):
        """Randomly select weather condition"""
        rand = np.random.random()
        cumulative = 0
        for weather, config in self.weather_impact.items():
            cumulative += config['prob']
            if rand <= cumulative:
                return weather, config['delay_factor']
        return 'clear', 1.0
    
    def _get_time_window_type(self):
        """Randomly assign time window type"""
        types = list(self.time_windows.keys())
        probs = [0.3, 0.5, 0.2]  # 30% tight, 50% normal, 20% flexible
        return np.random.choice(types, p=probs)
    
    def _calculate_delay(self, stop_factors):
        """
        Calculate delay based on multiple factors
        Returns delay in minutes (can be negative for early arrivals)
        """
        base_delay = 0
        
        # Driver skill (better drivers have lower delays)
        driver_skill = stop_factors['driver_skill']
        base_delay += np.random.exponential(scale=5 * (1 - driver_skill))
        
        # Traffic factor
        base_delay *= stop_factors['traffic_factor']
        
        # Weather factor
        base_delay *= stop_factors['weather_factor']
        
        # Time window pressure (tight windows cause more delays)
        window_length = stop_factors['window_length']
        if window_length < 45:
            base_delay *= 1.3
        elif window_length > 90:
            base_delay *= 0.8
        
        # Distance deviation (unexpected distance causes delays)
        distance_dev = stop_factors['distance_deviation']
        if abs(distance_dev) > 0.2:  # >20% deviation
            base_delay += abs(distance_dev) * 10
        
        # Stop position in route (later stops accumulate delays)
        stop_position = stop_factors['stop_position_norm']
        base_delay += stop_position * 3  # Up to 3 min additional delay
        
        # Cumulative delay from previous stops
        base_delay += stop_factors['prev_stop_delay'] * 0.3
        
        # Weekend effect (less traffic, lower delays)
        if not stop_factors['is_weekday']:
            base_delay *= 0.7
        
        # Add some random noise
        base_delay += np.random.normal(0, 2)
        
        # Some deliveries are early (negative delay)
        if np.random.random() < 0.15:  # 15% chance of early arrival
            base_delay = -abs(np.random.exponential(scale=3))
        
        return base_delay
    
    def generate_routes(self, num_routes=20000, avg_stops_per_route=12):
        """
        Generate synthetic delivery routes
        
        Args:
            num_routes: Number of routes to generate
            avg_stops_per_route: Average number of stops per route
        
        Returns:
            DataFrame with synthetic delivery data
        """
        print("\n" + "=" * 80)
        print("GENERATING SYNTHETIC DELIVERY DATA")
        print("=" * 80)
        print(f"\nTarget: {num_routes:,} routes with ~{avg_stops_per_route} stops each")
        
        data = []
        route_id = 0
        
        for route_num in range(num_routes):
            if (route_num + 1) % 1000 == 0:
                print(f"  Generated {route_num + 1:,} / {num_routes:,} routes...")
            
            # Route-level attributes
            country = np.random.choice(self.countries)
            driver_id = f"D{np.random.randint(1, 501):04d}"  # 500 drivers
            driver_skill = self._get_driver_skill()
            day_of_week = np.random.choice(self.days_of_week)
            is_weekday = day_of_week not in ['Saturday', 'Sunday']
            
            # Weather for this route
            weather, weather_factor = self._get_weather_condition()
            
            # Number of stops in this route (Poisson distribution)
            num_stops = max(3, min(25, int(np.random.poisson(avg_stops_per_route))))
            
            # Total route distance (km)
            total_distance = np.random.uniform(20, 150)
            
            # Starting time (between 6 AM and 10 AM)
            start_hour = np.random.uniform(6, 10)
            current_time = start_hour
            
            cumulative_delay = 0
            prev_stop_delay = 0
            
            for stop_idx in range(num_stops):
                route_id += 1
                
                # Stop position
                stop_position_norm = stop_idx / max(1, num_stops - 1)
                
                # Planned vs actual indices
                indexp = stop_idx + 1
                indexa = indexp + np.random.randint(-1, 3)  # Sometimes order changes
                
                # Distance to this stop
                distancep = total_distance * (stop_idx + 1) / num_stops + np.random.normal(0, 2)
                distancep = max(0, distancep)
                
                # Actual distance (can deviate)
                distance_deviation = np.random.normal(0, 0.15)  # ±15% typical deviation
                distancea = distancep * (1 + distance_deviation)
                distancea = max(0, distancea)
                
                # Time window
                window_type = self._get_time_window_type()
                window_config = self.time_windows[window_type]
                window_length = window_config['length']
                
                # Arrival time
                current_time += np.random.uniform(10, 30) / 60  # 10-30 min between stops
                hour_of_day = int(current_time % 24)
                
                # Get traffic factor for this time
                traffic_factor = self._get_traffic_factor(hour_of_day)
                
                # Calculate delay
                stop_factors = {
                    'driver_skill': driver_skill,
                    'traffic_factor': traffic_factor,
                    'weather_factor': weather_factor,
                    'window_length': window_length,
                    'distance_deviation': distance_deviation,
                    'stop_position_norm': stop_position_norm,
                    'prev_stop_delay': prev_stop_delay,
                    'is_weekday': is_weekday
                }
                
                delay_minutes = self._calculate_delay(stop_factors)
                
                # Update for next stop
                if delay_minutes > 0:
                    cumulative_delay += delay_minutes
                    prev_stop_delay = delay_minutes
                else:
                    prev_stop_delay = 0
                
                # Create time values
                base_date = datetime(2024, 1, 1)
                arrived_time = base_date + timedelta(hours=current_time + delay_minutes / 60)
                earliest_time = base_date + timedelta(hours=current_time - window_length / 120)
                latest_time = base_date + timedelta(hours=current_time + window_length / 120)
                
                # Depot and delivery flags
                depot = 1 if stop_idx == 0 else 0
                delivery = 1 if stop_idx > 0 else 0
                
                # Create record
                data.append({
                    'route_id': route_id,
                    'stop_id': f"S{route_id}_{stop_idx}",
                    'driver_id': driver_id,
                    'country': country,
                    'day_of_week': day_of_week,
                    'indexp': indexp,
                    'indexa': indexa,
                    'distancep': round(distancep, 2),
                    'distancea': round(distancea, 2),
                    'depot': depot,
                    'delivery': delivery,
                    'arrived_time': arrived_time.strftime('%H:%M:%S'),
                    'earliest_time': earliest_time.strftime('%H:%M:%S'),
                    'latest_time': latest_time.strftime('%H:%M:%S'),
                    'actual_arrival_delay': delay_minutes / (24 * 60),  # Convert to fraction of day
                    'delay_flag': 1 if delay_minutes > 0 else 0,
                    # Hidden metadata (for analysis, not used in training)
                    '_weather': weather,
                    '_driver_skill': driver_skill,
                    '_traffic_factor': traffic_factor,
                    '_window_type': window_type
                })
        
        df = pd.DataFrame(data)
        
        print(f"\n✅ Generated {len(df):,} stops across {num_routes:,} routes")
        
        return df
    
    def generate_and_save(self, num_routes=20000, avg_stops_per_route=12, 
                         output_path="data/synthetic_delivery_data.csv"):
        """Generate data and save to CSV"""
        
        df = self.generate_routes(num_routes, avg_stops_per_route)
        
        # Print statistics
        self._print_statistics(df)
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove metadata columns before saving
        meta_cols = [col for col in df.columns if col.startswith('_')]
        df_clean = df.drop(columns=meta_cols)
        
        df_clean.to_csv(output_path, index=False)
        
        print(f"\n✅ Data saved to: {output_path.absolute()}")
        
        # Also save with metadata for analysis
        metadata_path = output_path.parent / f"synthetic_delivery_data_with_metadata.csv"
        df.to_csv(metadata_path, index=False)
        print(f"✅ Data with metadata saved to: {metadata_path.absolute()}")
        
        return df
    
    def _print_statistics(self, df):
        """Print statistics about generated data"""
        
        print("\n" + "=" * 80)
        print("GENERATED DATA STATISTICS")
        print("=" * 80)
        
        # Convert delay to minutes for analysis
        df['delay_minutes'] = df['actual_arrival_delay'] * 24 * 60
        
        print(f"\n📊 Basic Stats:")
        print(f"  Total records: {len(df):,}")
        print(f"  Total routes: {df['route_id'].nunique():,}")
        print(f"  Total drivers: {df['driver_id'].nunique():,}")
        print(f"  Avg stops/route: {len(df) / df['route_id'].nunique():.1f}")
        
        print(f"\n⏱️  Delay Statistics:")
        delay_rate = df['delay_flag'].mean()
        print(f"  Delay rate: {delay_rate:.2%}")
        print(f"  On-time rate: {1-delay_rate:.2%}")
        
        delayed_df = df[df['delay_flag'] == 1]
        print(f"\n  For DELAYED stops:")
        print(f"    Count: {len(delayed_df):,}")
        print(f"    Mean: {delayed_df['delay_minutes'].mean():.2f} minutes")
        print(f"    Median: {delayed_df['delay_minutes'].median():.2f} minutes")
        print(f"    Std: {delayed_df['delay_minutes'].std():.2f} minutes")
        print(f"    Min: {delayed_df['delay_minutes'].min():.2f} minutes")
        print(f"    Max: {delayed_df['delay_minutes'].max():.2f} minutes")
        
        print(f"\n  Percentiles (delayed stops):")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(delayed_df['delay_minutes'], p)
            print(f"    {p}th: {val:.2f} minutes")
        
        print(f"\n  Overall statistics:")
        print(f"    Mean: {df['delay_minutes'].mean():.2f} minutes")
        print(f"    Variance: {df['delay_minutes'].var():.4f}")
        print(f"    Range: [{df['delay_minutes'].min():.2f}, {df['delay_minutes'].max():.2f}]")
        
        print(f"\n🌍 Distribution:")
        print(f"  Countries: {df['country'].nunique()}")
        print(f"  Weekday deliveries: {df['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']).mean():.1%}")
        
        print(f"\n📈 Data Quality:")
        print(f"  Zero delays: {(df['delay_minutes'] == 0).mean():.2%}")
        print(f"  Early arrivals: {(df['delay_minutes'] < 0).mean():.2%}")
        print(f"  Target variance: {df['delay_minutes'].var():.4f} ✅ (should be >0.01)")
        
        if df['delay_minutes'].var() > 0.01:
            print(f"\n  ✅ Good variance - models can learn!")
        else:
            print(f"\n  ⚠️  Low variance - may need adjustment")


def main():
    """Generate synthetic delivery data"""
    
    print("\n" + "=" * 80)
    print("🔧 SYNTHETIC DELIVERY DATA GENERATOR")
    print("=" * 80)
    
    # Configuration
    num_routes = 20000  # Total routes
    avg_stops = 12      # Average stops per route
    # Expected total: ~240,000 stops 
    
    print(f"\nConfiguration:")
    print(f"  Routes: {num_routes:,}")
    print(f"  Avg stops/route: {avg_stops}")
    print(f"  Expected total stops: ~{num_routes * avg_stops:,}")
    
    print(f"\n🎯 Features of synthetic data:")
    print(f"  ✅ Realistic delay patterns (5-25 minutes)")
    print(f"  ✅ Driver skill variation")
    print(f"  ✅ Traffic patterns by time of day")
    print(f"  ✅ Weather effects")
    print(f"  ✅ Time window constraints")
    print(f"  ✅ Cumulative delay effects")
    print(f"  ✅ Weekend vs weekday differences")
    print(f"  ✅ Proper variance for learning")
    
    response = input(f"\n▶️  Generate synthetic dataset? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Generate data
    generator = SyntheticDeliveryDataGenerator(seed=42)
    df = generator.generate_and_save(
        num_routes=num_routes,
        avg_stops_per_route=avg_stops,
        output_path="data/synthetic_delivery_data.csv"
    )
    
    print("\n" + "=" * 80)
    print("✅ SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 80)
    
    print(f"\n📁 Files created:")
    print(f"  • data/synthetic_delivery_data.csv")
    print(f"  • data/synthetic_delivery_data_with_metadata.csv")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Review the generated data:")
    print(f"     python -c \"import pandas as pd; df=pd.read_csv('data/synthetic_delivery_data.csv'); print(df.head())\"")
    print(f"  ")
    print(f"  2. Check data quality:")
    print(f"     python quick_data_check.py")
    print(f"  ")
    print(f"  3. Train models on synthetic data:")
    print(f"     python train_improved.py")
    print(f"  ")
    print(f"  4. Compare with real data results")
    
    print(f"\n💡 Tip: The metadata file contains hidden variables (weather, driver skill)")
    print(f"    used to generate the data. You can use it to verify the models are")
    print(f"    learning the right patterns!")


if __name__ == "__main__":
    main()

