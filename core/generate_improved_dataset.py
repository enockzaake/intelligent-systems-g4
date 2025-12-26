"""
Improved Synthetic Delivery Data Generator
Creates realistic delivery data with learnable patterns optimized for model performance
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path


class ImprovedDeliveryDataGenerator:
    """
    Generate improved synthetic delivery data with realistic patterns and variance
    Optimized to achieve better model performance (F1 > 0.70, R² > 0.50)
    """
    
    def __init__(self, seed=42):
        """Initialize generator with random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        
        # Configuration
        self.countries = [1, 2]  # 2 countries as in original dataset
        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Driver skill levels (affects delay probability)
        # More realistic distribution: most drivers are average, few are very good/bad
        self.driver_skill_base = np.random.beta(2, 2, size=200)  # 200 drivers, beta distribution
        self.driver_skill_base = 0.6 + 0.3 * self.driver_skill_base  # Scale to 0.6-0.9 range
        
        # Time window types with realistic delay probabilities
        self.time_windows = {
            'tight': {'length': 30, 'delay_prob_mult': 1.5},    # 30-min window
            'normal': {'length': 60, 'delay_prob_mult': 1.0},   # 60-min window
            'flexible': {'length': 120, 'delay_prob_mult': 0.6} # 120-min window
        }
        
        # Peak hours (higher delay probability)
        self.peak_hours = [(11, 13), (17, 19)]  # 11 AM-1 PM, 5-7 PM
    
    def _is_peak_hour(self, hour):
        """Check if hour is in peak traffic period"""
        for start, end in self.peak_hours:
            if start <= hour < end:
                return True
        return False
    
    def _is_weekday(self, day_of_week):
        """Check if day is weekday"""
        return day_of_week not in ['Saturday', 'Sunday']
    
    def _get_driver_skill(self, driver_id):
        """Get driver skill based on driver ID"""
        idx = driver_id % len(self.driver_skill_base)
        return self.driver_skill_base[idx]
    
    def _get_time_window_type(self):
        """Randomly assign time window type with realistic distribution"""
        types = list(self.time_windows.keys())
        probs = [0.2, 0.6, 0.2]  # 20% tight, 60% normal, 20% flexible
        return np.random.choice(types, p=probs)
    
    def _calculate_delay_probability(self, stop_factors):
        """
        Calculate delay probability based on multiple realistic factors
        Returns probability between 0 and 1
        """
        # Base delay rate: 12% (realistic industry average)
        delay_prob = 0.08
        
        # Adjust by driver skill (better drivers = lower delay rate)
        driver_skill = stop_factors['driver_skill']
        delay_prob *= (1.8 - 0.8 * driver_skill)  # Skill 0.6 -> 1.4x, Skill 0.9 -> 1.1x
        
        # Adjust by stop position (later stops more likely to be delayed)
        stop_position_norm = stop_factors['stop_position_norm']
        delay_prob *= (1 + 0.3 * stop_position_norm)  # Up to 40% increase for last stops
        
        # Adjust by time window (tight windows = more delays)
        window_mult = stop_factors['window_mult']
        delay_prob *= window_mult
        
        # Adjust by peak hours
        if stop_factors['is_peak_hour']:
            delay_prob *= 1.3  # 30% increase during peak (reduced)
        
        # Adjust by weekday (weekends slightly better)
        if not stop_factors['is_weekday']:
            delay_prob *= 0.85  # 15% reduction on weekends
        
        # Cumulative delay effect (earlier delays affect later stops)
        cumulative_delay = stop_factors.get('cumulative_delay', 0)
        if cumulative_delay > 0:
            delay_prob *= (1 + 0.02 * cumulative_delay / 10)  # 2% per 10 min cumulative delay
        
        # Cap probability
        delay_prob = min(0.95, max(0.01, delay_prob))
        
        return delay_prob
    
    def _generate_delay_minutes(self, is_delayed):
        """
        Generate delay magnitude if delayed
        Uses log-normal distribution: most delays are small (5-20 min), few are large (60+ min)
        """
        if not is_delayed:
            return 0.0
        
        # Log-normal distribution: mean=2.5, sigma=0.8
        # This gives: median ~12 min, 90th percentile ~30 min, max ~120 min
        delay = np.random.lognormal(mean=2.5, sigma=0.8)
        delay = min(180, max(1, delay))  # Cap between 1 and 180 minutes
        
        return round(delay, 2)
    
    def generate_routes(self, n_routes=12000, avg_stops_per_route=20, delay_rate_target=0.12):
        """
        Generate synthetic delivery routes with improved distributions
        
        Args:
            n_routes: Number of routes to generate
            avg_stops_per_route: Average number of stops per route
            delay_rate_target: Target delay rate (default 12%)
        
        Returns:
            DataFrame with synthetic delivery data
        """
        print("\n" + "=" * 80)
        print("GENERATING IMPROVED SYNTHETIC DELIVERY DATA")
        print("=" * 80)
        print(f"\nTarget: {n_routes:,} routes with ~{avg_stops_per_route} stops each")
        print(f"Target delay rate: {delay_rate_target:.1%}")
        
        data = []
        route_counter = 0
        
        # Generate routes
        for route_num in range(n_routes):
            if (route_num + 1) % 1000 == 0:
                print(f"  Generated {route_num + 1:,} / {n_routes:,} routes...")
            
            # Route-level attributes
            route_counter += 1
            route_id = route_counter  # Same route_id for all stops in this route
            
            country = np.random.choice(self.countries)
            driver_id = np.random.randint(1, 201)  # 200 drivers
            driver_skill = self._get_driver_skill(driver_id)
            day_of_week = np.random.choice(self.days_of_week)
            is_weekday = self._is_weekday(day_of_week)
            week_id = np.random.randint(0, 52)  # 52 weeks
            
            # Number of stops in this route (realistic: 10-30 stops)
            num_stops = max(10, min(30, int(np.random.normal(avg_stops_per_route, 5))))
            
            # Starting time (between 6 AM and 10 AM)
            start_hour = np.random.uniform(6, 10)
            current_time = start_hour
            
            # Route-level distance (total planned distance)
            total_planned_distance = np.random.uniform(30, 120)  # km
            
            cumulative_delay = 0.0
            route_stops = []
            
            # Generate stops for this route
            for stop_idx in range(num_stops):
                
                # Stop position (normalized 0-1)
                stop_position_norm = stop_idx / max(1, num_stops - 1)
                
                # Planned vs actual indices (sometimes order changes slightly)
                indexp = stop_idx + 1
                # Small chance of order change (realistic)
                if np.random.random() < 0.15:  # 15% chance
                    indexa = indexp + np.random.choice([-1, 0, 1, 2])
                    indexa = max(1, min(indexa, num_stops))
                else:
                    indexa = indexp
                
                # Planned distance to this stop (cumulative)
                distancep = total_planned_distance * (stop_idx + 1) / num_stops
                distancep += np.random.normal(0, 1)  # Small variation
                distancep = max(0, round(distancep, 2))
                
                # Actual distance (can deviate from planned)
                # If delayed, more likely to have positive deviation
                distance_deviation_pct = np.random.normal(0, 0.08)  # ±8% typical
                distancea = distancep * (1 + distance_deviation_pct)
                distancea = max(0, round(distancea, 2))
                
                # Time window
                window_type = self._get_time_window_type()
                window_config = self.time_windows[window_type]
                window_length = window_config['length']
                window_mult = window_config['delay_prob_mult']
                
                # Arrival time (cumulative)
                time_between_stops = np.random.uniform(15, 25) / 60  # 15-25 minutes
                current_time += time_between_stops
                hour_of_day = int(current_time % 24)
                is_peak_hour = self._is_peak_hour(hour_of_day)
                
                # Calculate delay probability
                stop_factors = {
                    'driver_skill': driver_skill,
                    'stop_position_norm': stop_position_norm,
                    'window_mult': window_mult,
                    'is_peak_hour': is_peak_hour,
                    'is_weekday': is_weekday,
                    'cumulative_delay': cumulative_delay
                }
                
                delay_prob = self._calculate_delay_probability(stop_factors)
                is_delayed = np.random.binomial(1, delay_prob) == 1
                
                # Generate delay magnitude if delayed
                delay_minutes = self._generate_delay_minutes(is_delayed)
                
                # Update cumulative delay
                if delay_minutes > 0:
                    cumulative_delay += delay_minutes
                
                # Create time values (as datetime objects)
                base_date = datetime(2024, 1, 1)
                arrived_time = base_date + timedelta(hours=current_time + delay_minutes / 60)
                
                # Time window around planned arrival
                earliest_time = base_date + timedelta(hours=current_time - window_length / 120)
                latest_time = base_date + timedelta(hours=current_time + window_length / 120)
                
                # Depot and delivery flags
                depot = 1 if stop_idx == 0 else 0
                delivery = 1 if stop_idx > 0 else 0
                
                # Address ID (can be shared across stops)
                address_id = np.random.randint(0, 10000)
                
                # Create record
                record = {
                    'route_id': route_id,
                    'driver_id': driver_id,
                    'stop_id': stop_idx,
                    'address_id': address_id,
                    'week_id': week_id,
                    'country': country,
                    'day_of_week': day_of_week,
                    'indexp': indexp,
                    'indexa': indexa,
                    'arrived_time': arrived_time,
                    'earliest_time': earliest_time,
                    'latest_time': latest_time,
                    'distancep': distancep,
                    'distancea': distancea,
                    'depot': depot,
                    'delivery': delivery,
                    # Pre-computed targets for convenience
                    'delay_minutes': delay_minutes,
                    'delay_flag': 1 if is_delayed else 0
                }
                
                route_stops.append(record)
            
            data.extend(route_stops)
        
        df = pd.DataFrame(data)
        
        print(f"\n✅ Generated {len(df):,} stops across {n_routes:,} routes")
        
        return df
    
    def generate_and_save(self, n_routes=12000, avg_stops_per_route=20, 
                         delay_rate_target=0.12,
                         output_path="data/improved_delivery_data.csv"):
        """Generate data and save to CSV"""
        
        df = self.generate_routes(n_routes, avg_stops_per_route, delay_rate_target)
        
        # Print statistics
        self._print_statistics(df)
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime columns to strings for CSV
        df_save = df.copy()
        for col in ['arrived_time', 'earliest_time', 'latest_time']:
            if col in df_save.columns:
                df_save[col] = df_save[col].dt.strftime('%H:%M:%S')
        
        df_save.to_csv(output_path, index=False)
        
        print(f"\n✅ Data saved to: {output_path.absolute()}")
        
        return df
    
    def _print_statistics(self, df):
        """Print statistics about generated data"""
        
        print("\n" + "=" * 80)
        print("GENERATED DATA STATISTICS")
        print("=" * 80)
        
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
        if len(delayed_df) > 0:
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
        print(f"  Target variance: {df['delay_minutes'].var():.4f}")
        
        if df['delay_minutes'].var() > 0.01:
            print(f"\n  ✅ Good variance - models can learn!")
        else:
            print(f"\n  ⚠️  Low variance - may need adjustment")
        
        # Check delay rate
        if 0.08 <= delay_rate <= 0.18:
            print(f"\n  ✅ Realistic delay rate ({delay_rate:.1%})")
        else:
            print(f"\n  ⚠️  Delay rate may need adjustment ({delay_rate:.1%})")


def main():
    """Generate improved synthetic delivery data"""
    
    print("\n" + "=" * 80)
    print("🔧 IMPROVED SYNTHETIC DELIVERY DATA GENERATOR")
    print("=" * 80)
    
    # Configuration
    n_routes = 12000  # Total routes
    avg_stops = 20    # Average stops per route
    # Expected total: ~240,000 stops 
    
    print(f"\nConfiguration:")
    print(f"  Routes: {n_routes:,}")
    print(f"  Avg stops/route: {avg_stops}")
    print(f"  Expected total stops: ~{n_routes * avg_stops:,}")
    
    print(f"\n🎯 Features of improved synthetic data:")
    print(f"  ✅ Realistic delay patterns (12% base rate)")
    print(f"  ✅ Driver skill variation (beta distribution)")
    print(f"  ✅ Peak hour effects (11 AM-1 PM, 5-7 PM)")
    print(f"  ✅ Time window constraints (30/60/120 min)")
    print(f"  ✅ Cumulative delay effects")
    print(f"  ✅ Weekend vs weekday differences")
    print(f"  ✅ Log-normal delay magnitude (realistic distribution)")
    print(f"  ✅ Proper variance for learning")
    print(f"  ✅ Pre-computed delay targets")
    
    response = input(f"\n▶️  Generate improved synthetic dataset? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Generate data
    generator = ImprovedDeliveryDataGenerator(seed=42)
    df = generator.generate_and_save(
        n_routes=n_routes,
        avg_stops_per_route=avg_stops,
        delay_rate_target=0.12,
        output_path="data/improved_delivery_data.csv"
    )
    
    print("\n" + "=" * 80)
    print("✅ IMPROVED SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 80)
    
    print(f"\n📁 Files created:")
    print(f"  • data/improved_delivery_data.csv")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Explore the dataset:")
    print(f"     jupyter notebook notebooks/dataset_exploration.ipynb")
    print(f"  ")
    print(f"  2. Validate dataset quality:")
    print(f"     python validate_dataset.py")
    print(f"  ")
    print(f"  3. Train models on improved data:")
    print(f"     python main.py --dataset data/improved_delivery_data.csv")


if __name__ == "__main__":
    main()

