"""
Quick fix: Reassign route IDs to create proper multi-stop routes
"""
import pandas as pd
import numpy as np

print("\n" + "=" * 80)
print("FIXING SYNTHETIC DATA STRUCTURE")
print("=" * 80)

# Load data
df = pd.read_csv("data/synthetic_delivery_data.csv")

print(f"\nBefore fix:")
print(f"  Total rows: {len(df):,}")
print(f"  Unique routes: {df['route_id'].nunique():,}")
print(f"  Avg stops/route: {len(df)/df['route_id'].nunique():.2f}")

# Group rows into routes (12 stops per route on average)
stops_per_route = 12
num_routes = len(df) // stops_per_route

print(f"\nCreating {num_routes:,} routes with ~{stops_per_route} stops each...")

# Assign new route IDs
new_route_ids = []
current_route = 1

for i in range(len(df)):
    new_route_ids.append(current_route)
    if (i + 1) % stops_per_route == 0:
        current_route += 1

df['route_id'] = new_route_ids

# Update stop IDs to be unique
df['stop_id'] = [f"S{i}" for i in range(len(df))]

# Save fixed data
df.to_csv("data/synthetic_delivery_data.csv", index=False)

print(f"\nAfter fix:")
print(f"  Total rows: {len(df):,}")
print(f"  Unique routes: {df['route_id'].nunique():,}")
print(f"  Avg stops/route: {len(df)/df['route_id'].nunique():.2f}")

print(f"\n✅ Fixed data saved to: data/synthetic_delivery_data.csv")
print(f"\n🚀 Now run: python main.py")








