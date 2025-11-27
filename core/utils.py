import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def save_json(data, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_metrics(metrics, precision=4):
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted[key] = round(value, precision)
        else:
            formatted[key] = value
    return formatted


def calculate_delay_statistics(df):
    stats = {
        "total_stops": len(df),
        "total_routes": df["route_id"].nunique() if "route_id" in df.columns else 0,
        "delay_rate": df["delayed_flag"].mean() if "delayed_flag" in df.columns else 0,
        "avg_delay_minutes": df["delay_minutes"].mean() if "delay_minutes" in df.columns else 0,
        "max_delay_minutes": df["delay_minutes"].max() if "delay_minutes" in df.columns else 0,
        "median_delay_minutes": df["delay_minutes"].median() if "delay_minutes" in df.columns else 0,
    }
    return stats


def export_predictions_to_csv(predictions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(predictions, pd.DataFrame):
        predictions.to_csv(output_path, index=False)
    else:
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
    
    print(f"Predictions exported to {output_path}")


def get_model_info(model):
    info = {
        "type": type(model).__name__,
        "trained": getattr(model, 'trained', False)
    }
    
    if hasattr(model, 'model'):
        info["base_model"] = type(model.model).__name__
    
    return info


def print_section_header(title, width=80):
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def print_subsection_header(title, width=80):
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)

