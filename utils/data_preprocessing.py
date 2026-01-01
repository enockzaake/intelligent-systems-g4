import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self):
        df = pd.read_csv(self.data_path)
        
        if 'actual_arrival_delay' in df.columns:
            df['actual_arrival_delay'] = df['actual_arrival_delay'] * 24 * 60
        
        time_cols = ["arrived_time", "earliest_time", "latest_time"]
        for col in time_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif df[col].dtype in ['float64', 'int64']:
                    df[col] = pd.to_datetime(df[col], unit='s', errors="coerce")
        
        return df
    
    def create_features(self, df):
        df = df.copy()
        
        # Helper function to safely parse time strings
        def safe_parse_time(series):
            """Parse time strings, handling various formats"""
            # If already datetime, return as is
            if pd.api.types.is_datetime64_any_dtype(series):
                return series
            
            # Try parsing with explicit format for HH:MM or H:MM
            try:
                return pd.to_datetime(series, format='%H:%M', errors='coerce')
            except:
                # Fallback to general parsing
                return pd.to_datetime(series, errors='coerce')
        
        # Check for pre-computed delay values first (from synthetic data generator)
        if 'delay_minutes' in df.columns and 'delay_flag' in df.columns:
            # Use pre-computed values from synthetic data generator
            df["delay_minutes"] = df["delay_minutes"]
            df["delayed_flag"] = df["delay_flag"]
        elif 'actual_arrival_delay' in df.columns and 'delay_flag' in df.columns:
            # Use values from cleaned original data
            df["delay_minutes"] = np.maximum(0, df["actual_arrival_delay"])
            df["delayed_flag"] = df["delay_flag"]
        else:
            # Recalculate from time columns (fallback)
            arrived_time = safe_parse_time(df["arrived_time"])
            latest_time = safe_parse_time(df["latest_time"])
            
            df["delay_minutes"] = np.maximum(
                0, 
                (arrived_time - latest_time).dt.total_seconds() / 60
            )
            df["delayed_flag"] = (df["delay_minutes"] > 0).astype(int)
        
        # Parse and extract hour
        df["hour_of_arrival"] = safe_parse_time(df["arrived_time"]).dt.hour
        
        # Calculate time window length
        df["time_window_length"] = (
            safe_parse_time(df["latest_time"]) - safe_parse_time(df["earliest_time"])
        ).dt.total_seconds() / 60
        
        df["delay_ratio"] = df["delay_minutes"] / (df["time_window_length"] + 1e-6)
        
        weekday_map = {
            "Monday": 1, "Tuesday": 1, "Wednesday": 1, "Thursday": 1, "Friday": 1,
            "Saturday": 0, "Sunday": 0
        }
        df["weekday_flag"] = df["day_of_week"].map(weekday_map).fillna(0).astype(int)
        
        df["stop_deviation"] = df["indexa"] - df["indexp"]
        df["distance_deviation"] = (df["distancea"] - df["distancep"]) / (df["distancep"] + 1e-6)
        
        df = df.sort_values(["route_id", "indexp"]).reset_index(drop=True)
        
        route_stops = df.groupby("route_id")["stop_id"].transform("count")
        df["stop_position_norm"] = df.groupby("route_id").cumcount() / (route_stops + 1e-6)
        
        df["prev_stop_delay"] = df.groupby("route_id")["delay_minutes"].shift(1).fillna(0)
        df["cumulative_delay"] = df.groupby("route_id")["delay_minutes"].cumsum()
        
        df["route_total_stops"] = df.groupby("route_id")["stop_id"].transform("count")
        df["route_avg_distance"] = df.groupby("route_id")["distancep"].transform("mean")
        df["route_total_distance"] = df.groupby("route_id")["distancep"].transform("sum")
        
        return df
    
    def encode_categorical(self, df, fit=True):
        df = df.copy()
        categorical_cols = ["day_of_week", "country", "driver_id"]
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f"{col}_encoded"] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def prepare_features(self, df, target_col, fit_scaler=True):
        feature_cols = [
            "hour_of_arrival", "time_window_length", "weekday_flag",
            "stop_deviation", "distance_deviation", "stop_position_norm",
            "prev_stop_delay", "cumulative_delay", "route_total_stops",
            "route_avg_distance", "route_total_distance",
            "indexp", "indexa", "distancep", "distancea",
            "depot", "delivery",
            "day_of_week_encoded", "country_encoded", "driver_id_encoded"
        ]
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y, feature_cols
    
    def split_by_routes(self, df, test_size=0.2, random_state=42):
        unique_routes = df["route_id"].unique()
        train_routes, test_routes = train_test_split(
            unique_routes, test_size=test_size, random_state=random_state
        )
        
        train_df = df[df["route_id"].isin(train_routes)].copy()
        test_df = df[df["route_id"].isin(test_routes)].copy()
        
        return train_df, test_df
    
    def prepare_lstm_sequences(self, df, sequence_length=10):
        """Prepare sequences for LSTM with proper error handling"""
        df = df.sort_values(["route_id", "indexp"]).reset_index(drop=True)
        
        sequences = []
        targets_clf = []
        targets_reg = []
        
        # Get unique routes
        unique_routes = df["route_id"].unique()
        
        # Process each route
        for route_id in unique_routes:
            route_data = df[df["route_id"] == route_id].copy()
            
            # Skip routes with too few stops
            if len(route_data) < 2:
                continue
            
            # Get features - ensure they exist
            feature_cols = [col for col in self.feature_columns if col in route_data.columns]
            if not feature_cols:
                continue
                
            route_features = route_data[feature_cols].values
            
            # Get targets
            if "delayed_flag" not in route_data.columns or "delay_minutes" not in route_data.columns:
                continue
                
            route_targets_clf = route_data["delayed_flag"].values
            route_targets_reg = route_data["delay_minutes"].values
            
            # Create sequences for each stop in the route
            for i in range(len(route_data)):
                start_idx = max(0, i - sequence_length + 1)
                seq = route_features[start_idx:i+1]
                
                # Pad if necessary
                if len(seq) < sequence_length:
                    padding = np.zeros((sequence_length - len(seq), seq.shape[1]))
                    seq = np.vstack([padding, seq])
                
                sequences.append(seq)
                targets_clf.append(route_targets_clf[i])
                targets_reg.append(route_targets_reg[i])
        
        # Convert to arrays
        if len(sequences) == 0:
            print(f"⚠️  Warning: No sequences generated! Check data structure.")
            print(f"   Routes: {len(unique_routes)}, Features: {len(self.feature_columns) if self.feature_columns else 0}")
            return np.array([]), np.array([]), np.array([])
        
        return np.array(sequences), np.array(targets_clf), np.array(targets_reg)
    
    def save_preprocessor(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        with open(save_dir / "label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)
        
        with open(save_dir / "feature_columns.pkl", "wb") as f:
            pickle.dump(self.feature_columns, f)
    
    def load_preprocessor(self, save_dir):
        save_dir = Path(save_dir)
        
        with open(save_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        
        with open(save_dir / "label_encoders.pkl", "rb") as f:
            self.label_encoders = pickle.load(f)
        
        with open(save_dir / "feature_columns.pkl", "rb") as f:
            self.feature_columns = pickle.load(f)
    
    def process_full_pipeline(self):
        df = self.load_data()
        df = self.create_features(df)
        df = self.encode_categorical(df, fit=True)
        
        train_df, test_df = self.split_by_routes(df)
        
        X_train_clf, y_train_clf, feature_cols = self.prepare_features(
            train_df, "delayed_flag", fit_scaler=True
        )
        X_test_clf, y_test_clf, _ = self.prepare_features(
            test_df, "delayed_flag", fit_scaler=False
        )
        
        X_train_reg, y_train_reg, _ = self.prepare_features(
            train_df, "delay_minutes", fit_scaler=False
        )
        X_test_reg, y_test_reg, _ = self.prepare_features(
            test_df, "delay_minutes", fit_scaler=False
        )
        
        return {
            "classification": {
                "X_train": X_train_clf,
                "X_test": X_test_clf,
                "y_train": y_train_clf,
                "y_test": y_test_clf
            },
            "regression": {
                "X_train": X_train_reg,
                "X_test": X_test_reg,
                "y_train": y_train_reg,
                "y_test": y_test_reg
            },
            "train_df": train_df,
            "test_df": test_df,
            "feature_columns": feature_cols
        }

