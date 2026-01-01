"""
Deep Learning Route Optimizer using Transformer Architecture
Learns optimal route sequences from driver behavior data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pickle


class RouteSequenceDataset(Dataset):
    """
    Dataset for route sequence learning.
    Each sample contains stops for one route with features and target sequence.
    """
    
    def __init__(self, df: pd.DataFrame, route_ids: Optional[List] = None, max_stops: int = 50):
        self.df = df.copy()
        self.max_stops = max_stops
        
        if route_ids is None:
            route_ids = self.df['route_id'].unique()
        
        self.routes = []
        self.feature_stats = None
        
        print(f"Processing {len(route_ids)} routes...")
        
        for route_id in route_ids:
            route_df = self.df[self.df['route_id'] == route_id].copy()
            
            # Skip routes that are too long
            if len(route_df) > max_stops:
                continue
            
            # Sort by planned sequence for consistent input
            route_df = route_df.sort_values('indexp')
            
            # Extract features
            features = self.extract_features(route_df)
            
            # Target: actual sequence (indexa)
            target_sequence = route_df['indexa'].values
            
            # Map to 0-indexed positions
            _, target_indices = np.unique(target_sequence, return_inverse=True)
            
            self.routes.append({
                'route_id': route_id,
                'features': features,
                'target_sequence': target_indices,
                'num_stops': len(route_df),
                'driver_id': route_df['driver_id'].iloc[0],
                'country': route_df['country'].iloc[0]
            })
        
        print(f"Successfully processed {len(self.routes)} routes")
    
    def extract_features(self, route_df: pd.DataFrame) -> np.ndarray:
        """Extract features for each stop in the route."""
        features_list = []
        
        for idx, row in route_df.iterrows():
            feat = [
                # Spatial features
                float(row.get('distancep', 0)),
                float(row.get('distancea', 0)),
                
                # Time features (convert to minutes)
                self.parse_time_to_minutes(row.get('earliest_time', '00:00:00')),
                self.parse_time_to_minutes(row.get('latest_time', '23:59:59')),
                self.parse_time_to_minutes(row.get('latest_time', '23:59:59')) - 
                    self.parse_time_to_minutes(row.get('earliest_time', '00:00:00')),  # time window width
                
                # Binary flags
                1.0 if row.get('depot', 0) == 1 else 0.0,
                1.0 if row.get('delivery', 0) == 1 else 0.0,
                
                # Categorical encodings
                float(self.encode_day(row.get('day_of_week', 'Monday'))),
                float(self.encode_country(row.get('country', 'Netherlands'))),
                
                # Planned sequence position (normalized)
                float(row.get('indexp', 0)) / max(len(route_df) - 1, 1),
                
                # Route-level features
                float(len(route_df)),  # total stops
                float(route_df['distancep'].mean()),  # avg distance
                
                # Delay features if available
                float(row.get('delay_flag', 0)),
                float(row.get('delay_minutes', 0)),
            ]
            
            features_list.append(feat)
        
        return np.array(features_list, dtype=np.float32)
    
    def parse_time_to_minutes(self, time_val) -> float:
        """Convert time string to minutes from midnight."""
        if pd.isna(time_val):
            return 480.0  # Default 8 AM
        
        if isinstance(time_val, str):
            try:
                parts = time_val.split(':')
                return float(parts[0]) * 60 + float(parts[1])
            except:
                return 480.0
        return float(time_val)
    
    def encode_day(self, day: str) -> int:
        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        return day_map.get(day, 0)
    
    def encode_country(self, country: str) -> int:
        country_map = {
            'Netherlands': 0, 'Spain': 1, 'Italy': 2, 
            'Germany': 3, 'UK': 4
        }
        return country_map.get(country, 0)
    
    def __len__(self):
        return len(self.routes)
    
    def __getitem__(self, idx):
        route = self.routes[idx]
        
        # Pad features to max_stops
        features = route['features']
        num_stops = route['num_stops']
        
        if num_stops < self.max_stops:
            padding = np.zeros((self.max_stops - num_stops, features.shape[1]), dtype=np.float32)
            features = np.vstack([features, padding])
        
        # Pad target sequence
        target = route['target_sequence']
        if len(target) < self.max_stops:
            target = np.pad(target, (0, self.max_stops - len(target)), constant_values=-1)
        
        return {
            'features': torch.FloatTensor(features),
            'target_sequence': torch.LongTensor(target),
            'num_stops': num_stops,
            'route_id': route['route_id']
        }


class RouteOptimizerTransformer(nn.Module):
    """
    Transformer-based model for learning optimal route sequences.
    Uses attention mechanism to predict stop visit order.
    """
    
    def __init__(
        self,
        feature_dim: int = 14,
        embedding_dim: int = 64,  # Reduced from 128 for faster training
        num_heads: int = 4,  # Reduced from 8 for faster training
        num_layers: int = 2,  # Reduced from 3 for faster training
        dropout: float = 0.1,
        max_stops: int = 50
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_stops = max_stops
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_stops, embedding_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder for sequence prediction
        self.sequence_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, max_stops)  # Predict position for each stop
        )
        
    def forward(self, stop_features, padding_mask=None):
        """
        Args:
            stop_features: [batch_size, num_stops, feature_dim]
            padding_mask: [batch_size, num_stops] - True for padded positions
        Returns:
            position_logits: [batch_size, num_stops, max_positions]
        """
        batch_size, num_stops, _ = stop_features.shape
        
        # Embed features
        embedded = self.feature_embedding(stop_features)  # [B, N, D]
        
        # Add positional encoding
        embedded = embedded + self.pos_encoding[:, :num_stops, :]
        
        # Encode all stops with attention
        if padding_mask is not None:
            encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        else:
            encoded = self.encoder(embedded)
        
        # Predict position for each stop
        position_logits = self.sequence_decoder(encoded)  # [B, N, max_stops]
        
        return position_logits


class DLRouteOptimizer:
    """
    Main class for training and using the DL route optimizer.
    """
    
    def __init__(
        self,
        feature_dim: int = 14,
        embedding_dim: int = 64,  # Reduced for faster training
        num_heads: int = 4,  # Reduced for faster training
        num_layers: int = 2,  # Reduced for faster training
        max_stops: int = 50,
        device: str = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = RouteOptimizerTransformer(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_stops=max_stops
        ).to(self.device)
        
        self.max_stops = max_stops
        self.feature_dim = feature_dim
        
        print(f"DL Route Optimizer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        batch_size: int = 16,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        save_dir: str = "outputs_v2/dl_models"
    ):
        """Train the DL route optimizer."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create datasets
        print("\n" + "=" * 80)
        print("CREATING DATASETS")
        print("=" * 80)
        
        train_routes = train_df['route_id'].unique()
        train_dataset = RouteSequenceDataset(train_df, train_routes, self.max_stops)
        
        if val_df is not None:
            val_routes = val_df['route_id'].unique()
            val_dataset = RouteSequenceDataset(val_df, val_routes, self.max_stops)
        else:
            val_dataset = None
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0
            )
        else:
            val_loader = None
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Training loop
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion
            )
            
            # Validate
            if val_loader:
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)
                scheduler.step(val_loss)
            else:
                val_loss, val_acc = train_loss, train_acc
            
            # Log progress
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
            
            # Save history (convert tensors to float)
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc)
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_dir / "best_model.pt")
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(save_dir / f"checkpoint_epoch_{epoch + 1}.pt")
        
        # Save final model and history
        self.save_model(save_dir / "final_model.pt")
        
        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved to: {save_dir.absolute()}")
        
        return training_history
    
    def _train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            targets = batch['target_sequence'].to(self.device)
            num_stops = batch['num_stops']
            
            # Create padding mask
            batch_size, max_len = features.shape[0], features.shape[1]
            padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
            for i, n in enumerate(num_stops):
                padding_mask[i, n:] = True
            
            # Forward pass
            position_logits = self.model(features, padding_mask)
            
            # Compute loss (only on non-padded positions)
            loss = 0
            for i in range(batch_size):
                n = num_stops[i]
                logits = position_logits[i, :n, :n]  # [n, n]
                target = targets[i, :n]  # [n]
                loss += criterion(logits, target)
                
                # Accuracy
                pred = logits.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total += n
            
            loss = loss / batch_size
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, dataloader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                targets = batch['target_sequence'].to(self.device)
                num_stops = batch['num_stops']
                
                # Create padding mask
                batch_size, max_len = features.shape[0], features.shape[1]
                padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
                for i, n in enumerate(num_stops):
                    padding_mask[i, n:] = True
                
                # Forward pass
                position_logits = self.model(features, padding_mask)
                
                # Compute loss
                loss = 0
                for i in range(batch_size):
                    n = num_stops[i]
                    logits = position_logits[i, :n, :n]
                    target = targets[i, :n]
                    loss += criterion(logits, target)
                    
                    # Accuracy
                    pred = logits.argmax(dim=-1)
                    correct += (pred == target).sum().item()
                    total += n
                
                loss = loss / batch_size
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def predict_route_sequence(self, route_df: pd.DataFrame) -> Dict:
        """
        Predict optimal visit sequence for a route.
        
        Args:
            route_df: DataFrame with stop information
            
        Returns:
            Dictionary with predicted sequence and metrics
        """
        self.model.eval()
        
        # Prepare data
        dataset = RouteSequenceDataset(route_df, route_df['route_id'].unique()[:1], self.max_stops)
        if len(dataset) == 0:
            return None
        
        batch = dataset[0]
        features = batch['features'].unsqueeze(0).to(self.device)
        num_stops = batch['num_stops']
        
        # Create padding mask
        padding_mask = torch.zeros(1, features.shape[1], dtype=torch.bool, device=self.device)
        padding_mask[0, num_stops:] = True
        
        with torch.no_grad():
            position_logits = self.model(features, padding_mask)
            
            # Get predictions for actual stops
            logits = position_logits[0, :num_stops, :num_stops]  # [n, n]
            predicted_positions = logits.argmax(dim=-1).cpu().numpy()
        
        # Create result
        route_df_sorted = route_df.sort_values('indexp').reset_index(drop=True)
        
        result = {
            'route_id': route_df['route_id'].iloc[0],
            'num_stops': num_stops,
            'predicted_sequence': predicted_positions.tolist(),
            'planned_sequence': route_df_sorted['indexp'].values.tolist(),
            'actual_sequence': route_df_sorted['indexa'].values.tolist(),
            'stop_ids': route_df_sorted['stop_id'].values.tolist(),
            'confidence_scores': torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy().tolist()
        }
        
        return result
    
    def save_model(self, filepath: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_dim': self.feature_dim,
            'max_stops': self.max_stops,
            'embedding_dim': self.model.embedding_dim
        }, filepath)
    
    def load_model(self, filepath: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {filepath}")

