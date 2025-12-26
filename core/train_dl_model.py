"""
Training script for DL Route Optimizer
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dl_route_optimizer import DLRouteOptimizer, RouteSequenceDataset
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Train DL Route Optimizer')
    parser.add_argument('--data', type=str, default='data/synthetic_delivery_data.csv',
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='outputs_v2/dl_models',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate (higher for faster convergence)')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension (reduced for faster training)')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads (reduced for faster training)')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of transformer layers (reduced for faster training)')
    parser.add_argument('--max-stops', type=int, default=50,
                       help='Maximum stops per route')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DL ROUTE OPTIMIZER TRAINING")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Num heads: {args.num_heads}")
    print(f"Num layers: {args.num_layers}")
    print(f"Max stops: {args.max_stops}")
    print()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} stops from {df['route_id'].nunique():,} routes")
    
    # Split data by routes (80/20 split)
    route_ids = df['route_id'].unique()
    np.random.seed(42)
    np.random.shuffle(route_ids)
    
    split_idx = int(0.8 * len(route_ids))
    train_route_ids = route_ids[:split_idx]
    val_route_ids = route_ids[split_idx:]
    
    train_df = df[df['route_id'].isin(train_route_ids)]
    val_df = df[df['route_id'].isin(val_route_ids)]
    
    print(f"Training routes: {len(train_route_ids):,} ({len(train_df):,} stops)")
    print(f"Validation routes: {len(val_route_ids):,} ({len(val_df):,} stops)")
    print()
    
    # Initialize optimizer
    optimizer = DLRouteOptimizer(
        feature_dim=14,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_stops=args.max_stops
    )
    
    # Train
    history = optimizer.train_model(
        train_df=train_df,
        val_df=val_df,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.output
    )
    
    # Save training summary (ensure all values are JSON serializable)
    output_dir = Path(args.output)
    summary = {
        'data_file': args.data,
        'num_train_routes': int(len(train_route_ids)),
        'num_val_routes': int(len(val_route_ids)),
        'hyperparameters': {
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'learning_rate': float(args.lr),
            'embedding_dim': int(args.embedding_dim),
            'num_heads': int(args.num_heads),
            'num_layers': int(args.num_layers),
            'max_stops': int(args.max_stops)
        },
        'best_val_loss': float(min(h['val_loss'] for h in history)),
        'best_val_acc': float(max(h['val_acc'] for h in history)),
        'final_train_loss': float(history[-1]['train_loss']),
        'final_train_acc': float(history[-1]['train_acc'])
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best validation loss: {summary['best_val_loss']:.4f}")
    print(f"Best validation accuracy: {summary['best_val_acc']:.2%}")
    print(f"Final training loss: {summary['final_train_loss']:.4f}")
    print(f"Final training accuracy: {summary['final_train_acc']:.2%}")
    print(f"\nAll outputs saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

