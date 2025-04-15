import argparse
import os
import logging
import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from .data import MolecularDatasetFactory
from .models import MolecularGNNFactory
from .training import ModelTrainer
from .utils import OptunaHPTuner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Molecular GNN for property prediction')
    
    # Dataset arguments
    parser.add_argument('--data_file', type=str, required=True,
                      help='Path to the data file (CSV or Excel)')
    parser.add_argument('--smiles_col', type=str, default='smiles',
                      help='Column name containing SMILES strings')
    parser.add_argument('--target_col', type=str, default='activity',
                      help='Column name containing target values')
    parser.add_argument('--task_type', type=str, choices=['classification', 'regression'],
                      default='classification', help='Task type')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Test set size (fraction of total)')
    parser.add_argument('--val_size', type=float, default=0.1,
                      help='Validation set size (fraction of training)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Maximum number of epochs to train')
    parser.add_argument('--patience', type=int, default=15,
                      help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                      help='Weight decay for optimizer')
    
    # Model arguments
    parser.add_argument('--hidden_channels', type=int, default=64,
                      help='Hidden layer dimensions')
    parser.add_argument('--num_layers', type=int, default=3,
                      help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout rate')
    parser.add_argument('--layer_type', type=str, choices=['GCN', 'GAT', 'GraphConv'],
                      default='GCN', help='GNN layer type')
    parser.add_argument('--pooling', type=str, choices=['mean', 'sum', 'max'],
                      default='mean', help='Graph pooling strategy')
    
    # Hyperparameter tuning arguments
    parser.add_argument('--tune', action='store_true',
                      help='Perform hyperparameter tuning')
    parser.add_argument('--n_trials', type=int, default=50,
                      help='Number of trials for hyperparameter tuning')
    parser.add_argument('--timeout', type=int, default=None,
                      help='Timeout for hyperparameter tuning in seconds')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Output directory for saved models and results')
    parser.add_argument('--model_name', type=str, default='molecular_gnn_model',
                      help='Base name for saved model files')
    
    return parser.parse_args()


def main():
    """Main function to run the molecular GNN from the command line."""
    args = parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    logger.info(f"Loading data from {args.data_file}")
    
    dataset = MolecularDatasetFactory.from_csv(
        args.data_file,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        root=os.path.join(args.output_dir, 'data')
    )
    
    # Dataset statistics
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Create train/val/test split
    indices = list(range(len(dataset)))
    train_val_indices, test_indices = train_test_split(
        indices, test_size=args.test_size, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=args.val_size, random_state=42
    )
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]
    
    logger.info(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}, Test set: {len(test_dataset)}")
    
    # Determine input dimension
    input_dim = dataset[0].x.shape[1]
    logger.info(f"Input feature dimension: {input_dim}")
    
    if args.tune:
        # Hyperparameter tuning
        logger.info("Starting hyperparameter tuning")
        
        tuner = OptunaHPTuner(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            task_type=args.task_type
        )
        
        study = tuner.tune(n_trials=args.n_trials, timeout=args.timeout)
        
        # Save study results
        tuner.save_study(os.path.join(args.output_dir, 'hptuning'))
        
        # Train best model
        model, trainer = tuner.train_best_model(test_dataset)
        
        # Save best model
        trainer.save_model(os.path.join(args.output_dir, f"{args.model_name}_best.pt"))
        
        # Plot training history
        trainer.plot_training_history(
            save_path=os.path.join(args.output_dir, 'training_history.png')
        )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(DataLoader(test_dataset, batch_size=args.batch_size))
        logger.info(f"Test set metrics: {test_metrics}")
        
        if args.task_type == 'classification':
            # Plot ROC curve and confusion matrix
            trainer.plot_roc_curve(
                DataLoader(test_dataset, batch_size=args.batch_size),
                save_path=os.path.join(args.output_dir, 'roc_curve.png')
            )
            trainer.plot_confusion_matrix(
                DataLoader(test_dataset, batch_size=args.batch_size),
                save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
            )
    else:
        # Train with fixed hyperparameters
        logger.info("Training with fixed hyperparameters")
        