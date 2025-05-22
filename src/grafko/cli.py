import argparse
import os
import logging
import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from .data import MolecularDatasetFactory
from .models import MolecularGNNFactory
from .training import ModelTrainer, get_weighted_sampler
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
    parser.add_argument('--num_classes', type=int, default=2, # Add if needed for classification
                        help='Number of classes for classification task')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Test set size (fraction of total)')
    parser.add_argument('--val_size', type=float, default=0.1,
                      help='Validation set size (fraction of training)')
    parser.add_argument('--imbalance_threshold', type=float, default=2.0, # Add threshold for sampler
                        help='Ratio to consider dataset imbalanced for weighted sampling')

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
        indices, test_size=args.test_size, random_state=42, stratify=dataset.y if args.task_type == 'classification' else None
    )
    # Stratify validation split based on the train_val portion
    train_val_labels = dataset.y[train_val_indices] if args.task_type == 'classification' else None
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=args.val_size / (1.0 - args.test_size), random_state=42, stratify=train_val_labels
    )

    train_dataset = dataset[torch.tensor(train_indices)] # Use tensors for PyG subsetting
    val_dataset = dataset[torch.tensor(val_indices)]
    test_dataset = dataset[torch.tensor(test_indices)]

    logger.info(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}, Test set: {len(test_dataset)}")

    # Determine input dimension and output dimension
    input_dim = dataset.num_node_features
    # Determine output channels based on task type and data
    if args.task_type == 'regression':
        out_channels = 1
    else: # classification
        # Ensure labels are integers for unique count
        unique_labels = torch.unique(dataset.y.long())
        out_channels = len(unique_labels)
        # If num_classes arg is provided and different, log a warning or error
        if args.num_classes != out_channels:
             logger.warning(f"Provided --num_classes ({args.num_classes}) differs from unique labels found ({out_channels}). Using {out_channels}.")
        args.num_classes = out_channels # Use the actual number of classes found

    logger.info(f"Input feature dimension: {input_dim}")
    logger.info(f"Output dimension: {out_channels}")


    # --- Create DataLoaders ---
    # Validation Loader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # Test Loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) if test_dataset else None

    # --- Handle Training Data Loader with Sampler ---
    train_sampler = None
    if args.task_type == 'classification':
        train_sampler = get_weighted_sampler(train_dataset, args.num_classes, args.imbalance_threshold)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None) # Shuffle only if sampler is not used
    )
    # --------------------------

    if args.tune:
        # Hyperparameter tuning
        logger.info("Starting hyperparameter tuning")

        tuner = OptunaHPTuner(
            train_dataset=train_dataset, # Pass dataset for tuning
            val_dataset=val_dataset,     # Pass dataset for tuning
            device=device,
            task_type=args.task_type,
            input_dim=input_dim,
            out_channels=out_channels,
            num_classes=args.num_classes # Pass num_classes if needed by tuner
        )

        study = tuner.tune(n_trials=args.n_trials, timeout=args.timeout)

        # Save study results
        tuner.save_study(os.path.join(args.output_dir, 'hptuning'))

        # Train best model - tuner should handle creating the final loader/trainer
        model, trainer = tuner.train_best_model(test_dataset) # Pass test dataset if needed

        # Save best model
        trainer.save_model(os.path.join(args.output_dir, f"{args.model_name}_best.pt"))

        # Plot training history
        trainer.plot_training_history(
            save_path=os.path.join(args.output_dir, 'training_history_best.png')
        )

        # Evaluate on test set (using the loader created outside tuner if needed)
        if test_loader:
            test_metrics = trainer.evaluate(test_loader)
            logger.info(f"Best model - Test set metrics: {test_metrics}")

            if args.task_type == 'classification':
                # Plot ROC curve and confusion matrix
                trainer.plot_roc_curve(
                    test_loader,
                    save_path=os.path.join(args.output_dir, 'roc_curve_best.png')
                )
                trainer.plot_confusion_matrix(
                    test_loader,
                    save_path=os.path.join(args.output_dir, 'confusion_matrix_best.png')
                )
    else:
        # Train with fixed hyperparameters
        logger.info("Training with fixed hyperparameters")

        # Create model
        model = MolecularGNNFactory.create_model(
            model_type='basic_gnn', # Or use args.layer_type appropriately if factory supports it
            config={
                'in_channels': input_dim,
                'hidden_channels': args.hidden_channels,
                'out_channels': out_channels, # Use calculated out_channels
                'num_layers': args.num_layers,
                'dropout_rate': args.dropout,
                'layer_type': args.layer_type,
                'activation': 'relu',
                'pooling': args.pooling,
                'task_type': args.task_type
            }
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Create trainer
        trainer = ModelTrainer(model, optimizer, device, task_type=args.task_type)

        # Train model
        epochs = args.epochs
        patience = args.patience

        # Set up scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )

        # --- Decide whether to use full training data or train/val split ---
        # Option 1: Train on train_dataset, validate on val_dataset
        logger.info("Training on train set, validating on validation set.")
        history = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience, scheduler=scheduler)
        final_eval_loader = val_loader
        final_eval_set_name = "Validation"

        # Option 2: Combine train+val for final training run (less common if test set exists)
        # logger.info("Training on combined train+validation set, validating on test set.")
        # full_train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset]) # Standard ConcatDataset
        # # Need to handle sampler for concatenated dataset if needed - more complex
        # full_train_sampler = None # Or recalculate sampler for full_train_dataset
        # full_train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, sampler=full_train_sampler, shuffle=(full_train_sampler is None))
        # history = trainer.train(full_train_loader, test_loader, epochs=epochs, patience=patience, scheduler=scheduler)
        # final_eval_loader = test_loader
        # final_eval_set_name = "Test"
        # --------------------------------------------------------------------

        # Save model
        trainer.save_model(os.path.join(args.output_dir, f"{args.model_name}.pt"))

        # Plot training history
        trainer.plot_training_history(
            save_path=os.path.join(args.output_dir, 'training_history.png')
        )

        # Final evaluation on the appropriate set
        if final_eval_loader:
            final_metrics = trainer.evaluate(final_eval_loader)
            logger.info(f"Final {final_eval_set_name} metrics: {final_metrics}")

            if args.task_type == 'classification':
                 # Plot ROC curve and confusion matrix on the final evaluation set
                trainer.plot_roc_curve(
                    final_eval_loader,
                    save_path=os.path.join(args.output_dir, f'roc_curve_{final_eval_set_name.lower()}.png')
                )
                trainer.plot_confusion_matrix(
                    final_eval_loader,
                    save_path=os.path.join(args.output_dir, f'confusion_matrix_{final_eval_set_name.lower()}.png')
                )
        else:
             logger.warning("No final evaluation set specified (test_loader was None).")


if __name__ == '__main__':
    main()