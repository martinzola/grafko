import optuna
import torch
from torch_geometric.data import DataLoader
import numpy as np
import logging
from typing import Dict, Any, Callable, List, Optional, Union
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

from ..models.gnn import MolecularGNNFactory
from ..training.trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptunaHPTuner:
    """Hyperparameter tuning using Optuna."""
    
    def __init__(self, train_dataset, val_dataset, device=None, task_type: str = 'classification',
                 study_name: Optional[str] = None, storage: Optional[str] = None):
        """
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            device: Device to use for training
            task_type: 'classification' or 'regression'
            study_name: Name of the Optuna study
            storage: Storage URL for Optuna study
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.task_type = task_type
        self.study_name = study_name or f"molecular_gnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        
        # Determine input feature dimensions
        self.input_dim = train_dataset[0].x.shape[1]
        
        # Set best params
        self.best_params = None
        self.best_model = None
        self.best_study = None
    
    def create_model(self, trial: optuna.Trial) -> torch.nn.Module:
        """Create a model with trial-suggested hyperparameters."""
        # Define hyperparameters to tune
        hidden_channels = trial.suggest_int('hidden_channels', 32, 256)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        layer_type = trial.suggest_categorical('layer_type', ['GCN', 'GAT', 'GraphConv'])
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
        pooling = trial.suggest_categorical('pooling', ['mean', 'max', 'sum'])
        
        # Create model config
        config = {
            'in_channels': self.input_dim,
            'hidden_channels': hidden_channels,
            'out_channels': 1,  # Binary classification or regression
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'layer_type': layer_type,
            'activation': activation,
            'pooling': pooling,
            'task_type': self.task_type
        }
        
        # Create model using factory
        return MolecularGNNFactory.create_model('basic_gnn', config)
    
    def create_optimizer(self, trial: optuna.Trial, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Create an optimizer with trial-suggested hyperparameters."""
        # Define optimizer hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Create optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop'])
        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        return optimizer
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Create model and optimizer
        model = self.create_model(trial)
        optimizer = self.create_optimizer(trial, model)
        
        # Create data loaders
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create trainer
        trainer = ModelTrainer(model, optimizer, self.device, self.task_type)
        
        # Training parameters
        epochs = trial.suggest_int('epochs', 20, 100)
        patience = trial.suggest_int('patience', 5, 20)
        
        # Train model
        history = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience)
        
        # Get final validation metric
        if self.task_type == 'classification':
            metric_value = history['metrics'].get('roc_auc', [-1])[-1]
            return metric_value  # Maximize AUC
        else:
            metric_value = history['metrics'].get('rmse', [float('inf')])[-1]
            return -metric_value  # Minimize RMSE
    
    def tune(self, n_trials: int = 50, timeout: Optional[int] = None) -> optuna.Study:
        """Run hyperparameter tuning.
        
        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (if None, no timeout)
            
        Returns:
            Optuna study object
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        self.best_params = study.best_params
        self.best_study = study
        
        return study
    
    def train_best_model(self, test_dataset=None):
        """Train a model with the best hyperparameters.
        
        Args:
            test_dataset: Test dataset for final evaluation
            
        Returns:
            Best model and trainer
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run tuning first.")
            
        logger.info("Training model with best hyperparameters")
        
        # Create model with best params
        config = {
            'in_channels': self.input_dim,
            'hidden_channels': self.best_params['hidden_channels'],
            'out_channels': 1,  # Binary classification or regression
            'num_layers': self.best_params['num_layers'],
            'dropout_rate': self.best_params['dropout_rate'],
            'layer_type': self.best_params['layer_type'],
            'activation': 'relu',
            'pooling': self.best_params['pooling'],
            'task_type': self.task_type
        }
        
        # Create model and optimizer
        model = MolecularGNNFactory.create_model('basic_gnn', config)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.best_params['lr'],
            weight_decay=self.best_params.get('weight_decay', 0)
        )
        
        # Create data loaders
        batch_size = self.best_params['batch_size']
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            test_loader = None
        
        # Create trainer
        trainer = ModelTrainer(model, optimizer, self.device, self.task_type)
        
        # Train model
        epochs = self.best_params.get('epochs', 100)
        patience = self.best_params.get('patience', 15)
        
        # Set up scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
        
        # Train with combined train+val data
        full_train_dataset = self.train_dataset + self.val_dataset
        full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
        
        if test_loader is not None:
            # Use test set for validation
            history = trainer.train(full_train_loader, test_loader, epochs=epochs, patience=patience, scheduler=scheduler)
            
            # Final evaluation
            test_metrics = trainer.evaluate(test_loader)
            logger.info(f"Final test metrics: {test_metrics}")
        else:
            # Use validation set for validation
            history = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience, scheduler=scheduler)
        
        self.best_model = model
        return model, trainer
    
    def save_study(self, save_dir: str):
        """Save study results and hyperparameter importance.
        
        Args:
            save_dir: Directory to save study results to
        """
        if self.best_study is None:
            raise ValueError("No study to save. Run tuning first.")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best parameters
        with open(os.path.join(save_dir, 'best_params.json'), 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save all trial results
        trial_data = []
        for trial in self.best_study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })
                
        with open(os.path.join(save_dir, 'all_trials.json'), 'w') as f:
            json.dump(trial_data, f, indent=2)
        
        # Plot hyperparameter importance
        try:
            param_importance = optuna.importance.get_param_importances(self.best_study)
            plt.figure(figsize=(10, 6))
            plt.bar(param_importance.keys(), param_importance.values())
            plt.xlabel('Hyperparameter')
            plt.ylabel('Importance')
            plt.title('Hyperparameter Importance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'param_importance.png'))
        except Exception as e:
            logger.warning(f"Could not plot parameter importance: {e}")
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(self.best_study)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'optimization_history.png'))
        
        # Plot parameter contour plots
        try:
            for param1 in self.best_params:
                for param2 in self.best_params:
                    if param1 != param2:
                        plt.figure(figsize=(8, 6))
                        optuna.visualization.matplotlib.plot_contour(
                            self.best_study, 
                            params=[param1, param2]
                        )
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, f'contour_{param1}_{param2}.png'))
        except Exception as e:
            logger.warning(f"Could not plot parameter contours: {e}")
            
        logger.info(f"Study results saved to {save_dir}")