import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Callable
import os
from torch.utils.data import WeightedRandomSampler, Dataset # Add this import if not present

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping implementation to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs with no improvement after which training will stop.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore model weights from best value.
            mode: One of {'min', 'max'}. For 'min', training stops when monitored value stops decreasing.
                  For 'max', training stops when monitored value stops increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, model, current_score):
        """Check if training should be stopped."""
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            improved = False
            if self.mode == 'min' and current_score <= (self.best_score - self.min_delta):
                improved = True
            elif self.mode == 'max' and current_score >= (self.best_score + self.min_delta):
                improved = True
                
            if improved:
                self.best_score = current_score
                self.counter = 0
                if self.restore_best_weights:
                    self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
    
    def restore_weights(self, model):
        """Restore best weights to the model."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelTrainer:
    """Trainer class for molecular GNN models."""
    
    def __init__(self, model, optimizer=None, device=None, task_type: str = 'classification'):
        """
        Args:
            model: PyTorch model to train
            optimizer: PyTorch optimizer (if None, Adam will be used)
            device: Device to use for training (if None, will use CUDA if available)
            task_type: 'classification' or 'regression'
        """
        self.model = model
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        
        # Set optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
            
        self.task_type = task_type
        
        # Set criterion based on task type
        if task_type == 'classification':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:  # regression
            self.criterion = torch.nn.MSELoss()
            
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            

            # Forward pass
            output = self.model(data.x, data.edge_index, data.batch, 
                              edge_attr=data.edge_attr.float() if hasattr(data, 'edge_attr') else None)
            
            # Calculate loss
            data.y = data.y.long() if self.task_type == 'classification' else data.y.float()
            target = data.y  # Use the target as is for classification
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            
        return total_loss / len(train_loader.dataset)
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a data loader.
        
        Args:
            loader: DataLoader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        y_true = []
        y_pred = []
        y_scores = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                
                # Forward pass
                output = self.model(data.x, data.edge_index, data.batch, 
                                  edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None)
                
                # Convert target to Long for classification
                data.y = data.y.long() if self.task_type == 'classification' else data.y.float()
                target = data.y
                
                # Calculate loss
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.num_graphs
                
                # Store predictions
                if self.task_type == 'classification':
                    prob = torch.softmax(output, dim=1).cpu().numpy()  # Use softmax for multi-class probabilities
                    threshold = 0.4
                    pred = (prob[:, 1] > threshold).astype(int)  # Get the predicted class
                    y_scores.extend(prob[:, 1])  # Store probabilities for the positive class (class 1)
                else:  # regression
                    pred = output.cpu().numpy()
                    y_scores.extend(pred.flatten())
                    
                y_pred.extend(pred.flatten())
                y_true.extend(target.cpu().numpy().flatten())

        
        # Calculate metrics
        results = {'loss': total_loss / len(loader.dataset)}
        
        if self.task_type == 'classification':
            results.update({
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_scores)
            })
        else:  # regression
            results.update({
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            })
            
        return results
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, patience: int = 10,
              callbacks: List[Callable] = None,
              scheduler=None, num_classes: int = 2) -> Dict[str, Any]:
        """Train the model.

        Args:
            train_loader: DataLoader for training data (potentially with sampler already configured)
            val_loader: DataLoader for validation data
            epochs: Maximum number of epochs to train
            patience: Early stopping patience
            callbacks: List of callbacks to call after each epoch
            scheduler: Learning rate scheduler

        Returns:
            Training history.
        """
        if callbacks is None:
            callbacks = []

        # Add early stopping
        early_stopping = EarlyStopping(
            patience=patience,
            mode='max' if self.task_type == 'classification' else 'min',
            restore_best_weights=True
        )

        logger.info(f"Starting training for {epochs} epochs with {self.task_type} task")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            
            # Update learning rate scheduler if provided
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            for key, value in val_metrics.items():
                if key not in self.history['metrics']:
                    self.history['metrics'][key] = []
                self.history['metrics'][key].append(value)
            
            # Call callbacks
            for callback in callbacks:
                callback(self, epoch, train_loss, val_metrics)
            
            # Check early stopping
            if self.task_type == 'classification':
                early_stopping(self.model, val_metrics.get('roc_auc', val_metrics.get('f1', -val_loss)))
            else:
                early_stopping(self.model, -val_metrics.get('r2', val_loss))
            
            if epoch % 5 == 0:
                if self.task_type == 'classification':
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}, "
                              f"Val AUC: {val_metrics.get('roc_auc', 0):.4f}, "
                              f"Val F1: {val_metrics.get('f1', 0):.4f}")
                else:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}, "
                              f"Val RMSE: {val_metrics.get('rmse', 0):.4f}, "
                              f"Val R²: {val_metrics.get('r2', 0):.4f}")
            
            # Early stopping check
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
        # Restore best weights
        early_stopping.restore_weights(self.model)
        
        # Final evaluation
        train_metrics = self.evaluate(train_loader)
        val_metrics = self.evaluate(val_loader)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final training metrics: {train_metrics}")
        logger.info(f"Final validation metrics: {val_metrics}")
        
        return self.history
    
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with the model.
        
        Args:
            loader: DataLoader for prediction data
            
        Returns:
            Tuple of (true values, predicted values)
        """
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                output = self.model(data.x, data.edge_index, data.batch, 
                                  edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None)
                
                # Get predictions
                if self.task_type == 'classification':
                    pred = torch.sigmoid(output).cpu().numpy()
                else:  # regression
                    pred = output.cpu().numpy()
                
                y_pred.extend(pred.flatten())
                y_true.extend(data.y.cpu().numpy().flatten())
                
        return np.array(y_true), np.array(y_pred)
    
    def save_model(self, save_path: str):
        """Save model and training history.
        
        Args:
            save_path: Path to save the model to
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model and metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'task_type': self.task_type
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a saved model.
        
        Args:
            load_path: Path to the saved model
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
            
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.task_type = checkpoint.get('task_type', self.task_type)
        
        logger.info(f"Model loaded from {load_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history.
        
        Args:
            save_path: Path to save the plot to (if None, will display instead)
        """
        if not self.history['train_loss']:
            logger.warning("No training history available to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot training and validation loss
        plt.subplot(2, 1, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot other metrics
        plt.subplot(2, 1, 2)
        for key, values in self.history['metrics'].items():
            if key != 'loss':
                plt.plot(values, label=key)
                
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Validation Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_confusion_matrix(self, loader: DataLoader, save_path: Optional[str] = None):
        """Plot confusion matrix for classification.
        
        Args:
            loader: DataLoader for data
            save_path: Path to save the plot to (if None, will display instead)
        """
        if self.task_type != 'classification':
            logger.warning("Confusion matrix only applicable for classification tasks")
            return
            
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        y_true, y_pred = self.predict(loader)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_roc_curve(self, loader: DataLoader, save_path: Optional[str] = None):
        """Plot ROC curve for classification.
        
        Args:
            loader: DataLoader for data
            save_path: Path to save the plot to (if None, will display instead)
        """
        if self.task_type != 'classification':
            logger.warning("ROC curve only applicable for classification tasks")
            return
            
        from sklearn.metrics import roc_curve, auc
        
        y_true, y_pred = self.predict(loader)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC curve plot saved to {save_path}")
        else:
            plt.show()
    
def get_weighted_sampler(dataset: Dataset, num_classes: int, imbalance_threshold: float = 2.0) -> Optional[WeightedRandomSampler]:
    """
    Check dataset balance and return a WeightedRandomSampler if imbalanced.

    Args:
        dataset: PyTorch Dataset containing the data. Assumes each item `data`
                 has a `data.y` attribute that is a tensor containing a single class label.
        num_classes: Number of classes in the dataset.
        imbalance_threshold: Ratio between the largest and smallest class count
                             to consider the dataset imbalanced.

    Returns:
        WeightedRandomSampler if imbalanced, otherwise None.
    """
    logger.info("Checking dataset balance...")
    if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
         logger.error("Provided dataset object does not behave like a standard PyTorch Dataset.")
         # Attempt to handle if it's a DataLoader by accessing its dataset, but this is fragile
         if hasattr(dataset, 'dataset') and isinstance(dataset.dataset, Dataset):
             logger.warning("Received DataLoader instead of Dataset, accessing underlying dataset.")
             dataset = dataset.dataset
         else:
             logger.error("Cannot determine dataset from input.")
             return None # Cannot proceed

    class_counts = [0] * num_classes
    try:
        for i in range(len(dataset)):
            data = dataset[i]
            if not hasattr(data, 'y') or data.y is None:
                logger.warning(f"Sample {i} has no attribute 'y' or it is None. Skipping for weighting.")
                continue
            # Assuming data.y is a tensor containing the single class label for this graph
            if data.y.numel() != 1:
                 logger.warning(f"Sample {i} has data.y with {data.y.numel()} elements. Expecting 1 for weighting. Using first element.")
            label = int(data.y.item()) # Get the scalar value and convert to int
            if 0 <= label < num_classes:
                class_counts[label] += 1
            else:
                logger.warning(f"Sample {i} has out-of-bounds label: {label}. Skipping.")
    except Exception as e:
        logger.error(f"Error accessing dataset elements for weighting: {e}")
        return None # Cannot calculate weights

    logger.info(f"Class distribution: {class_counts}")

    # Check for imbalance
    min_count = min(c for c in class_counts if c > 0)
    max_count = max(class_counts)

    if min_count == 0:
        logger.warning("One or more classes have zero samples. Cannot use weighted sampling.")
        return None

    is_imbalanced = (max_count / min_count) > imbalance_threshold

    if is_imbalanced:
        logger.info(f"Dataset is imbalanced (ratio {max_count / min_count:.2f} > {imbalance_threshold}). Applying weighted sampling.")
        class_weights = [1.0 / count if count > 0 else 0.0 for count in class_counts]

        # Assign one weight per sample in the dataset
        sample_weights = [0.0] * len(dataset)
        try:
            for i in range(len(dataset)):
                 data = dataset[i]
                 if not hasattr(data, 'y') or data.y is None: continue # Skip samples skipped above
                 label = int(data.y.item())
                 if 0 <= label < num_classes:
                     sample_weights[i] = class_weights[label]
                 # Samples with bad labels will have weight 0.0
        except Exception as e:
             logger.error(f"Error assigning sample weights: {e}")
             return None

        # Ensure sample_weights has the correct length
        if len(sample_weights) != len(dataset):
             logger.error(f"Mismatch in sample_weights length ({len(sample_weights)}) and dataset length ({len(dataset)}).")
             return None

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info(f"Class weights: {class_weights}")
        return sampler
    else:
        logger.info(f"Dataset is considered balanced (ratio {max_count / min_count:.2f} <= {imbalance_threshold}). No weighted sampling applied.")
        return None