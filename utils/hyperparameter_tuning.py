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