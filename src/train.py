import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import KFold
from model import GAE

# Plotting style parameters
COLOR = "black"
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (14, 9),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "legend.fontsize": "medium",
        "legend.title_fontsize": 18,
        "axes.titlesize": 18,
        "axes.labelsize": "large",
        "ytick.labelsize": 12,
        "xtick.labelsize": 12,
        # colour-consistent theme
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "grid.color": COLOR,
    }
)
plt.rcParams["text.latex.preamble"] = r"\usepackage[version=3]{mhchem}"

class TrainGAE:
    def __init__(self, data, in_channels, hidden_channels, latent_dim, n_epochs=100, lr=0.01):
        self.data = data
        self.data_raw = data.x.clone()  # Store original raw data
        
        # Normalize input data to [-1, 1]
        self.x_min = data.x.min()
        self.x_max = data.x.max()
        self.x_range = self.x_max - self.x_min
        
        if self.x_range > 0:
            # Normalize to [0, 1] then to [-1, 1]
            self.data.x = 2 * (data.x - self.x_min) / self.x_range - 1
        else:
            # If all values are the same, set to 0
            self.data.x = torch.zeros_like(data.x)
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.lr = lr
    
    def denormalize(self, x_normalized):
        """Convert from [-1, 1] back to original scale"""
        if self.x_range > 0:
            return (x_normalized + 1) / 2 * self.x_range + self.x_min
        else:
            return x_normalized + self.x_min

    def train_single_fold(self, train_mask, val_mask):
        """Train on a single fold and return validation loss"""
        # Initialize model
        model = GAE(
            self.in_channels,
            self.hidden_channels,
            self.latent_dim 
        )

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Loss function MSE
        criterion = torch.nn.MSELoss()

        best_val_loss = float('inf')
        
        model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            
            # Forward pass on full graph
            x_reconstructed, z = model(self.data.x, self.data.edge_index)
            
            # Training loss (only on training nodes)
            train_loss = criterion(x_reconstructed[train_mask], self.data.x[train_mask])
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            
            # Validation loss
            model.eval()
            with torch.no_grad():
                x_reconstructed_val, _ = model(self.data.x, self.data.edge_index)
                val_loss = criterion(x_reconstructed_val[val_mask], self.data.x[val_mask])
            model.train()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
            
            if epoch % 20 == 0:
                print(f'  Epoch {epoch:03d}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        return best_val_loss, model

    def cross_validate(self, n_splits=5):
        """Perform k-fold cross-validation"""
        print(f"\n{'='*60}")
        print(f"Starting {n_splits}-Fold Cross-Validation")
        print(f"{'='*60}\n")
        
        # Create node indices
        num_nodes = self.data.x.shape[0]
        node_indices = np.arange(num_nodes)
        
        # Initialize k-fold
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(node_indices)):
            print(f"\nFold {fold + 1}/{n_splits}")
            print("-" * 40)
            
            # Create masks
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            
            print(f"Training nodes: {train_mask.sum().item()}")
            print(f"Validation nodes: {val_mask.sum().item()}")
            
            # Train on this fold
            val_loss, model = self.train_single_fold(train_mask, val_mask)
            fold_results.append(val_loss)
            
            print(f"Best validation loss: {val_loss:.4f}")
        
        # Calculate statistics
        mean_val_loss = np.mean(fold_results)
        std_val_loss = np.std(fold_results)
        
        print(f"\n{'='*60}")
        print("Cross-Validation Results")
        print(f"{'='*60}")
        print(f"Fold losses: {[f'{loss:.4f}' for loss in fold_results]}")
        print(f"Mean validation loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return fold_results, mean_val_loss, std_val_loss

    def save_model_and_visualizations(self, model, loss_history=None, output_dir='../output/training', experiment_name=None):
        """
        Save the trained model and create 4-column visualization plot plus loss plot
        
        Args:
            model: Trained GAE model
            loss_history: List of loss values across epochs
            output_dir: Directory to save outputs
            experiment_name: Name for this experiment (defaults to timestamp)
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        model_dir = os.path.join(output_dir, 'models')
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f'gae_model_{experiment_name}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'latent_dim': self.latent_dim,
            'n_epochs': self.n_epochs,
            'lr': self.lr
        }, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            x_reconstructed, z = model(self.data.x, self.data.edge_index)
            
            # Calculate reconstruction error in normalized space
            error_normalized = torch.abs(self.data.x - x_reconstructed)
            
            # Denormalize for visualization in original scale
            original_denorm = self.denormalize(self.data.x)
            reconstruction_denorm = self.denormalize(x_reconstructed)
            error_denorm = torch.abs(original_denorm - reconstruction_denorm)
            
            # Convert to numpy for plotting (use normalized values)
            original = self.data.x.cpu().numpy()
            reconstruction = x_reconstructed.cpu().numpy()
            error_np = error_normalized.cpu().numpy()
            latent = z.cpu().numpy()
        
        # Create 4-column visualization
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # Plot 1: Original Feature Matrix (normalized)
        im1 = axes[0].imshow(original, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[0].set_title('Original Feature Matrix\n(Normalized [-1, 1])', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Nodes')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Reconstructed Feature Matrix (normalized)
        im2 = axes[1].imshow(reconstruction, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[1].set_title('Reconstruction\n(Normalized [-1, 1])', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Nodes')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot 3: Reconstruction Error (in normalized space)
        im3 = axes[2].imshow(error_np, aspect='auto', cmap='Reds')
        axes[2].set_title('Reconstruction Error\n(Normalized Space)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Features')
        axes[2].set_ylabel('Nodes')
        plt.colorbar(im3, ax=axes[2])
        
        # Plot 4: Latent Space Representation
        if self.latent_dim == 1:
            # For 1D latent space, plot as a heatmap
            im4 = axes[3].imshow(latent, aspect='auto', cmap='coolwarm')
            axes[3].set_title('Latent Space (1D)', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Latent Dimension')
            axes[3].set_ylabel('Nodes')
            plt.colorbar(im4, ax=axes[3])
        elif self.latent_dim == 2:
            # For 2D latent space, scatter plot
            axes[3].scatter(latent[:, 0], latent[:, 1], alpha=0.6, c=range(len(latent)), cmap='coolwarm')
            axes[3].set_title('Latent Space (2D)', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Latent Dim 1')
            axes[3].set_ylabel('Latent Dim 2')
            axes[3].grid(True, alpha=0.3)
        else:
            # For higher dimensions, show as heatmap
            im4 = axes[3].imshow(latent, aspect='auto', cmap='coolwarm')
            axes[3].set_title(f'Latent Space ({self.latent_dim}D)', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Latent Dimensions')
            axes[3].set_ylabel('Nodes')
            plt.colorbar(im4, ax=axes[3])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_dir, f'gae_visualization_{experiment_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        plt.close()
        
        # Plot loss history if available
        if loss_history is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(loss_history, linewidth=2)
            ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            loss_plot_path = os.path.join(plots_dir, f'loss_history_{experiment_name}.png')
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            print(f"Loss history plot saved to: {loss_plot_path}")
            plt.close()
        
        # Calculate and print metrics (in normalized space)
        mse = F.mse_loss(x_reconstructed, self.data.x).item()
        mae = torch.mean(error_normalized).item()
        
        # Also calculate metrics in original scale
        mse_denorm = F.mse_loss(reconstruction_denorm, original_denorm).item()
        mae_denorm = torch.mean(error_denorm).item()
        
        print("\n" + "="*60)
        print("Reconstruction Metrics")
        print("="*60)
        print("Normalized space [-1, 1]:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print("\nOriginal scale:")
        print(f"  MSE: {mse_denorm:.10f}")
        print(f"  MAE: {mae_denorm:.10f}")
        print("="*60 + "\n")
        
        return model_path, plot_path

    def train_full(self, save_results=False, output_dir='../output/training', experiment_name=None):
        """Train on full dataset (no validation split)"""
        print(f"\n{'='*60}")
        print("Training on Full Dataset")
        print(f"{'='*60}\n")
        
        # Print initial data statistics
        print(f"Original (raw) data statistics:")
        print(f"  Shape: {self.data_raw.shape}")
        print(f"  Range: [{self.data_raw.min():.6f}, {self.data_raw.max():.6f}]")
        print(f"  Mean: {self.data_raw.mean():.6f}, Std: {self.data_raw.std():.6f}")
        
        print(f"\nNormalized data statistics (used for training):")
        print(f"  Shape: {self.data.x.shape}")
        print(f"  Range: [{self.data.x.min():.6f}, {self.data.x.max():.6f}]")
        print(f"  Mean: {self.data.x.mean():.6f}, Std: {self.data.x.std():.6f}")
        print(f"  Non-zero elements: {(self.data.x != 0).sum().item()} / {self.data.x.numel()}")
        print()
        
        # Initialize model
        model = GAE(
            self.in_channels,
            self.hidden_channels,
            self.latent_dim 
        )

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        # Loss function MSE
        criterion = torch.nn.MSELoss()

        # Track loss history
        loss_history = []
        
        model.train()

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            x_reconstructed, z = model(self.data.x, self.data.edge_index)
            
            # Debug: Check reconstruction quality (in normalized space [-1, 1])
            if epoch % 100 == 0:
                diff = x_reconstructed - self.data.x
                print(f"  Data (normalized)  - Mean: {self.data.x.mean():.6f}, Std: {self.data.x.std():.6f}, Range: [{self.data.x.min():.6f}, {self.data.x.max():.6f}]")
                print(f"  Recon (normalized) - Mean: {x_reconstructed.mean():.6f}, Std: {x_reconstructed.std():.6f}, Range: [{x_reconstructed.min():.6f}, {x_reconstructed.max():.6f}]")
                print(f"  Diff (normalized)  - Mean: {diff.mean():.6f}, Std: {diff.std():.6f}, Range: [{diff.min():.6f}, {diff.max():.6f}]")
            
               
            # Reconstruction loss
            loss = criterion(x_reconstructed, self.data.x) 
            
            # Backward pass
            loss.backward()
            optimizer.step()

            if epoch > 200:
                # Store loss
                loss_history.append(loss.item())
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch:03d}, Loss: {loss.item():.6f}')

        # Evaluation
        model.eval()
        with torch.no_grad():
            x_reconstructed, z = model(self.data.x, self.data.edge_index)
            
            print(f"\n{'='*60}")
            print("Final Results")
            print(f"{'='*60}")
            print(f"Original features shape: {self.data.x.shape}")
            print(f"Latent representation shape: {z.shape}")
            print(f"Reconstructed features shape: {x_reconstructed.shape}")
            
            # Calculate reconstruction error in normalized space
            reconstruction_error_norm = F.mse_loss(x_reconstructed, self.data.x)
            print(f"\nFinal reconstruction error (normalized [-1, 1]): {reconstruction_error_norm.item():.6f}")
            
            # Calculate reconstruction error in original scale
            original_denorm = self.denormalize(self.data.x)
            reconstruction_denorm = self.denormalize(x_reconstructed)
            reconstruction_error_orig = F.mse_loss(reconstruction_denorm, original_denorm)
            print(f"Final reconstruction error (original scale): {reconstruction_error_orig.item():.10f}")
            
            # Visualize latent space
            print("\nLatent space values (first 10 nodes):")
            print(z[:10].squeeze())
            print(f"{'='*60}\n")
        
        # Save model and visualizations if requested
        if save_results:
            self.save_model_and_visualizations(model, loss_history, output_dir, experiment_name)
        
        return model, z

def main(data=None):
    """
    Main training function
    
    Args:
        data: PyTorch Geometric Data object. If None, expects to be run from command line with data available
    """
    parser = argparse.ArgumentParser(description='Train Graph Autoencoder with Cross-Validation')

    
    parser.add_argument('--data', type=str, required=True,
                        help='Data in geometric torch format')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels (default: 64)')
    parser.add_argument('--latent_dim', type=int, default=1,
                        help='Latent dimension size (default: 1)')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--mode', type=str, default='cv', choices=['cv', 'full'],
                        help='Training mode: "cv" for cross-validation, "full" for full dataset (default: cv)')
    parser.add_argument('--save', action='store_true',
                        help='Save model and visualizations')
    parser.add_argument('--output_dir', type=str, default='../output/training',
                        help='Directory to save outputs (default: ../output/training)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment (defaults to timestamp)')
    
    args = parser.parse_args()
    
    in_channels = args.data.x.shape[1]
    
    print("\nDataset Info:")
    print(f"Number of nodes: {args.data.x.shape[0]}")
    print(f"Number of features: {in_channels}")
    print(f"Number of edges: {args.data.edge_index.shape[1]}")
    
    trainer = TrainGAE(
        data=args.data,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        n_epochs=args.n_epochs,
        lr=args.lr
    )
    
    if args.mode == 'cv':
        # Perform cross-validation
        fold_results, mean_loss, std_loss = trainer.cross_validate(n_splits=args.n_splits)
        
        print("\nTraining final model on full dataset...")
        final_model, latent_repr = trainer.train_full(
            save_results=args.save,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
        
    else:
        # Train on full dataset only
        final_model, latent_repr = trainer.train_full(
            save_results=args.save,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
    
    return final_model, latent_repr, trainer

if __name__ == '__main__':
    main()