#!/usr/bin/env python3
"""
VGAE Cookbook
-------------
This cookbook orchestrates the complete pipeline for training a Variational Graph Autoencoder:
1. Preprocessing: Load EEG data and create graph structure
2. Training: Train the GAE model with visualization and model saving

Usage:
    python cookbook.py --config config_example.json
    
Or modify the parameters in this file directly.
"""

import sys
import os
import argparse
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import EEGtoGraph
from train_unique_graph import TrainGAE
from inference import InferenceGAE


class VGAEPipeline:
    """Complete pipeline for VGAE training from EEG data"""
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.data = None
        self.adjacency = None
        self.features = None
        self.labels = None
        self.trainer = None
        
    def step1_preprocessing(self):
        """Step 1: Load data and create graph structure"""
        print("\n" + "="*70)
        print("STEP 1: PREPROCESSING")
        print("="*70)
        
        # Load electrode coordinates
        print("\nLoading electrode coordinates...")
        try:
            from eeg_positions import get_elec_coords
            
            # Load labels from file
            labels = np.loadtxt(self.config['coordinates_file'], usecols=(0,), dtype=str)
            
            # Get electrode coordinates
            coords_data = get_elec_coords(system='1005', as_mne_montage=False)
            
            # Filter to only include biosemi64 electrodes
            coords_df = coords_data[coords_data['label'].isin(labels)].copy()
            
            print(f"✓ Loaded {len(coords_df)} electrode coordinates")
            
        except Exception as e:
            print(f"\nERROR loading coordinates: {str(e)}")
            raise
        
        # Create graph structure
        print("\nCreating graph from EEG data...")
        self.data, self.adjacency, self.features, self.labels, distance_matrix = EEGtoGraph.create_graph(
            coords_df=coords_df,
            main_path=self.config['main_path'],
            subject_id=self.config['subject_id'],
            session_num=self.config['session_num'],
            task=self.config.get('task', 'lg'),
            window_points=self.config.get('window_points', 152),
            epoch=self.config.get('epoch', 0),
            k=self.config.get('k_neighbors', 6),
            output_dir=self.config['preprocessing_output_dir'],
            corr_type=self.config.get('corr_type', 'pearson'),
            save=self.config.get('save_preprocessing', True),
            plot_neighbors=self.config.get('plot_neighbors', False)
        )
        
        print("\n✓ Preprocessing completed successfully!")
        print(f"  - Graph nodes: {self.data.x.shape[0]}")
        print(f"  - Node features: {self.data.x.shape[1]}")
        print(f"  - Graph edges: {self.data.edge_index.shape[1]}")
        
        return self.data
    
    def step2_training(self):
        """Step 2: Train the GAE model"""
        print("\n" + "="*70)
        print("STEP 2: TRAINING")
        print("="*70)
        
        if self.data is None:
            raise ValueError("Must run preprocessing first!")
        
        # Get input channels from data
        in_channels = self.data.x.shape[1]
        
        # Initialize trainer
        self.trainer = TrainGAE(
            data=self.data,
            in_channels=in_channels,
            hidden_channels=self.config.get('hidden_channels', 64),
            latent_dim=self.config.get('latent_dim', 1),
            n_epochs=self.config.get('n_epochs', 100),
            lr=self.config.get('learning_rate', 0.01)
        )
        
        # Training mode
        mode = self.config.get('training_mode', 'full')
        
        if mode == 'cv':
            # Perform cross-validation
            print("\nPerforming cross-validation...")
            fold_results, mean_loss, std_loss = self.trainer.cross_validate(
                n_splits=self.config.get('n_splits', 5)
            )
            
            # Train final model on full dataset
            print("\nTraining final model on full dataset...")
            final_model, latent_repr = self.trainer.train_full(
                save_results=self.config.get('save_model', True),
                output_dir=self.config['training_output_dir'],
                experiment_name=self.config.get('experiment_name', None)
            )
            
        else:
            # Train on full dataset only
            print("\nTraining on full dataset...")
            final_model, latent_repr = self.trainer.train_full(
                save_results=self.config.get('save_model', True),
                output_dir=self.config['training_output_dir'],
                experiment_name=self.config.get('experiment_name', None)
            )
        
        print("\n✓ Training completed successfully!")
        
        return final_model, latent_repr
    
    def step3_inference(self, model_path):
        """Step 3: Run inference with a trained model"""
        print("\n" + "="*70)
        print("STEP 3: INFERENCE")
        print("="*70)
        
        if self.data is None:
            raise ValueError("Must load data first!")
        
        print(f"\nLoading model from: {model_path}")
        
        # Initialize inference
        inference_engine = InferenceGAE(model_path, self.data)
        
        # Run inference
        x_reconstructed, latent = inference_engine.run_inference()
        
        # Create visualization
        print("\nGenerating visualization...")
        inference_engine.visualize_results(
            output_dir=self.config.get('inference_output_dir', '../output/inference'),
            experiment_name=self.config.get('experiment_name', None)
        )
        
        print("\n✓ Inference completed successfully!")
        
        return x_reconstructed, latent
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("\n" + "="*70)
        print("VGAE PIPELINE - Full Execution")
        print("="*70)
        experiment = self.config.get('experiment_name', 'default')
        print(f"\nExperiment: {experiment}")
        print(f"Subject: {self.config['subject_id']}")
        print(f"Session: {self.config['session_num']}")
        
        # Step 1: Preprocessing
        self.step1_preprocessing()
        
        # Step 2: Training
        model, latent = self.step2_training()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nOutputs saved to:")
        print(f"  - Preprocessing: {self.config['preprocessing_output_dir']}")
        print(f"  - Training: {self.config['training_output_dir']}")
        
        return model, latent, self.data
    
    def run_inference_pipeline(self, model_path):
        """Run inference pipeline on new data"""
        print("\n" + "="*70)
        print("VGAE INFERENCE PIPELINE")
        print("="*70)
        print(f"\nSubject: {self.config['subject_id']}")
        print(f"Session: {self.config['session_num']}")
        print(f"Model: {model_path}")
        
        # Step 1: Preprocessing (load test data)
        self.step1_preprocessing()
        
        # Step 3: Inference
        x_reconstructed, latent = self.step3_inference(model_path)
        
        print("\n" + "="*70)
        print("INFERENCE PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nOutputs saved to:")
        print(f"  - Preprocessing: {self.config['preprocessing_output_dir']}")
        print(f"  - Inference: {self.config.get('inference_output_dir', '../output/inference')}")
        
        return x_reconstructed, latent, self.data


def create_default_config():
    """Create a default configuration dictionary"""
    config = {
        # Data paths
        'main_path': '/path/to/eeg/data',
        'coordinates_file': '/path/to/biosemi64.txt',
        
        # Subject info
        'subject_id': '01',
        'session_num': '01',
        'task': 'lg',
        
        # Preprocessing parameters
        'window_points': 152,
        'epoch': 0,
        'k_neighbors': 6,
        'corr_type': 'pearson',
        'plot_neighbors': False,
        'save_preprocessing': True,
        'preprocessing_output_dir': '../output/preprocessing',
        
        # Training parameters
        'hidden_channels': 64,
        'latent_dim': 1,
        'n_epochs': 100,
        'learning_rate': 0.01,
        'training_mode': 'full',  # 'cv' or 'full'
        'n_splits': 5,  # for cross-validation
        'save_model': True,
        'training_output_dir': '../output/training',
        
        # Experiment info
        'experiment_name': None  # defaults to timestamp
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description='VGAE Pipeline for EEG Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--main_path', type=str, required=True,
                        help='Path to the main EEG data directory')
    parser.add_argument('--coordinates_file', type=str, required=True,
                        help='Path to biosemi64.txt file with electrode labels')
    
    # Subject/session arguments (optional with defaults for inference)
    parser.add_argument('--subject_id', type=str, default='AA069',
                        help='Subject ID (default: AA069 for inference)')
    parser.add_argument('--session_num', type=str, default='01',
                        help='Session number (default: 01)')
    
    # Optional preprocessing arguments
    parser.add_argument('--task', type=str, default='lg',
                        help='Task name')
    parser.add_argument('--window_points', type=int, default=152,
                        help='Number of time points in the window')
    parser.add_argument('--epoch', type=int, default=0,
                        help='Epoch number to process')
    parser.add_argument('--k_neighbors', type=int, default=6,
                        help='Number of nearest neighbors for adjacency matrix')
    parser.add_argument('--preprocessing_output_dir', type=str, default='../output/preprocessing',
                        help='Directory to save preprocessing outputs')
    
    # Optional training arguments
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels')
    parser.add_argument('--latent_dim', type=int, default=1,
                        help='Latent dimension size')
    parser.add_argument('--n_epochs', type=int, default=2000,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--training_mode', type=str, default='full', choices=['cv', 'full'],
                        help='Training mode: "cv" for cross-validation, "full" for full dataset')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--training_output_dir', type=str, default='../output/training',
                        help='Directory to save training outputs')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='Pipeline mode: "train" to train a new model, "inference" to run inference')
    
    # Inference-specific arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model for inference (required if mode=inference)')
    parser.add_argument('--inference_output_dir', type=str, default='../output/inference',
                        help='Directory to save inference outputs')
    
    # Experiment info
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment (defaults to timestamp)')
    parser.add_argument('--plot_neighbors', action='store_true',
                        help='Plot k-nearest neighbors visualization')
    
    args = parser.parse_args()
    
    # Validate inference mode requirements
    if args.mode == 'inference' and args.model_path is None:
        parser.error("--model_path is required when mode=inference")
    
    # Create config from arguments
    config = vars(args)
    config['save_preprocessing'] = True
    config['save_model'] = True
    
    # Create and run pipeline
    print("\n" + "="*70)
    print("Initializing VGAE Pipeline")
    print("="*70)
    
    pipeline = VGAEPipeline(config)
    
    if args.mode == 'train':
        # Training mode
        model, latent, data = pipeline.run_full_pipeline()
    else:
        # Inference mode
        x_reconstructed, latent, data = pipeline.run_inference_pipeline(args.model_path)
    
    return 0


if __name__ == '__main__':
    exit(main())
