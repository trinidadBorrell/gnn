# Graph Autoencoder for EEG Analysis

A PyTorch Geometric implementation of Graph Autoencoder (GAE) for learning latent representations from EEG brain connectivity data with multi-subject support and Ray Tune hyperparameter optimization.

## Overview

This pipeline processes multiple subjects' EEG data, creates subject-level train/val/test splits with **no data leakage**, and trains a Graph Autoencoder with optional Ray Tune hyperparameter optimization.

### Key Principle: No Subject Leakage
- **Subject-level splitting**: All sessions from the same subject go into the same split (train, val, or test)
- **Ray Tune uses only train+val**: The optimizer never sees test data
- **Final evaluation on test**: Test set is used exactly once after selecting best hyperparameters
- **Global normalization**: All features normalized to [-1, 1] range before training

## Quick Start

### Step 1: Preprocessing & Dataset Creation

```bash
python cookbook/pipeline_many_subjects_param_op.py \
    --main_path /path/to/eeg/data \
    --coordinates_file /path/to/biosemi64.txt \
    --task lg
```

### Step 2: Training with Ray Tune

```bash
python src/train.py \
    --train_dataset output/preprocessing/data/train_dataset.pt \
    --val_dataset output/preprocessing/data/val_dataset.pt \
    --test_dataset output/preprocessing/data/test_dataset.pt \
    --tuning \
    --num_samples 20 \
    --save \
    --experiment_name my_experiment
```

## Structure

```
gnn/
├── src/
│   ├── preprocessing.py    # EEG to graph conversion
│   ├── model.py           # GAE architecture (SAGEConv layers)
│   ├── train.py           # Training with Ray Tune & visualization
│   └── inference.py       # Inference with trained model
└── cookbook/
    ├── cookbook.py                      # Single-subject pipeline
    └── pipeline_many_subjects_param_op.py  # Multi-subject pipeline
```

## Detailed Usage

### Preprocessing Arguments

**Required:**
- `--main_path`: Path to EEG data directory (structure: `sub-{ID}/ses-{num}/eeg/*.fif`)
- `--coordinates_file`: Path to `biosemi64.txt` with electrode labels

**Optional:**
- `--task`: Task name (`lg` or `rs`, default: `lg`)
- `--window_points`: Time points in window (default: 152)
- `--k_neighbors`: Nearest neighbors for adjacency matrix (default: 6)
- `--train_frac`: Train fraction (default: 0.7)
- `--val_frac`: Validation fraction (default: 0.15)
- `--test_frac`: Test fraction (default: 0.15)
- `--split_seed`: Random seed for reproducibility (default: 42)
- `--preprocessing_output_dir`: Output directory (default: `../output/preprocessing`)

**Output:**
- `train_dataset.pt`: Training graphs (subject-level split)
- `val_dataset.pt`: Validation graphs (subject-level split)
- `test_dataset.pt`: Test graphs (held-out test set)

### Training Arguments

**Required:**
- `--train_dataset`: Path to `train_dataset.pt`
- `--val_dataset`: Path to `val_dataset.pt`
- `--test_dataset`: Path to `test_dataset.pt`

**Optional:**
- `--tuning`: Enable Ray Tune hyperparameter optimization
- `--num_samples`: Number of Ray Tune trials (default: 10)
- `--hidden_channels`: Hidden layer size (default: 64)
- `--latent_dim`: Latent dimension size (default: 2)
- `--n_epochs`: Training epochs (default: 500)
- `--lr`: Learning rate (default: 0.001)
- `--save`: Save model and visualizations
- `--output_dir`: Output directory (default: `../output/training`)
- `--experiment_name`: Experiment name (default: timestamp)

**Output:**
- `gae_model_{experiment_name}.pt`: Trained model checkpoint
- `loss_history_{experiment_name}.png`: Training loss curve
- `gae_visualization_{experiment_name}.png`: 5×4 grid of GAE reconstructions

## Ray Tune Hyperparameter Optimization

### Search Space

```python
{
    'hidden_channels': [32, 64, 128],
    'latent_dim': [1, 2, 4, 8],
    'n_epochs': [200, 500, 1000],
    'lr': loguniform(1e-4, 1e-2)
}
```

### Optimization Process

1. **For each config sample:**
   - Train model on `train_graphs`
   - Compute MSE on `val_graphs`
   - Report `val_mse` to Ray Tune

2. **ASHA Scheduler:**
   - Early stops poorly performing trials
   - Allocates more resources to promising configs

3. **Best config selection:**
   - Choose config with lowest `val_mse`

4. **Final training:**
   - Retrain on `train_graphs + val_graphs` with best config
   - Evaluate **once** on `test_graphs`

### Why This is Correct

✅ **No test leakage**: Test set never used during hyperparameter search  
✅ **No subject leakage**: Subjects never split across train/val/test  
✅ **Unbiased evaluation**: Test MSE is computed only once with best config  
✅ **Proper validation**: Val set guides hyperparameter selection

## Features

- **Graph Construction**: K-nearest neighbor adjacency from electrode positions
- **Feature Extraction**: Temporal EEG windows as node features
- **Multi-Subject Support**: Subject-level train/val/test splitting
- **GAE Training**: Graph-based autoencoder with latent space learning
- **Ray Tune Integration**: Automated hyperparameter optimization with ASHA scheduler
- **Visualization**: 5×4 grid showing original, reconstruction, error, and latent space
- **Model Saving**: PyTorch checkpoints with hyperparameters

## Architecture

**Model**: GAE with GraphSAGE convolutions
- Encoder: `input → hidden (SAGEConv) → latent (SAGEConv)`
- Decoder: `latent → hidden (SAGEConv) → output (SAGEConv)`
- Loss: MSE reconstruction error
- Normalization: Features scaled to [-1, 1] range

## Complete Example

```bash
# Step 1: Create datasets with subject-level splits
python cookbook/pipeline_many_subjects_param_op.py \
    --main_path /path/to/eeg/data \
    --coordinates_file /path/to/biosemi64.txt \
    --task lg \
    --train_frac 0.7 \
    --val_frac 0.15 \
    --test_frac 0.15

# Step 2: Train with Ray Tune optimization
python src/train.py \
    --train_dataset output/preprocessing/data/train_dataset.pt \
    --val_dataset output/preprocessing/data/val_dataset.pt \
    --test_dataset output/preprocessing/data/test_dataset.pt \
    --tuning \
    --num_samples 20 \
    --save \
    --experiment_name lg_task_tuned

# Step 3 (Optional): Train without tuning using specific hyperparameters
python src/train.py \
    --train_dataset output/preprocessing/data/train_dataset.pt \
    --val_dataset output/preprocessing/data/val_dataset.pt \
    --test_dataset output/preprocessing/data/test_dataset.pt \
    --hidden_channels 128 \
    --latent_dim 8 \
    --n_epochs 1000 \
    --lr 0.0002 \
    --save
```

## Notes

- **GAE model** (`src/model.py`) uses `SAGEConv` layers and is not modified during pipeline execution
- **Normalization** is applied to all graphs globally ([-1, 1] range) before training
- **Subject IDs** are extracted from directory names (`sub-{ID}`)
- **Sessions** from the same subject always stay together in one split
- **Data structure**: Expected format is `main_path/sub-{ID}/ses-{num}/eeg/sub-{ID}_ses-{num}_task-{task}_acq-01_epo.fif`


