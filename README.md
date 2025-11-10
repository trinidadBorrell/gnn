# Graph Autoencoder for EEG Analysis

A PyTorch Geometric implementation of Graph Autoencoder (GAE) for learning latent representations from EEG brain connectivity data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
cd cookbook
python cookbook.py \
  --main_path /path/to/eeg/data \
  --subject_id 01 \
  --session_num 01 \
  --coordinates_file /path/to/biosemi64.txt
```

## Structure

```
gnn/
├── src/
│   ├── preprocessing.py    # EEG to graph conversion
│   ├── model.py           # GAE architecture
│   ├── train.py           # Training with visualization
│   └── inference.py           # Inference with trined model
└── cookbook/
    └── cookbook.py        # Complete pipeline
```

## Features

- **Graph Construction**: K-nearest neighbor adjacency from electrode positions
- **Feature Extraction**: Temporal EEG windows as node features
- **GAE Training**: Graph-based autoencoder with latent space learning
- **Visualization**: 4-column plot (original, reconstruction, error, latent space)
- **Model Saving**: PyTorch checkpoints with hyperparameters

## Output

Training produces:
- Trained model checkpoint (`.pt`)
- 4-column visualization showing original features, reconstruction, error, and latent space
- Preprocessing artifacts (adjacency matrix, feature matrices)

## Parameters

Key training parameters:
- `--hidden_channels`: Hidden layer size (default: 64)
- `--latent_dim`: Latent space dimensionality (default: 1)
- `--n_epochs`: Training epochs (default: 100)
- `--training_mode`: `full` or `cv` for cross-validation

See `cookbook/cookbook.py --help` for all options.

## Architecture

**Model**: GAE with GraphSAGE convolutions
- Encoder: `input → hidden → latent`
- Decoder: `latent → hidden → output`
- Loss: MSE reconstruction error


