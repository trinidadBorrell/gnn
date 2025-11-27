"""
VGAE Multi-Subject Cookbook
---------------------------
This cookbook orchestrates the preprocessing pipeline for *multiple* subjects and
creates subject-level train/val/test splits with no data leakage across subjects.

Steps:
1. Load electrode coordinates.
2. Traverse the main EEG data directory and build a graph per subject/session
   using EEGtoGraph.create_graph.
3. Split subjects into train/val/test (subject-level, no subject appears in more
   than one split).
4. Wrap each split in GraphAutoencoderDataset.
5. Save the three datasets to disk for later training (e.g., by train.py).

Usage (example):
    python pipeline_many_subjects_param_op.py \
        --main_path /path/to/data \
        --coordinates_file /path/to/biosemi64.txt

"""

import sys
import os
import argparse
from typing import List, Tuple, Dict

import numpy as np


# Add src directory to path so we can import preprocessing utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import EEGtoGraph, GraphAutoencoderDataset


def load_electrode_coordinates(coordinates_file: str):
    """Load electrode coordinates and return a filtered coordinates DataFrame."""
    try:
        from eeg_positions import get_elec_coords

        labels = np.loadtxt(coordinates_file, usecols=(0,), dtype=str)
        coords_data = get_elec_coords(system="1005", as_mne_montage=False)
        coords_df = coords_data[coords_data["label"].isin(labels)].copy()

        print(f"Loaded {len(coords_df)} electrode coordinates")
        return coords_df
    except Exception as e:
        print(f"ERROR loading coordinates: {str(e)}")
        raise


def discover_subject_sessions(main_path: str, task: str) -> List[Tuple[str, str]]:
    """Discover all (subject_id, session_num) pairs that contain the desired task.

    This assumes a directory structure like:
        main_path/sub-{ID}/ses-{num}/eeg/sub-{ID}_ses-{num}_task-{task}_acq-01_epo.fif
    """
    subject_sessions: List[Tuple[str, str]] = []

    if not os.path.isdir(main_path):
        raise FileNotFoundError(f"main_path does not exist or is not a directory: {main_path}")

    for subj_name in sorted(os.listdir(main_path)):
        subj_path = os.path.join(main_path, subj_name)
        if not os.path.isdir(subj_path) or not subj_name.startswith("sub-"):
            continue

        subject_id = subj_name.split("-", 1)[1]

        for ses_name in sorted(os.listdir(subj_path)):
            ses_path = os.path.join(subj_path, ses_name)
            if not os.path.isdir(ses_path) or not ses_name.startswith("ses-"):
                continue

            session_num = ses_name.split("-", 1)[1]
            eeg_dir = os.path.join(ses_path, "eeg")
            if not os.path.isdir(eeg_dir):
                continue

            # Look for the expected fif file for this task
            expected_fname = f"sub-{subject_id}_ses-{session_num}_task-{task}_acq-01_epo.fif"
            fif_path = os.path.join(eeg_dir, expected_fname)
            if os.path.exists(fif_path):
                subject_sessions.append((subject_id, session_num))

    print(f"Discovered {len(subject_sessions)} subject-session pairs for task '{task}'")
    return subject_sessions


def build_graphs_for_all_subjects(
    main_path: str,
    coords_df,
    subject_sessions: List[Tuple[str, str]],
    task: str,
    window_points: int,
    epoch: int,
    k_neighbors: int,
    output_dir: str,
    corr_type: str,
    save_preprocessing: bool,
    plot_neighbors: bool,
):
    """Build graphs for all (subject_id, session_num) pairs.

    Returns a list of (subject_id, data) tuples.
    """
    samples: List[Tuple[str, object]] = []

    for subject_id, session_num in subject_sessions:
        print("\n" + "-" * 60)
        print(f"Processing subject {subject_id}, session {session_num}")

        data, adjacency, feature_mat, labels, distance_matrix = EEGtoGraph.create_graph(
            coords_df=coords_df,
            main_path=main_path,
            subject_id=subject_id,
            session_num=session_num,
            task=task,
            window_points=window_points,
            epoch=epoch,
            k=k_neighbors,
            output_dir=output_dir,
            corr_type=corr_type,
            save=save_preprocessing,
            plot_neighbors=plot_neighbors,
        )

        samples.append((subject_id, data))

    return samples


def subject_level_split(
    samples: List[Tuple[str, object]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int,
):
    """Split samples into train/val/test at subject level (no subject leakage).

    Returns three lists of Data objects: train_graphs, val_graphs, test_graphs.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    if len(samples) == 0:
        raise ValueError("No samples available to split.")

    subjects = sorted({sid for sid, _ in samples})
    print(f"Total unique subjects: {len(subjects)}")

    # First, split off train subjects
    if len(subjects) < 3:
        raise ValueError("Need at least 3 subjects to perform train/val/test split.")

    # Compute effective val+test fraction
    remaining_frac = val_frac + test_frac

    from sklearn.model_selection import train_test_split  # local import to avoid confusion

    train_subjects, tmp_subjects = train_test_split(
        subjects,
        test_size=remaining_frac,
        random_state=random_state,
        shuffle=True,
    )

    # Now split tmp_subjects into val and test according to their relative fractions
    if len(tmp_subjects) < 2:
        raise ValueError("Not enough subjects left to split into val and test.")

    if remaining_frac == 0:
        raise ValueError("val_frac + test_frac must be > 0.")

    val_ratio_within_remaining = val_frac / remaining_frac

    val_subjects, test_subjects = train_test_split(
        tmp_subjects,
        test_size=(1.0 - val_ratio_within_remaining),
        random_state=random_state,
        shuffle=True,
    )

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects:   {len(val_subjects)}")
    print(f"Test subjects:  {len(test_subjects)}")

    train_graphs = [d for sid, d in samples if sid in train_subjects]
    val_graphs = [d for sid, d in samples if sid in val_subjects]
    test_graphs = [d for sid, d in samples if sid in test_subjects]

    print("\nGraphs per split:")
    print(f"  Train: {len(train_graphs)}")
    print(f"  Val:   {len(val_graphs)}")
    print(f"  Test:  {len(test_graphs)}")

    return train_graphs, val_graphs, test_graphs


def save_datasets(output_dir: str, train_ds, val_ds, test_ds) -> Dict[str, str]:
    """Save train/val/test datasets as .pt files and return their paths."""
    import torch

    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train_dataset.pt")
    val_path = os.path.join(data_dir, "val_dataset.pt")
    test_path = os.path.join(data_dir, "test_dataset.pt")

    torch.save(train_ds, train_path)
    torch.save(val_ds, val_path)
    torch.save(test_ds, test_path)

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="VGAE multi-subject pipeline: preprocessing + subject-level splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--main_path",
        type=str,
        required=True,
        help="Path to the main EEG data directory",
    )
    parser.add_argument(
        "--coordinates_file",
        type=str,
        required=True,
        help="Path to biosemi64.txt file with electrode labels",
    )

    # Preprocessing / graph creation arguments
    parser.add_argument("--task", type=str, default="lg", help="Task name (e.g., lg or rs)")
    parser.add_argument(
        "--window_points",
        type=int,
        default=152,
        help="Number of time points in the window",
    )
    parser.add_argument("--epoch", type=int, default=0, help="Epoch number to process")
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=6,
        help="Number of nearest neighbors for adjacency matrix",
    )
    parser.add_argument(
        "--preprocessing_output_dir",
        type=str,
        default="../output/preprocessing",
        help="Directory to save preprocessing outputs",
    )
    parser.add_argument(
        "--corr_type",
        type=str,
        default="pearson",
        help="Type of correlation for feature matrix",
    )
    parser.add_argument(
        "--save_preprocessing",
        type=bool,
        default=True,
        help="Whether to save preprocessing outputs (matrices, images)",
    )
    parser.add_argument(
        "--plot_neighbors",
        action="store_true",
        help="Plot k-nearest neighbors visualization",
    )

    # Split parameters
    parser.add_argument("--train_frac", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Test fraction")
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for splits")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("VGAE Multi-Subject Pipeline")
    print("=" * 70)
    print(f"Main path:        {args.main_path}")
    print(f"Coordinates file: {args.coordinates_file}")
    print(f"Task:             {args.task}")

    # 1) Load electrode coordinates
    coords_df = load_electrode_coordinates(args.coordinates_file)

    # 2) Discover all subject-session pairs for the task
    subject_sessions = discover_subject_sessions(args.main_path, args.task)
    if len(subject_sessions) == 0:
        print("No subject-session pairs found. Exiting.")
        return 1

    # 3) Build graphs for all subjects/sessions
    samples = build_graphs_for_all_subjects(
        main_path=args.main_path,
        coords_df=coords_df,
        subject_sessions=subject_sessions,
        task=args.task,
        window_points=args.window_points,
        epoch=args.epoch,
        k_neighbors=args.k_neighbors,
        output_dir=args.preprocessing_output_dir,
        corr_type=args.corr_type,
        save_preprocessing=args.save_preprocessing,
        plot_neighbors=args.plot_neighbors,
    )

    # 4) Subject-level train/val/test split (no leakage)
    train_graphs, val_graphs, test_graphs = subject_level_split(
        samples=samples,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.split_seed,
    )

    # 5) Wrap in GraphAutoencoderDataset
    train_dataset = GraphAutoencoderDataset(train_graphs)
    val_dataset = GraphAutoencoderDataset(val_graphs)
    test_dataset = GraphAutoencoderDataset(test_graphs)

    # 6) Save datasets for later training
    paths = save_datasets(args.preprocessing_output_dir, train_dataset, val_dataset, test_dataset)

    print("\n" + "=" * 70)
    print("Datasets saved")
    print("=" * 70)
    print(f"Train dataset: {paths['train']}")
    print(f"Val dataset:   {paths['val']}")
    print(f"Test dataset:  {paths['test']}")

    print("\nSubject-level splits (no leakage) have been created.")
    print("You can now use these .pt files as input to a training script (e.g., train.py)")

    return 0


if __name__ == "__main__":
    exit(main())

