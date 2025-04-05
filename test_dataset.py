"""Test script for BioreactorDataset."""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dataset import BioreactorDataset

def test_bioreactor_dataset():
    # Example column definitions
    Z_columns = ["feed_start", "feed_end", "Glc_feed_rate", "Glc_0", "VCD_0"]
    F_columns = ["Glc"]
    X_columns = ["VCD", "Glc", "Lac", "Titer"]
    
    # Create dataset instances
    train_dataset = BioreactorDataset(
        owu_file="owu",
        doe_file="owu_doe",
        train_path="dataset/interpolation/train",
        test_path="dataset/interpolation/test",
        t_steps=15,
        init_volume=1000,
        X_columns=X_columns,
        F_columns=F_columns,
        Z_columns=Z_columns,
        mode="train",
        val_split=0.2
    )
    
    val_dataset = BioreactorDataset(
        owu_file="owu",
        doe_file="owu_doe",
        train_path="dataset/interpolation/train",
        test_path="dataset/interpolation/test",
        t_steps=15,
        init_volume=1000,
        X_columns=X_columns,
        F_columns=F_columns,
        Z_columns=Z_columns,
        mode="val",
        val_split=0.2
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Basic tests
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Get a sample batch
    features, targets = next(iter(train_loader))
    print("\nFeatures shape:", features.shape)
    print("Targets shape:", targets.shape)

    # Verify features contain all components
    total_features = len(X_columns) + len(F_columns) + 1 + len(Z_columns)  # +1 for volume
    assert features.shape[-1] == total_features, "Incorrect number of feature channels"
    assert targets.shape[-1] == len(X_columns), "Incorrect number of target channels"

    # Plot example sequence
    def plot_example_sequence(features, targets, idx=0):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot features
        time_steps = range(features.shape[1])
        for i in range(features.shape[2]):
            axes[0].plot(time_steps, features[idx, :, i].numpy(), 
                        label=f'Feature {i}')
        axes[0].set_title('Features')
        axes[0].legend()
        
        # Plot targets (next step predictions)
        for i in range(targets.shape[2]):
            axes[1].plot(time_steps, targets[idx, :, i].numpy(), 
                        label=f'Target {i}')
        axes[1].set_title('Targets (Next Step Values)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

    # Plot an example sequence
    plot_example_sequence(features, targets)

    # Test autoregressive property
    # Get column names for better readability
    state_vars = X_columns  # Assuming X_columns is defined above
    
    print("\nTimestep t (features current state):")
    features_t = features[0, 0, :len(state_vars)]
    for i, (var, val) in enumerate(zip(state_vars, features_t)):
        print(f"{var:10}: {val:.6f}")
    
    print("\nTimestep t+1 (target/prediction):")
    targets_t = targets[0, 0, :]
    for i, (var, val) in enumerate(zip(state_vars, targets_t)):
        print(f"{var:10}: {val:.6f}")
    
    print("\nTimestep t+1 (actual next features):")
    features_t1 = features[0, 1, :len(state_vars)]
    for i, (var, val) in enumerate(zip(state_vars, features_t1)):
        print(f"{var:10}: {val:.6f}")
    
    print("Features (t) shape:", features_t.shape)
    print("Targets (t) shape:", targets_t.shape)
    print("Features (t+1) shape:", features_t1.shape)
    
    # Verify that targets at t match features at t+1
    assert torch.allclose(targets_t, features_t1), "Autoregressive property not maintained"
    print("Autoregressive property verified!")

if __name__ == "__main__":
    test_bioreactor_dataset()

"""
Training set size: 40
Validation set size: 10

Features shape: torch.Size([32, 15, 11])
Targets shape: torch.Size([32, 15, 4])

Timestep t (features current state):
VCD       : 0.550000
Glc       : 25.000000
Lac       : 0.000000
Titer     : 0.000000

Timestep t+1 (target/prediction):
VCD       : 1.691673
Glc       : 24.020660
Lac       : 1.472006
Titer     : 0.094907

Timestep t+1 (actual next features):
VCD       : 1.691673
Glc       : 24.020660
Lac       : 1.472006
Titer     : 0.094907
Features (t) shape: torch.Size([4])
Targets (t) shape: torch.Size([4])
Features (t+1) shape: torch.Size([4])
Autoregressive property verified!
"""