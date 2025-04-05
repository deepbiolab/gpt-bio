"""Bioreactor dataset for autoregressive models.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import os
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_utils import process_owu_data, process_doe_data

class BioreactorDataset(Dataset):
    def __init__(
        self,
        owu_file: str,
        doe_file: str,
        train_path: str = "dataset/interpolation/train",
        test_path: str = "dataset/interpolation/test",
        t_steps: int = 15,
        init_volume: float = 1000,
        Z_columns: List[str] = [],
        X_columns: List[str] = [],
        F_columns: List[str] = [],
        mode: str = "train",
        val_split: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        self.mode = mode
        self.root_path = train_path if mode in ["train", "val"] else test_path
        
        # Process column names
        self.Z_columns = Z_columns
        self.X_columns = [f"X:{col}" for col in X_columns]
        self.F_columns = [f"F:{col}" for col in F_columns]
        
        # Load and process data
        doe_data = self._read_doe(doe_file)
        owu_data = self._read_owu(owu_file)
        
        # Process data into tensors
        self.X, self.F = process_owu_data(owu_data, t_steps, self.X_columns, self.F_columns)
        self.Z = process_doe_data(doe_data, self.Z_columns)
        
        # Calculate volume
        self.V = (init_volume + (self.F.sum(axis=-1, keepdims=True)).cumsum(axis=1)) / 1000
        
        # Convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)  # (B, T, C_x)
        self.F = torch.tensor(self.F, dtype=torch.float32)  # (B, T, C_f)
        self.V = torch.tensor(self.V, dtype=torch.float32)  # (B, T, 1)
        self.Z = torch.tensor(self.Z, dtype=torch.float32)  # (B, 1, C_z)
        self.Z = self.Z.expand(-1, t_steps, -1)  # (B, T, C_z)
        
        # Concatenate features
        self.features = torch.cat([self.X, self.F, self.V, self.Z], dim=-1)  # (B, T, C_total)
        
        # Create targets (shifted by one timestep)
        self.targets = torch.roll(self.X, shifts=-1, dims=1)
        self.targets[:, -1] = self.targets[:, -2]  # Handle last timestep
        
        # Split data if needed
        if mode in ["train", "val"]:
            self._split_data(val_split, random_seed)
            
    def _read_owu(self, file: str) -> pd.DataFrame:
        """Read and process the OWU data file."""
        data = pd.read_csv(f"{self.root_path}/{file}.csv")
        owu_df = data.copy()
        num_runs = len(pd.read_csv(f"{self.root_path}/{file}_doe.csv"))

        if "run" not in owu_df.columns:
            owu_df.index = pd.MultiIndex.from_product(
                [list(range(num_runs)), list(range(self.t_steps))],
                names=["run", "time"],
            )
        else:
            owu_df.set_index(["run", "time"], inplace=True)
        owu_df = owu_df[self.X_columns + self.F_columns]
        return owu_df

    def _read_doe(self, file: str) -> pd.DataFrame:
        """Read the Design of Experiments data file."""
        data = pd.read_csv(
            f"{self.root_path}/{file}.csv",
            usecols=self.Z_columns,
        )
        return data.copy()
    
    def _split_data(self, val_split: float, random_seed: int) -> None:
        np.random.seed(random_seed)
        total_size = len(self.features)
        val_size = int(val_split * total_size)
        indices = np.random.permutation(total_size)
        
        if self.mode == "train":
            self.indices = indices[val_size:]
        else:  # val
            self.indices = indices[:val_size]
            
        self.features = self.features[self.indices]
        self.targets = self.targets[self.indices]
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]