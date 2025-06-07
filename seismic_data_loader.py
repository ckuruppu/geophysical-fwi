#!/usr/bin/env python3
"""
Data loading and preprocessing for seismic waveform inversion.
Handles multiple data families and creates PyTorch datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import random

class SeismicDataset(Dataset):
    """
    PyTorch Dataset for seismic waveform inversion data.
    """
    
    def __init__(self, 
                 data_root: str = "waveform-inversion",
                 families: List[str] = None,
                 transform: Optional[callable] = None,
                 augment: bool = False):
        """
        Initialize the dataset.
        
        Args:
            data_root: Root directory containing the data
            families: List of data families to include
            transform: Optional data transformation function
            augment: Whether to apply data augmentation
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.augment = augment
        
        if families is None:
            families = ["FlatVel_A", "FlatVel_B", "FlatFault_A", "FlatFault_B"]
        
        self.families = families
        self.data_samples = []
        
        # Load all data samples
        self._load_data_samples()
        
    def _load_data_samples(self):
        """Load all data samples from specified families."""
        
        for family in self.families:
            family_path = self.data_root / "train_samples" / family
            
            if not family_path.exists():
                print(f"Warning: Family {family} not found at {family_path}")
                continue
            
            # Handle different family structures
            if family.startswith(("FlatVel", "CurveVel", "Style")):
                # Vel/Style families use data/ and model/ subdirectories
                self._load_vel_style_family(family_path, family)
            else:
                # Fault families use seis_* and vel_* files
                self._load_fault_family(family_path, family)
    
    def _load_vel_style_family(self, family_path: Path, family_name: str):
        """Load Vel or Style family data."""
        data_dir = family_path / "data"
        model_dir = family_path / "model"
        
        if not (data_dir.exists() and model_dir.exists()):
            print(f"Warning: Missing data or model directory for {family_name}")
            return
        
        data_files = sorted(data_dir.glob("*.npy"))
        model_files = sorted(model_dir.glob("*.npy"))
        
        for data_file, model_file in zip(data_files, model_files):
            self.data_samples.append({
                'seismic_path': data_file,
                'velocity_path': model_file,
                'family': family_name,
                'type': 'vel_style'
            })
    
    def _load_fault_family(self, family_path: Path, family_name: str):
        """Load Fault family data."""
        seis_files = sorted(family_path.glob("seis*.npy"))
        vel_files = sorted(family_path.glob("vel*.npy"))
        
        # Match seismic and velocity files
        for seis_file in seis_files:
            # Extract identifier from filename (e.g., "2_1_0" from "seis2_1_0.npy")
            identifier = seis_file.stem.replace("seis", "")
            vel_file = family_path / f"vel{identifier}.npy"
            
            if vel_file.exists():
                self.data_samples.append({
                    'seismic_path': seis_file,
                    'velocity_path': vel_file,
                    'family': family_name,
                    'type': 'fault'
                })
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a data sample.
        
        Returns:
            Tuple of (seismic_data, velocity_model)
        """
        sample_info = self.data_samples[idx]
        
        # Load data
        seismic_data = np.load(sample_info['seismic_path'])
        velocity_model = np.load(sample_info['velocity_path'])
        
        # Handle different data structures
        if sample_info['type'] == 'vel_style':
            # Vel/Style families: (batch, sources, time, receivers) and (batch, layers, H, W)
            # Select random sample from batch
            batch_size = seismic_data.shape[0]
            sample_idx = random.randint(0, batch_size - 1)
            
            seismic_sample = seismic_data[sample_idx]  # (sources, time, receivers)
            velocity_sample = velocity_model[sample_idx, 0]  # (H, W) - take first layer
            
        else:  # fault family
            # Fault families: already single samples
            seismic_sample = seismic_data  # (sources, time, receivers)
            velocity_sample = velocity_model[0]  # (H, W) - take first layer
        
        # Convert to tensors
        seismic_tensor = torch.from_numpy(seismic_sample).float()
        velocity_tensor = torch.from_numpy(velocity_sample).float().unsqueeze(0)  # Add channel dim
        
        # Apply augmentation if enabled
        if self.augment:
            seismic_tensor, velocity_tensor = self._apply_augmentation(seismic_tensor, velocity_tensor)
        
        # Apply transform if provided
        if self.transform:
            seismic_tensor = self.transform(seismic_tensor)
        
        return seismic_tensor, velocity_tensor
    
    def _apply_augmentation(self, seismic: torch.Tensor, velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation.
        """
        # 1. Add noise to seismic data
        if random.random() < 0.5:
            noise_level = random.uniform(0.01, 0.05)
            noise = torch.randn_like(seismic) * noise_level * torch.std(seismic)
            seismic = seismic + noise
        
        # 2. Time shifting (circular shift)
        if random.random() < 0.3:
            shift_samples = random.randint(-10, 10)
            seismic = torch.roll(seismic, shift_samples, dims=1)  # Shift along time axis
        
        # 3. Amplitude scaling
        if random.random() < 0.4:
            scale_factor = random.uniform(0.8, 1.2)
            seismic = seismic * scale_factor
        
        # 4. Receiver dropout (simulate missing receivers)
        if random.random() < 0.2:
            num_dropout = random.randint(1, 5)
            dropout_indices = random.sample(range(seismic.shape[-1]), num_dropout)
            seismic[:, :, dropout_indices] = 0
        
        return seismic, velocity

class SeismicDataModule:
    """
    Data module for handling train/validation splits and data loading.
    """
    
    def __init__(self,
                 data_root: str = "waveform-inversion",
                 batch_size: int = 16,
                 val_split: float = 0.2,
                 num_workers: int = 4,
                 augment_train: bool = True):
        
        self.data_root = data_root
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.augment_train = augment_train
        
        # Define data families for progressive training
        self.family_groups = {
            'simple': ["FlatVel_A", "FlatVel_B"],
            'intermediate': ["FlatVel_A", "FlatVel_B", "Style_A", "Style_B"],
            'complex': ["FlatVel_A", "FlatVel_B", "FlatFault_A", "FlatFault_B"],
            'all': ["FlatVel_A", "FlatVel_B", "FlatFault_A", "FlatFault_B", 
                   "CurveVel_A", "CurveVel_B", "CurveFault_A", "CurveFault_B",
                   "Style_A", "Style_B"]
        }
        
    def get_dataloaders(self, complexity: str = 'simple') -> Tuple[DataLoader, DataLoader]:
        """
        Get train and validation dataloaders for specified complexity level.
        
        Args:
            complexity: One of 'simple', 'intermediate', 'complex', 'all'
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        families = self.family_groups.get(complexity, self.family_groups['simple'])
        
        # Create full dataset
        full_dataset = SeismicDataset(
            data_root=self.data_root,
            families=families,
            augment=False  # We'll handle augmentation separately
        )
        
        # Split into train and validation
        dataset_size = len(full_dataset)
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Apply augmentation to training set
        if self.augment_train:
            train_dataset.dataset.augment = True
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_test_loader(self) -> DataLoader:
        """
        Get test data loader for final evaluation.
        """
        test_dir = Path(self.data_root) / "test"
        test_files = list(test_dir.glob("*.npy"))
        
        # Create simple test dataset
        test_data = []
        for file_path in test_files[:100]:  # Limit to first 100 for testing
            data = np.load(file_path)
            test_data.append(torch.from_numpy(data).float())
        
        test_dataset = torch.utils.data.TensorDataset(*test_data)
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

def collate_fn(batch):
    """
    Custom collate function for handling variable-sized data.
    """
    seismic_data = []
    velocity_models = []
    
    for seismic, velocity in batch:
        seismic_data.append(seismic)
        velocity_models.append(velocity)
    
    # Stack into batches
    seismic_batch = torch.stack(seismic_data, dim=0)
    velocity_batch = torch.stack(velocity_models, dim=0)
    
    return seismic_batch, velocity_batch

def get_data_statistics(data_root: str = "waveform-inversion") -> Dict[str, float]:
    """
    Compute dataset statistics for normalization.
    """
    dataset = SeismicDataset(data_root=data_root, families=["FlatVel_A"])
    
    seismic_values = []
    velocity_values = []
    
    # Sample a subset for statistics
    for i in range(min(100, len(dataset))):
        seismic, velocity = dataset[i]
        seismic_values.append(seismic.flatten())
        velocity_values.append(velocity.flatten())
    
    seismic_all = torch.cat(seismic_values)
    velocity_all = torch.cat(velocity_values)
    
    stats = {
        'seismic_mean': torch.mean(seismic_all).item(),
        'seismic_std': torch.std(seismic_all).item(),
        'velocity_mean': torch.mean(velocity_all).item(),
        'velocity_std': torch.std(velocity_all).item(),
        'velocity_min': torch.min(velocity_all).item(),
        'velocity_max': torch.max(velocity_all).item()
    }
    
    return stats
