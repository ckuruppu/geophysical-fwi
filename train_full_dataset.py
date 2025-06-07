#!/usr/bin/env python3
"""
Comprehensive training script for the full 13.22GB seismic waveform inversion dataset.
Optimized for memory efficiency, parallelism, and large-scale training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
import gc
import psutil
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from seismic_model_efficient import EfficientSeismicInversionNet
from physics_losses_efficient import EfficientPhysicsInformedLoss, EfficientMetrics

class MemoryEfficientSeismicDataset(Dataset):
    """
    Memory-efficient dataset that loads data on-demand using memory mapping.
    Handles the full 13.22GB dataset without loading everything into memory.
    """
    
    def __init__(self, data_root: str, complexity_levels: List[str] = None, 
                 transform=None, max_samples_per_scenario: int = None):
        
        self.data_root = Path(data_root)
        self.transform = transform
        self.max_samples_per_scenario = max_samples_per_scenario
        
        # Complexity mapping
        self.complexity_mapping = {
            'simple': ['FlatVel_A', 'FlatVel_B'],
            'intermediate': ['Style_A', 'Style_B', 'CurveVel_A', 'CurveVel_B'],
            'complex': ['FlatFault_A', 'FlatFault_B', 'CurveFault_A', 'CurveFault_B']
        }
        
        # Determine which scenarios to include
        if complexity_levels is None:
            complexity_levels = ['simple', 'intermediate', 'complex']
        
        scenarios_to_include = []
        for level in complexity_levels:
            scenarios_to_include.extend(self.complexity_mapping.get(level, []))
        
        # Build file index
        self.file_index = []
        self._build_file_index(scenarios_to_include)
        
        print(f"📊 Dataset initialized with {len(self.file_index)} samples")
    
    def _build_file_index(self, scenarios: List[str]):
        """Build an index of all data files for efficient access."""
        print("🔍 Building file index...")
        
        for scenario in scenarios:
            scenario_path = self.data_root / scenario
            if not scenario_path.exists():
                print(f"⚠️  Scenario {scenario} not found, skipping...")
                continue
            
            # Check structure
            data_dir = scenario_path / "data"
            model_dir = scenario_path / "model"
            
            if data_dir.exists() and model_dir.exists():
                # Structure: scenario/data/*.npy, scenario/model/*.npy
                data_files = sorted(list(data_dir.glob("*.npy")))
                model_files = sorted(list(model_dir.glob("*.npy")))
                
                for data_file, model_file in zip(data_files, model_files):
                    # Check if files contain batched data
                    try:
                        with open(data_file, 'rb') as f:
                            np.lib.format.read_magic(f)
                            shape, _, _ = np.lib.format.read_array_header_1_0(f)
                            
                        if len(shape) == 4:  # Batched data (batch, sources, time, receivers)
                            batch_size = shape[0]
                            # Add each sample in the batch
                            for i in range(min(batch_size, self.max_samples_per_scenario or batch_size)):
                                self.file_index.append({
                                    'data_file': data_file,
                                    'model_file': model_file,
                                    'sample_idx': i,
                                    'scenario': scenario
                                })
                        else:
                            # Single sample file
                            self.file_index.append({
                                'data_file': data_file,
                                'model_file': model_file,
                                'sample_idx': 0,
                                'scenario': scenario
                            })
                    except Exception as e:
                        print(f"⚠️  Error reading {data_file}: {e}")
                        continue
            
            else:
                # Structure: scenario/*.npy files directly
                seis_files = sorted(list(scenario_path.glob("seis*.npy")))
                vel_files = sorted(list(scenario_path.glob("vel*.npy")))
                
                for seis_file, vel_file in zip(seis_files, vel_files):
                    try:
                        with open(seis_file, 'rb') as f:
                            np.lib.format.read_magic(f)
                            shape, _, _ = np.lib.format.read_array_header_1_0(f)
                        
                        if len(shape) == 4:  # Batched data
                            batch_size = shape[0]
                            for i in range(min(batch_size, self.max_samples_per_scenario or batch_size)):
                                self.file_index.append({
                                    'data_file': seis_file,
                                    'model_file': vel_file,
                                    'sample_idx': i,
                                    'scenario': scenario
                                })
                        else:
                            self.file_index.append({
                                'data_file': seis_file,
                                'model_file': vel_file,
                                'sample_idx': 0,
                                'scenario': scenario
                            })
                    except Exception as e:
                        print(f"⚠️  Error reading {seis_file}: {e}")
                        continue
        
        print(f"✅ File index built: {len(self.file_index)} samples")
    
    def __len__(self):
        return len(self.file_index)
    
    def __getitem__(self, idx):
        """Load a single sample using memory mapping for efficiency."""
        if idx >= len(self.file_index):
            raise IndexError(f"Index {idx} out of range")
        
        file_info = self.file_index[idx]
        
        try:
            # Load data using memory mapping (doesn't load into memory until accessed)
            seismic_data = np.load(file_info['data_file'], mmap_mode='r')
            velocity_data = np.load(file_info['model_file'], mmap_mode='r')
            
            # Extract the specific sample
            sample_idx = file_info['sample_idx']
            
            if len(seismic_data.shape) == 4:  # Batched data
                seismic_sample = np.array(seismic_data[sample_idx])  # Copy to memory
                velocity_sample = np.array(velocity_data[sample_idx])
            else:
                seismic_sample = np.array(seismic_data)
                velocity_sample = np.array(velocity_data)
            
            # Ensure correct shapes
            if len(seismic_sample.shape) == 3:  # (sources, time, receivers)
                pass  # Already correct
            elif len(seismic_sample.shape) == 2:  # (time, receivers) - single source
                seismic_sample = seismic_sample[np.newaxis, :, :]  # Add source dimension
            
            if len(velocity_sample.shape) == 2:  # (height, width)
                velocity_sample = velocity_sample[np.newaxis, :, :]  # Add channel dimension
            elif len(velocity_sample.shape) == 3 and velocity_sample.shape[0] > 1:
                velocity_sample = velocity_sample[0:1, :, :]  # Take first channel only
            
            # Convert to tensors
            seismic_tensor = torch.from_numpy(seismic_sample).float()
            velocity_tensor = torch.from_numpy(velocity_sample).float()
            
            # Apply transforms if any
            if self.transform:
                seismic_tensor, velocity_tensor = self.transform(seismic_tensor, velocity_tensor)
            
            return seismic_tensor, velocity_tensor
            
        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {e}")
            # Return dummy data to avoid crashing
            return torch.zeros(5, 1000, 70), torch.zeros(1, 70, 70)

class FullDatasetTrainer:
    """
    Trainer optimized for the full 13.22GB dataset with memory management.
    """
    
    def __init__(self, 
                 model: EfficientSeismicInversionNet,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 save_dir: str = 'full_dataset_checkpoints'):
        
        print("🔧 Initializing full dataset trainer...")
        
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = EfficientPhysicsInformedLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        # Memory monitor
        self.memory_monitor = psutil.Process()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'memory_usage': [],
            'epoch_times': []
        }
        
        print("✅ Full dataset trainer initialized!")
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        return self.memory_monitor.memory_info().rss / 1024 / 1024
    
    def train_progressive(self, 
                         complexity_schedule: List[Tuple[str, int]] = None,
                         max_samples_per_scenario: int = 500) -> Dict:
        """
        Progressive training on the full dataset with memory optimization.
        """
        if complexity_schedule is None:
            complexity_schedule = [
                (['simple'], 10),
                (['simple', 'intermediate'], 8),
                (['simple', 'intermediate', 'complex'], 5)
            ]
        
        print(f"🚀 Starting full dataset progressive training")
        print(f"📋 Schedule: {complexity_schedule}")
        print(f"🔢 Max samples per scenario: {max_samples_per_scenario}")
        
        total_start_time = time.time()
        total_epochs = 0
        
        for phase_idx, (complexity_levels, num_epochs) in enumerate(complexity_schedule):
            print(f"\n{'='*60}")
            print(f"🎯 Phase {phase_idx + 1}/{len(complexity_schedule)}: {complexity_levels} ({num_epochs} epochs)")
            print(f"{'='*60}")
            
            # Create dataset for this phase
            dataset = MemoryEfficientSeismicDataset(
                data_root="waveform-inversion/train_samples",
                complexity_levels=complexity_levels,
                max_samples_per_scenario=max_samples_per_scenario
            )
            
            # Split into train/val
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=True if self.num_workers > 0 else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=True if self.num_workers > 0 else False
            )
            
            print(f"📊 Phase data: {len(train_dataset)} train, {len(val_dataset)} val samples")
            print(f"📊 Batches: {len(train_loader)} train, {len(val_loader)} val")
            
            # Train for this phase
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                
                # Training
                train_loss, train_metrics = self._train_epoch(train_loader, total_epochs)
                
                # Validation
                val_loss, val_metrics = self._validate_epoch(val_loader, total_epochs)
                
                # Update scheduler
                self.scheduler.step()
                
                # Memory management
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Record metrics
                epoch_time = time.time() - epoch_start_time
                memory_mb = self.get_memory_usage()
                
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_metrics'].append(train_metrics)
                self.history['val_metrics'].append(val_metrics)
                self.history['memory_usage'].append(memory_mb)
                self.history['epoch_times'].append(epoch_time)
                
                total_epochs += 1
                
                print(f"  📊 Epoch {total_epochs}:")
                print(f"    Train Loss: {train_loss:.6f}")
                print(f"    Val Loss: {val_loss:.6f}")
                print(f"    Train RMSE: {train_metrics.get('rmse', 0):.1f} m/s")
                print(f"    Val RMSE: {val_metrics.get('rmse', 0):.1f} m/s")
                print(f"    Memory: {memory_mb:.1f} MB")
                print(f"    Time: {epoch_time:.2f}s")
                
                # Save checkpoint every 5 epochs
                if total_epochs % 5 == 0:
                    checkpoint_path = self.save_dir / f'checkpoint_epoch_{total_epochs}.pth'
                    self._save_checkpoint(checkpoint_path, total_epochs)
                    print(f"    💾 Checkpoint saved")
        
        total_time = time.time() - total_start_time
        
        print(f"\n🎉 Full dataset training completed!")
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"📊 Total epochs: {total_epochs}")
        
        # Save final model
        final_path = self.save_dir / 'final_full_dataset_model.pth'
        self._save_checkpoint(final_path, total_epochs)
        
        # Save history
        history_path = self.save_dir / 'full_dataset_training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy types for JSON serialization
            history_json = {}
            for key, value in self.history.items():
                if isinstance(value, list) and value:
                    if isinstance(value[0], dict):
                        history_json[key] = [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                            for k, v in item.items()} for item in value]
                    else:
                        history_json[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
                else:
                    history_json[key] = value
            json.dump(history_json, f, indent=2)
        
        print(f"💾 Training history saved to {history_path}")
        
        return self.history

    def _train_epoch(self, train_loader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch with memory monitoring."""
        self.model.train()
        total_loss = 0.0
        all_metrics = []

        print(f"    🏃 Training epoch {epoch + 1}...")

        for batch_idx, (seismic_data, velocity_target) in enumerate(train_loader):
            # Move data to device
            seismic_data = seismic_data.to(self.device, non_blocking=True)
            velocity_target = velocity_target.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()
            velocity_pred = self.model(seismic_data)

            # Compute loss
            losses = self.criterion(velocity_pred, velocity_target, seismic_data)
            loss = losses['total']

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # Compute metrics
            with torch.no_grad():
                batch_metrics = EfficientMetrics.compute_metrics(velocity_pred, velocity_target)
                all_metrics.append(batch_metrics)

            # Progress logging
            if batch_idx % max(1, len(train_loader) // 5) == 0:
                memory_mb = self.get_memory_usage()
                print(f"      Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f} | "
                      f"RMSE: {batch_metrics.get('rmse', 0):.1f} | "
                      f"Memory: {memory_mb:.1f}MB")

            # Memory cleanup every 10 batches
            if batch_idx % 10 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_loss = total_loss / len(train_loader)

        return avg_loss, avg_metrics

    def _validate_epoch(self, val_loader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []

        print(f"    🔍 Validating epoch {epoch + 1}...")

        with torch.no_grad():
            for batch_idx, (seismic_data, velocity_target) in enumerate(val_loader):
                seismic_data = seismic_data.to(self.device, non_blocking=True)
                velocity_target = velocity_target.to(self.device, non_blocking=True)

                velocity_pred = self.model(seismic_data)
                losses = self.criterion(velocity_pred, velocity_target, seismic_data)
                loss = losses['total']

                total_loss += loss.item()

                # Compute metrics
                batch_metrics = EfficientMetrics.compute_metrics(velocity_pred, velocity_target)
                all_metrics.append(batch_metrics)

        # Average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_loss = total_loss / len(val_loader)

        return avg_loss, avg_metrics

    def _average_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Average metrics across batches."""
        if not metrics_list:
            return {}

        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key not in ['predicted_stats', 'target_stats']:
                avg_metrics[key] = np.mean([m[key] for m in metrics_list])

        return avg_metrics

    def _save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, path)

def main():
    """Main training function for the full dataset."""
    print("🚀 FULL DATASET SEISMIC WAVEFORM INVERSION TRAINING")
    print("="*60)

    try:
        # System info
        print(f"💻 System Info:")
        print(f"  CPUs: {psutil.cpu_count()}")
        print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"  Available: {psutil.virtual_memory().available / 1024**3:.1f} GB")

        # Create efficient model (optimized for large dataset)
        print("🏗️  Creating efficient model...")
        model = EfficientSeismicInversionNet(
            temporal_hidden_size=32,  # Smaller for memory efficiency
            spatial_output_channels=64,
            num_sources=5,
            output_size=(70, 70)
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")

        # Create trainer with optimized settings
        print("🎯 Creating full dataset trainer...")
        trainer = FullDatasetTrainer(
            model=model,
            batch_size=4,  # Small batch size for memory efficiency
            num_workers=4,  # Parallel data loading
            save_dir='full_dataset_checkpoints'
        )

        # Start training on full dataset
        print("🏃 Starting full dataset training...")
        history = trainer.train_progressive(
            complexity_schedule=[
                (['simple'], 5),                           # Start with simple models
                (['simple', 'intermediate'], 5),           # Add intermediate complexity
                (['simple', 'intermediate', 'complex'], 5) # Full complexity
            ],
            max_samples_per_scenario=1000  # Use 1000 samples per scenario for faster training
        )

        print("🎉 Full dataset training completed successfully!")
        if history['train_loss']:
            print(f"📊 Final training loss: {history['train_loss'][-1]:.6f}")
            print(f"📊 Final validation loss: {history['val_loss'][-1]:.6f}")
            print(f"⏱️  Average epoch time: {np.mean(history['epoch_times']):.2f}s")
            print(f"💾 Peak memory usage: {max(history['memory_usage']):.1f} MB")

    except Exception as e:
        print(f"❌ Full dataset training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
