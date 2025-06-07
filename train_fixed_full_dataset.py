#!/usr/bin/env python3
"""
Comprehensive training script for the FIXED model on the full 13.22GB dataset.
Uses proper output constraints, normalized losses, and memory-efficient data loading.
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

from seismic_model_fixed import FixedSeismicInversionNet, FixedPhysicsInformedLoss
from physics_losses_efficient import EfficientMetrics

class MemoryEfficientSeismicDataset(Dataset):
    """
    Memory-efficient dataset for the full 13.22GB seismic dataset.
    Uses memory mapping and on-demand loading.
    """
    
    def __init__(self, data_root: str, complexity_levels: List[str] = None, 
                 max_samples_per_scenario: int = None, validation_split: float = 0.2):
        
        self.data_root = Path(data_root)
        self.max_samples_per_scenario = max_samples_per_scenario
        self.validation_split = validation_split
        
        # Complexity mapping based on geological structure
        self.complexity_mapping = {
            'simple': ['FlatVel_A', 'FlatVel_B'],
            'intermediate': ['Style_A', 'Style_B', 'CurveVel_A', 'CurveVel_B'],
            'complex': ['FlatFault_A', 'FlatFault_B', 'CurveFault_A', 'CurveFault_B']
        }
        
        # Determine scenarios to include
        if complexity_levels is None:
            complexity_levels = ['simple', 'intermediate', 'complex']
        
        scenarios_to_include = []
        for level in complexity_levels:
            scenarios_to_include.extend(self.complexity_mapping.get(level, []))
        
        # Build file index
        self.file_index = []
        self._build_file_index(scenarios_to_include)
        
        print(f"📊 Dataset initialized with {len(self.file_index)} samples")
        print(f"📊 Scenarios included: {scenarios_to_include}")
    
    def _build_file_index(self, scenarios: List[str]):
        """Build an index of all data files for efficient access."""
        print("🔍 Building comprehensive file index...")
        
        total_samples = 0
        scenario_counts = {}
        
        for scenario in scenarios:
            scenario_path = self.data_root / scenario
            if not scenario_path.exists():
                print(f"⚠️  Scenario {scenario} not found, skipping...")
                continue
            
            scenario_samples = 0
            
            # Check data structure
            data_dir = scenario_path / "data"
            model_dir = scenario_path / "model"
            
            if data_dir.exists() and model_dir.exists():
                # Structure: scenario/data/*.npy, scenario/model/*.npy
                data_files = sorted(list(data_dir.glob("*.npy")))
                model_files = sorted(list(model_dir.glob("*.npy")))
                
                print(f"  📁 {scenario}: {len(data_files)} data files, {len(model_files)} model files")
                
                for data_file, model_file in zip(data_files, model_files):
                    try:
                        # Read file header to get shape
                        with open(data_file, 'rb') as f:
                            np.lib.format.read_magic(f)
                            shape, _, _ = np.lib.format.read_array_header_1_0(f)
                        
                        if len(shape) == 4:  # Batched data
                            batch_size = shape[0]
                            samples_to_add = min(batch_size, self.max_samples_per_scenario or batch_size)
                            
                            for i in range(samples_to_add):
                                self.file_index.append({
                                    'data_file': data_file,
                                    'model_file': model_file,
                                    'sample_idx': i,
                                    'scenario': scenario,
                                    'complexity': self._get_complexity(scenario)
                                })
                                scenario_samples += 1
                        else:
                            # Single sample file
                            self.file_index.append({
                                'data_file': data_file,
                                'model_file': model_file,
                                'sample_idx': 0,
                                'scenario': scenario,
                                'complexity': self._get_complexity(scenario)
                            })
                            scenario_samples += 1
                            
                        if self.max_samples_per_scenario and scenario_samples >= self.max_samples_per_scenario:
                            break
                            
                    except Exception as e:
                        print(f"⚠️  Error reading {data_file}: {e}")
                        continue
            
            else:
                # Structure: scenario/*.npy files directly
                seis_files = sorted(list(scenario_path.glob("seis*.npy")))
                vel_files = sorted(list(scenario_path.glob("vel*.npy")))
                
                print(f"  📁 {scenario}: {len(seis_files)} seismic files, {len(vel_files)} velocity files")
                
                for seis_file, vel_file in zip(seis_files, vel_files):
                    try:
                        with open(seis_file, 'rb') as f:
                            np.lib.format.read_magic(f)
                            shape, _, _ = np.lib.format.read_array_header_1_0(f)
                        
                        if len(shape) == 4:  # Batched data
                            batch_size = shape[0]
                            samples_to_add = min(batch_size, self.max_samples_per_scenario or batch_size)
                            
                            for i in range(samples_to_add):
                                self.file_index.append({
                                    'data_file': seis_file,
                                    'model_file': vel_file,
                                    'sample_idx': i,
                                    'scenario': scenario,
                                    'complexity': self._get_complexity(scenario)
                                })
                                scenario_samples += 1
                        else:
                            self.file_index.append({
                                'data_file': seis_file,
                                'model_file': vel_file,
                                'sample_idx': 0,
                                'scenario': scenario,
                                'complexity': self._get_complexity(scenario)
                            })
                            scenario_samples += 1
                            
                        if self.max_samples_per_scenario and scenario_samples >= self.max_samples_per_scenario:
                            break
                            
                    except Exception as e:
                        print(f"⚠️  Error reading {seis_file}: {e}")
                        continue
            
            scenario_counts[scenario] = scenario_samples
            total_samples += scenario_samples
            print(f"    ✅ {scenario}: {scenario_samples} samples")
        
        print(f"\n📊 Total samples indexed: {total_samples}")
        print(f"📊 Scenario breakdown: {scenario_counts}")
    
    def _get_complexity(self, scenario: str) -> str:
        """Get complexity level for a scenario."""
        for complexity, scenarios in self.complexity_mapping.items():
            if scenario in scenarios:
                return complexity
        return 'unknown'
    
    def __len__(self):
        return len(self.file_index)
    
    def __getitem__(self, idx):
        """Load a single sample using memory mapping."""
        if idx >= len(self.file_index):
            raise IndexError(f"Index {idx} out of range")
        
        file_info = self.file_index[idx]
        
        try:
            # Load data using memory mapping
            seismic_data = np.load(file_info['data_file'], mmap_mode='r')
            velocity_data = np.load(file_info['model_file'], mmap_mode='r')
            
            # Extract specific sample
            sample_idx = file_info['sample_idx']
            
            if len(seismic_data.shape) == 4:  # Batched data
                seismic_sample = np.array(seismic_data[sample_idx])
                velocity_sample = np.array(velocity_data[sample_idx])
            else:
                seismic_sample = np.array(seismic_data)
                velocity_sample = np.array(velocity_data)
            
            # Ensure correct shapes
            if len(seismic_sample.shape) == 3:  # (sources, time, receivers)
                pass  # Already correct
            elif len(seismic_sample.shape) == 2:  # (time, receivers)
                seismic_sample = seismic_sample[np.newaxis, :, :]
            
            if len(velocity_sample.shape) == 2:  # (height, width)
                velocity_sample = velocity_sample[np.newaxis, :, :]
            elif len(velocity_sample.shape) == 3 and velocity_sample.shape[0] > 1:
                velocity_sample = velocity_sample[0:1, :, :]
            
            # Convert to tensors
            seismic_tensor = torch.from_numpy(seismic_sample).float()
            velocity_tensor = torch.from_numpy(velocity_sample).float()
            
            # Ensure velocity is in realistic range
            velocity_tensor = torch.clamp(velocity_tensor, 1000, 6000)
            
            return seismic_tensor, velocity_tensor
            
        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {e}")
            # Return dummy data
            return torch.zeros(5, 1000, 70), torch.ones(1, 70, 70) * 3000

class FullDatasetTrainer:
    """
    Trainer for the full dataset with the fixed model architecture.
    """
    
    def __init__(self, 
                 model: FixedSeismicInversionNet,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 save_dir: str = 'fixed_full_dataset_checkpoints'):
        
        print("🔧 Initializing full dataset trainer with FIXED model...")
        
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.model.to(self.device)
        
        # Fixed loss function with proper scaling
        self.criterion = FixedPhysicsInformedLoss(
            mse_weight=1.0,
            smoothness_weight=0.1,
            gradient_weight=0.1,
            geological_weight=0.05,
            velocity_range=(1500.0, 4500.0)
        )
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,  # Higher learning rate since losses are now normalized
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
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
            'loss_components': [],
            'memory_usage': [],
            'epoch_times': []
        }
        
        print("✅ Fixed model trainer initialized!")
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        return self.memory_monitor.memory_info().rss / 1024 / 1024
    
    def train_progressive(self, 
                         complexity_schedule: List[Tuple[List[str], int]] = None,
                         max_samples_per_scenario: int = 2000) -> Dict:
        """
        Progressive training on the full dataset with the fixed model.
        """
        if complexity_schedule is None:
            complexity_schedule = [
                (['simple'], 15),                           # Start with simple models
                (['simple', 'intermediate'], 15),           # Add intermediate complexity  
                (['simple', 'intermediate', 'complex'], 20) # Full complexity - more epochs
            ]
        
        print(f"🚀 Starting FIXED MODEL training on full dataset")
        print(f"📋 Schedule: {complexity_schedule}")
        print(f"🔢 Max samples per scenario: {max_samples_per_scenario}")
        
        total_start_time = time.time()
        total_epochs = 0
        best_val_loss = float('inf')
        
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
            
            if len(dataset) == 0:
                print("⚠️  No data found for this phase, skipping...")
                continue
            
            # Split into train/val
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # Reproducible split
            )
            
            # Create data loaders with optimized settings
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=True if self.num_workers > 0 else False,
                drop_last=True  # Ensure consistent batch sizes
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
                train_loss, train_metrics, loss_components = self._train_epoch(train_loader, total_epochs)
                
                # Validation
                val_loss, val_metrics = self._validate_epoch(val_loader, total_epochs)
                
                # Update scheduler
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
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
                self.history['loss_components'].append(loss_components)
                self.history['memory_usage'].append(memory_mb)
                self.history['epoch_times'].append(epoch_time)
                
                total_epochs += 1
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = self.save_dir / 'best_fixed_model.pth'
                    self._save_checkpoint(best_model_path, total_epochs, f'phase_{phase_idx+1}_best')
                    print(f"    🏆 New best model! Val loss: {val_loss:.6f}")
                
                print(f"  📊 Epoch {total_epochs} ({complexity_levels}):")
                print(f"    Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                print(f"    Train RMSE: {train_metrics.get('rmse', 0):.1f} m/s | Val RMSE: {val_metrics.get('rmse', 0):.1f} m/s")
                print(f"    LR: {current_lr:.2e} | Memory: {memory_mb:.1f}MB | Time: {epoch_time:.2f}s")
                
                # Print loss components
                if loss_components:
                    print(f"    Loss components: MSE={loss_components.get('mse', 0):.4f}, "
                          f"Smooth={loss_components.get('smoothness', 0):.4f}, "
                          f"Grad={loss_components.get('gradient', 0):.4f}")
                
                # Save checkpoint every 10 epochs
                if total_epochs % 10 == 0:
                    checkpoint_path = self.save_dir / f'checkpoint_epoch_{total_epochs}.pth'
                    self._save_checkpoint(checkpoint_path, total_epochs, f'phase_{phase_idx+1}')
                    print(f"    💾 Checkpoint saved")
        
        total_time = time.time() - total_start_time
        
        print(f"\n🎉 FIXED MODEL training completed!")
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"📊 Total epochs: {total_epochs}")
        print(f"🏆 Best validation loss: {best_val_loss:.6f}")
        
        # Save final model
        final_path = self.save_dir / 'final_fixed_model.pth'
        self._save_checkpoint(final_path, total_epochs, 'final')
        
        # Save training history
        history_path = self.save_dir / 'fixed_training_history.json'
        self._save_history(history_path)
        
        print(f"💾 Final model and history saved!")
        
        return self.history

    def _train_epoch(self, train_loader, epoch: int) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Train for one epoch with detailed loss component tracking."""
        self.model.train()
        total_loss = 0.0
        total_components = {}
        all_metrics = []

        print(f"    🏃 Training epoch {epoch + 1}...")

        for batch_idx, (seismic_data, velocity_target) in enumerate(train_loader):
            # Move data to device
            seismic_data = seismic_data.to(self.device, non_blocking=True)
            velocity_target = velocity_target.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()
            velocity_pred = self.model(seismic_data)

            # Compute loss with components
            losses = self.criterion(velocity_pred, velocity_target, seismic_data)
            loss = losses['total']

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # Accumulate loss components
            for key, value in losses.items():
                if key not in total_components:
                    total_components[key] = 0.0
                total_components[key] += value.item()

            # Compute metrics
            with torch.no_grad():
                batch_metrics = EfficientMetrics.compute_metrics(velocity_pred, velocity_target)
                all_metrics.append(batch_metrics)

            # Progress logging
            if batch_idx % max(1, len(train_loader) // 10) == 0:
                memory_mb = self.get_memory_usage()
                print(f"      Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f} | "
                      f"RMSE: {batch_metrics.get('rmse', 0):.1f} m/s | "
                      f"Vel Range: [{velocity_pred.min():.0f}, {velocity_pred.max():.0f}] | "
                      f"Memory: {memory_mb:.1f}MB")

            # Memory cleanup every 20 batches
            if batch_idx % 20 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Average metrics and components
        avg_metrics = self._average_metrics(all_metrics)
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in total_components.items()}

        return avg_loss, avg_metrics, avg_components

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

    def _save_checkpoint(self, path: Path, epoch: int, phase: str):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'model_config': {
                'temporal_hidden_size': 32,
                'spatial_output_channels': 64,
                'num_sources': 5,
                'output_size': (70, 70),
                'velocity_range': (1500.0, 4500.0)
            }
        }, path)

    def _save_history(self, path: Path):
        """Save training history to JSON."""
        with open(path, 'w') as f:
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

def main():
    """Main training function for the full dataset with FIXED model."""
    print("🚀 FIXED MODEL TRAINING ON FULL 13.22GB DATASET")
    print("="*60)

    try:
        # System info
        print(f"💻 System Info:")
        print(f"  CPUs: {psutil.cpu_count()}")
        print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"  Available: {psutil.virtual_memory().available / 1024**3:.1f} GB")

        # Create FIXED model
        print("🏗️  Creating FIXED seismic inversion model...")
        model = FixedSeismicInversionNet(
            temporal_hidden_size=32,
            spatial_output_channels=64,
            num_sources=5,
            output_size=(70, 70),
            velocity_range=(1500.0, 4500.0)  # Realistic velocity range
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ FIXED model created with {total_params:,} parameters")
        print(f"📊 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

        # Test model output range
        test_input = torch.randn(1, 5, 1000, 70)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"✅ Model output range: [{test_output.min():.1f}, {test_output.max():.1f}] m/s")

        # Create trainer with optimized settings for full dataset
        print("🎯 Creating full dataset trainer...")
        trainer = FullDatasetTrainer(
            model=model,
            batch_size=8,  # Balanced batch size for memory efficiency
            num_workers=4,  # Parallel data loading
            save_dir='fixed_full_dataset_checkpoints'
        )

        # Start comprehensive training on full dataset
        print("🏃 Starting comprehensive training on FULL DATASET...")
        history = trainer.train_progressive(
            complexity_schedule=[
                (['simple'], 10),                           # Simple geological models
                (['simple', 'intermediate'], 10),           # Add style variations
                (['simple', 'intermediate', 'complex'], 15) # Full complexity with faults
            ],
            max_samples_per_scenario=3000  # Use substantial portion of each scenario
        )

        print("🎉 FIXED MODEL training completed successfully!")

        if history['train_loss']:
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            final_train_rmse = history['train_metrics'][-1].get('rmse', 0)
            final_val_rmse = history['val_metrics'][-1].get('rmse', 0)
            avg_epoch_time = np.mean(history['epoch_times'])
            peak_memory = max(history['memory_usage'])

            print(f"\n📊 FINAL RESULTS:")
            print(f"  Final Training Loss: {final_train_loss:.6f}")
            print(f"  Final Validation Loss: {final_val_loss:.6f}")
            print(f"  Final Training RMSE: {final_train_rmse:.1f} m/s")
            print(f"  Final Validation RMSE: {final_val_rmse:.1f} m/s")
            print(f"  Average Epoch Time: {avg_epoch_time:.2f}s")
            print(f"  Peak Memory Usage: {peak_memory:.1f} MB")

            # Calculate improvement
            initial_train_loss = history['train_loss'][0]
            improvement = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
            print(f"  Training Improvement: {improvement:.1f}%")

        print(f"\n🎯 Model files saved in: fixed_full_dataset_checkpoints/")
        print(f"  - best_fixed_model.pth (best validation loss)")
        print(f"  - final_fixed_model.pth (final epoch)")
        print(f"  - fixed_training_history.json (complete training log)")

    except Exception as e:
        print(f"❌ FIXED model training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
