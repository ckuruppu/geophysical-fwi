#!/usr/bin/env python3
"""
Training pipeline for seismic waveform inversion neural network.
Implements progressive training and comprehensive evaluation with parallelism.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, List, Tuple
import json
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from seismic_inversion_model import SeismicInversionNet
from physics_informed_losses import PhysicsInformedLoss, compute_all_metrics
from seismic_data_loader import SeismicDataModule

# Set multiprocessing start method
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

class SeismicInversionTrainer:
    """
    Comprehensive trainer for seismic waveform inversion with parallelism and debug output.
    """

    def __init__(self,
                 model: SeismicInversionNet,
                 data_module: SeismicDataModule,
                 device: str = 'auto',
                 save_dir: str = 'checkpoints',
                 use_parallel: bool = True):

        print("🔧 Initializing trainer...")

        self.model = model
        self.data_module = data_module
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.use_parallel = use_parallel

        # Device setup with detailed logging
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🚀 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"  GPU Count: {torch.cuda.device_count()}")

        # Move model to device
        print("📦 Moving model to device...")
        self.model.to(self.device)

        # Enable data parallelism if multiple GPUs available
        if self.use_parallel and torch.cuda.device_count() > 1:
            print(f"🔄 Enabling DataParallel across {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        # Loss function
        print("⚖️  Setting up loss function...")
        self.criterion = PhysicsInformedLoss()

        # Optimizer and scheduler
        print("🎯 Setting up optimizer and scheduler...")
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # Training history
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }

        # TensorBoard logging
        print("📊 Setting up TensorBoard logging...")
        self.writer = SummaryWriter(log_dir=self.save_dir / 'tensorboard')

        print("✅ Trainer initialization complete!")
        
    def train_progressive(self,
                         complexity_schedule: List[Tuple[str, int]] = None,
                         save_checkpoints: bool = True) -> Dict[str, List[float]]:
        """
        Progressive training from simple to complex geological models with detailed progress tracking.

        Args:
            complexity_schedule: List of (complexity_level, num_epochs) tuples
            save_checkpoints: Whether to save model checkpoints

        Returns:
            Training history dictionary
        """
        if complexity_schedule is None:
            complexity_schedule = [
                ('simple', 20),      # Start with simple layered models
                ('intermediate', 15), # Add style variations
                ('complex', 15),     # Add fault structures
                ('all', 10)          # Full complexity
            ]

        print(f"🚀 Starting progressive training on {self.device}")
        print(f"📋 Schedule: {complexity_schedule}")

        total_start_time = time.time()
        total_epochs = 0

        for phase_idx, (complexity, num_epochs) in enumerate(complexity_schedule):
            print(f"\n{'='*60}")
            print(f"🎯 Training Phase {phase_idx + 1}/{len(complexity_schedule)}: {complexity.upper()} ({num_epochs} epochs)")
            print(f"{'='*60}")

            phase_start_time = time.time()

            # Get dataloaders for this complexity level
            print("📦 Loading data for this phase...")
            try:
                train_loader, val_loader = self.data_module.get_dataloaders(complexity)
                print(f"✅ Data loaded successfully")
                print(f"  Training samples: {len(train_loader.dataset)}")
                print(f"  Validation samples: {len(val_loader.dataset)}")
                print(f"  Training batches: {len(train_loader)}")
                print(f"  Validation batches: {len(val_loader)}")
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                continue

            # Train for this phase
            print(f"🏃 Starting training for {num_epochs} epochs...")
            phase_history = self._train_phase(
                train_loader,
                val_loader,
                num_epochs,
                phase_name=complexity,
                start_epoch=total_epochs
            )

            # Update total history
            self.train_history['train_loss'].extend(phase_history['train_loss'])
            self.train_history['val_loss'].extend(phase_history['val_loss'])
            self.train_history['train_metrics'].extend(phase_history['train_metrics'])
            self.train_history['val_metrics'].extend(phase_history['val_metrics'])
            self.train_history['learning_rates'].extend(phase_history['learning_rates'])

            total_epochs += num_epochs
            phase_time = time.time() - phase_start_time

            print(f"✅ Phase {phase_idx + 1} completed in {phase_time:.2f}s")

            # Save checkpoint after each phase
            if save_checkpoints:
                print("💾 Saving checkpoint...")
                checkpoint_path = self.save_dir / f'checkpoint_{complexity}_epoch_{total_epochs}.pth'
                self._save_checkpoint(checkpoint_path, total_epochs, complexity)
                print(f"✅ Saved checkpoint: {checkpoint_path}")

        total_time = time.time() - total_start_time
        print(f"\n🎉 Progressive training completed!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Total epochs: {total_epochs}")

        # Save final model
        print("💾 Saving final model...")
        final_path = self.save_dir / 'final_model.pth'
        self._save_checkpoint(final_path, total_epochs, 'final')

        # Save training history
        print("📋 Saving training history...")
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)

        print("✅ All files saved successfully!")
        return self.train_history
    
    def _train_phase(self, 
                    train_loader, 
                    val_loader, 
                    num_epochs: int,
                    phase_name: str,
                    start_epoch: int = 0) -> Dict[str, List[float]]:
        """Train for a single phase."""
        
        phase_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            global_epoch = start_epoch + epoch
            
            # Training
            train_loss, train_metrics = self._train_epoch(train_loader, global_epoch)
            
            # Validation
            val_loss, val_metrics = self._validate_epoch(val_loader, global_epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            phase_history['train_loss'].append(train_loss)
            phase_history['val_loss'].append(val_loss)
            phase_history['train_metrics'].append(train_metrics)
            phase_history['val_metrics'].append(val_metrics)
            phase_history['learning_rates'].append(current_lr)
            
            # Logging
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val RMSE: {val_metrics['rmse']:.1f} | "
                  f"LR: {current_lr:.2e}")
            
            # TensorBoard logging
            self.writer.add_scalar(f'{phase_name}/train_loss', train_loss, global_epoch)
            self.writer.add_scalar(f'{phase_name}/val_loss', val_loss, global_epoch)
            self.writer.add_scalar(f'{phase_name}/val_rmse', val_metrics['rmse'], global_epoch)
            self.writer.add_scalar(f'{phase_name}/learning_rate', current_lr, global_epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = self.save_dir / f'best_{phase_name}.pth'
                self._save_checkpoint(best_path, global_epoch, f'best_{phase_name}')
        
        return phase_history
    
    def _train_epoch(self, train_loader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch with detailed progress tracking."""
        self.model.train()

        total_loss = 0.0
        all_metrics = []
        epoch_start_time = time.time()

        print(f"    🏃 Training epoch {epoch + 1}...")

        for batch_idx, (seismic_data, velocity_target) in enumerate(train_loader):
            batch_start_time = time.time()

            # Move data to device
            seismic_data = seismic_data.to(self.device, non_blocking=True)
            velocity_target = velocity_target.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()

            try:
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
                    batch_metrics = compute_all_metrics(velocity_pred, velocity_target)
                    all_metrics.append(batch_metrics)

                # Log batch progress with timing
                batch_time = time.time() - batch_start_time
                if batch_idx % max(1, len(train_loader) // 10) == 0:  # Log 10 times per epoch
                    print(f"      Batch {batch_idx:3d}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Time: {batch_time:.2f}s | "
                          f"RMSE: {batch_metrics.get('rmse', 0):.1f}")

            except Exception as e:
                print(f"      ❌ Error in batch {batch_idx}: {e}")
                continue

        # Average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')

        epoch_time = time.time() - epoch_start_time
        print(f"    ✅ Training epoch completed in {epoch_time:.2f}s")

        return avg_loss, avg_metrics
    
    def _validate_epoch(self, val_loader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch with progress tracking."""
        self.model.eval()

        total_loss = 0.0
        all_metrics = []
        val_start_time = time.time()

        print(f"    🔍 Validating epoch {epoch + 1}...")

        with torch.no_grad():
            for batch_idx, (seismic_data, velocity_target) in enumerate(val_loader):
                seismic_data = seismic_data.to(self.device, non_blocking=True)
                velocity_target = velocity_target.to(self.device, non_blocking=True)

                try:
                    # Forward pass
                    velocity_pred = self.model(seismic_data)

                    # Compute loss
                    losses = self.criterion(velocity_pred, velocity_target, seismic_data)
                    loss = losses['total']

                    total_loss += loss.item()

                    # Compute metrics
                    batch_metrics = compute_all_metrics(velocity_pred, velocity_target)
                    all_metrics.append(batch_metrics)

                except Exception as e:
                    print(f"      ❌ Error in validation batch {batch_idx}: {e}")
                    continue

        # Average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

        val_time = time.time() - val_start_time
        print(f"    ✅ Validation completed in {val_time:.2f}s")

        return avg_loss, avg_metrics
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across batches."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
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
            'train_history': self.train_history
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, phase {checkpoint['phase']}")
        
        return checkpoint['epoch']

def main():
    """Main training function with comprehensive error handling and progress tracking."""
    print("🚀 Initializing Seismic Waveform Inversion Training...")

    try:
        # Create model
        print("🏗️  Creating model architecture...")
        model = SeismicInversionNet(
            temporal_hidden_size=64,
            spatial_output_channels=128,
            num_sources=5,
            output_size=(70, 70)
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")
        print(f"📊 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

        # Create data module with optimized settings
        print("📦 Setting up data module...")
        data_module = SeismicDataModule(
            data_root="waveform-inversion",
            batch_size=2,  # Very small batch size for testing
            val_split=0.2,
            num_workers=0,  # Disable multiprocessing for debugging
            augment_train=False  # Disable augmentation for faster training
        )
        print("✅ Data module created")

        # Create trainer
        print("🎯 Creating trainer...")
        trainer = SeismicInversionTrainer(
            model=model,
            data_module=data_module,
            device='auto',
            save_dir='seismic_inversion_checkpoints',
            use_parallel=False  # Disable parallelism for debugging
        )

        # Start progressive training with minimal schedule for testing
        print("🏃 Starting progressive training...")
        history = trainer.train_progressive(
            complexity_schedule=[
                ('simple', 2),      # Very quick test
                ('intermediate', 1),
            ],
            save_checkpoints=True
        )

        print("🎉 Training completed successfully!")
        if history['val_loss']:
            print(f"📊 Final validation loss: {history['val_loss'][-1]:.4f}")
        else:
            print("📊 No validation loss recorded")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

        print("\n🔧 Troubleshooting suggestions:")
        print("1. Check if data directory exists and contains valid files")
        print("2. Reduce batch size if memory issues occur")
        print("3. Ensure PyTorch and dependencies are properly installed")
        print("4. Try running the demo script first to verify setup")

if __name__ == "__main__":
    main()
