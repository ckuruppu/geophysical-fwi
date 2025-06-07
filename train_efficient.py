#!/usr/bin/env python3
"""
Efficient training script for seismic waveform inversion.
Uses the efficient model architecture optimized for CPU training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from seismic_model_efficient import EfficientSeismicInversionNet
from seismic_data_loader import SeismicDataModule

class EfficientTrainer:
    """
    Efficient trainer optimized for CPU training with progress tracking.
    """
    
    def __init__(self,
                 model: EfficientSeismicInversionNet,
                 data_module: SeismicDataModule,
                 save_dir: str = 'efficient_checkpoints'):
        
        print("🔧 Initializing efficient trainer...")
        
        self.model = model
        self.data_module = data_module
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Use CPU for efficiency
        self.device = torch.device('cpu')
        print(f"🚀 Using device: {self.device}")
        
        self.model.to(self.device)
        
        # Simple loss function for efficiency
        self.criterion = nn.MSELoss()
        
        # Optimizer with smaller learning rate for stability
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,  # Smaller learning rate
            weight_decay=1e-5
        )
        
        # Simple scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,
            gamma=0.8
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        print("✅ Efficient trainer initialized!")
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch with detailed progress."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        print(f"    🏃 Training epoch {epoch + 1}...")
        epoch_start = time.time()
        
        for batch_idx, (seismic_data, velocity_target) in enumerate(train_loader):
            batch_start = time.time()
            
            # Move data to device
            seismic_data = seismic_data.to(self.device)
            velocity_target = velocity_target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            velocity_pred = self.model(seismic_data)
            
            # Compute loss
            loss = self.criterion(velocity_pred, velocity_target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Progress logging
            batch_time = time.time() - batch_start
            if batch_idx % max(1, num_batches // 5) == 0:  # Log 5 times per epoch
                print(f"      Batch {batch_idx + 1}/{num_batches} | "
                      f"Loss: {loss.item():.6f} | "
                      f"Time: {batch_time:.2f}s")
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        print(f"    ✅ Training epoch completed in {epoch_time:.2f}s")
        print(f"    📊 Average loss: {avg_loss:.6f}")
        
        return avg_loss, epoch_time
    
    def validate_epoch(self, val_loader, epoch: int) -> float:
        """Validate for one epoch."""
        if len(val_loader) == 0:
            print(f"    ⚠️  No validation data available")
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        print(f"    🔍 Validating epoch {epoch + 1}...")
        val_start = time.time()
        
        with torch.no_grad():
            for seismic_data, velocity_target in val_loader:
                seismic_data = seismic_data.to(self.device)
                velocity_target = velocity_target.to(self.device)
                
                velocity_pred = self.model(seismic_data)
                loss = self.criterion(velocity_pred, velocity_target)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        val_time = time.time() - val_start
        
        print(f"    ✅ Validation completed in {val_time:.2f}s")
        print(f"    📊 Validation loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def train_progressive(self, 
                         complexity_schedule: List[Tuple[str, int]] = None,
                         save_checkpoints: bool = True) -> Dict[str, List[float]]:
        """
        Progressive training with efficient settings.
        """
        if complexity_schedule is None:
            complexity_schedule = [
                ('simple', 3),      # Quick training
                ('intermediate', 2),
                ('complex', 2),
            ]
        
        print(f"🚀 Starting efficient progressive training")
        print(f"📋 Schedule: {complexity_schedule}")
        
        total_start_time = time.time()
        total_epochs = 0
        
        for phase_idx, (complexity, num_epochs) in enumerate(complexity_schedule):
            print(f"\n{'='*60}")
            print(f"🎯 Phase {phase_idx + 1}/{len(complexity_schedule)}: {complexity.upper()} ({num_epochs} epochs)")
            print(f"{'='*60}")
            
            phase_start = time.time()
            
            # Get data for this phase
            try:
                train_loader, val_loader = self.data_module.get_dataloaders(complexity)
                print(f"✅ Data loaded: {len(train_loader)} train, {len(val_loader)} val batches")
            except Exception as e:
                print(f"❌ Error loading data for {complexity}: {e}")
                continue
            
            # Train for this phase
            for epoch in range(num_epochs):
                epoch_start = time.time()
                
                # Training
                train_loss, train_time = self.train_epoch(train_loader, total_epochs)
                
                # Validation
                val_loss = self.validate_epoch(val_loader, total_epochs)
                
                # Update scheduler
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Store history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rates'].append(current_lr)
                self.history['epoch_times'].append(train_time)
                
                total_epochs += 1
                epoch_total_time = time.time() - epoch_start
                
                print(f"  📊 Epoch {total_epochs} Summary:")
                print(f"    Train Loss: {train_loss:.6f}")
                print(f"    Val Loss: {val_loss:.6f}")
                print(f"    Learning Rate: {current_lr:.2e}")
                print(f"    Total Time: {epoch_total_time:.2f}s")
            
            phase_time = time.time() - phase_start
            print(f"✅ Phase {phase_idx + 1} completed in {phase_time:.2f}s")
            
            # Save checkpoint after each phase
            if save_checkpoints:
                checkpoint_path = self.save_dir / f'checkpoint_{complexity}_epoch_{total_epochs}.pth'
                self.save_checkpoint(checkpoint_path, total_epochs, complexity)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 Training completed!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Total epochs: {total_epochs}")
        
        # Save final model and history
        final_path = self.save_dir / 'final_efficient_model.pth'
        self.save_checkpoint(final_path, total_epochs, 'final')
        
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"💾 Saved final model and history")
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, phase: str):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, path)
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.history['train_loss']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        if self.history['val_loss'] and any(v != float('inf') for v in self.history['val_loss']):
            valid_val_loss = [v for v in self.history['val_loss'] if v != float('inf')]
            if valid_val_loss:
                axes[0, 0].plot(epochs[:len(valid_val_loss)], valid_val_loss, 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[0, 1].plot(epochs, self.history['learning_rates'], 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Epoch time plot
        axes[1, 0].plot(epochs, self.history['epoch_times'], 'm-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss improvement plot
        if len(self.history['train_loss']) > 1:
            loss_improvement = [self.history['train_loss'][0] - loss for loss in self.history['train_loss']]
            axes[1, 1].plot(epochs, loss_improvement, 'c-')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Improvement')
            axes[1, 1].set_title('Cumulative Loss Improvement')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training history plot saved to {plot_path}")

def main():
    """Main efficient training function."""
    print("🚀 EFFICIENT SEISMIC WAVEFORM INVERSION TRAINING")
    print("="*60)
    
    try:
        # Create efficient model
        print("🏗️  Creating efficient model...")
        model = EfficientSeismicInversionNet(
            temporal_hidden_size=16,  # Small for CPU efficiency
            spatial_output_channels=32,
            num_sources=5,
            output_size=(70, 70)
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")
        print(f"📊 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Create efficient data module
        print("📦 Setting up data module...")
        data_module = SeismicDataModule(
            data_root="waveform-inversion",
            batch_size=1,  # Single sample for CPU efficiency
            val_split=0.2,
            num_workers=0,  # No multiprocessing for simplicity
            augment_train=False  # Disable augmentation for speed
        )
        print("✅ Data module created")
        
        # Create trainer
        print("🎯 Creating efficient trainer...")
        trainer = EfficientTrainer(
            model=model,
            data_module=data_module,
            save_dir='efficient_checkpoints'
        )
        
        # Start training
        print("🏃 Starting efficient training...")
        history = trainer.train_progressive(
            complexity_schedule=[
                ('simple', 2),      # Quick test
                ('intermediate', 1),
            ],
            save_checkpoints=True
        )
        
        # Plot results
        print("📊 Creating training plots...")
        trainer.plot_training_history()
        
        print("🎉 Efficient training completed successfully!")
        if history['train_loss']:
            print(f"📊 Final training loss: {history['train_loss'][-1]:.6f}")
            print(f"⏱️  Average epoch time: {np.mean(history['epoch_times']):.2f}s")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
