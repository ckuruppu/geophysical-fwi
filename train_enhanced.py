#!/usr/bin/env python3
"""
Enhanced training script with physics-informed losses and scaled model.
Combines computational efficiency with sophisticated loss functions and increased model capacity.
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

from seismic_model_scaled import ScaledSeismicInversionNet
from physics_losses_efficient import EfficientPhysicsInformedLoss, EfficientMetrics
from seismic_data_loader import SeismicDataModule

class EnhancedTrainer:
    """
    Enhanced trainer with physics-informed losses and advanced features.
    """
    
    def __init__(self,
                 model: ScaledSeismicInversionNet,
                 data_module: SeismicDataModule,
                 save_dir: str = 'enhanced_checkpoints',
                 use_physics_loss: bool = True):
        
        print("🔧 Initializing enhanced trainer...")
        
        self.model = model
        self.data_module = data_module
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.use_physics_loss = use_physics_loss
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.model.to(self.device)
        
        # Enhanced loss function
        if use_physics_loss:
            self.criterion = EfficientPhysicsInformedLoss(
                mse_weight=1.0,
                smoothness_weight=0.1,
                gradient_weight=0.05,
                geological_weight=0.02,
                velocity_range=(1500.0, 4500.0)
            )
            print("⚖️  Using physics-informed loss function")
        else:
            self.criterion = nn.MSELoss()
            print("⚖️  Using standard MSE loss function")
        
        # Advanced optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )
        
        # Training history with enhanced metrics
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'epoch_times': [],
            'loss_components': []
        }
        
        print("✅ Enhanced trainer initialized!")
    
    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Enhanced training epoch with detailed metrics."""
        self.model.train()
        total_loss = 0.0
        total_components = {}
        all_metrics = []
        num_batches = len(train_loader)
        
        print(f"    🏃 Training epoch {epoch + 1}...")
        epoch_start = time.time()
        
        for batch_idx, (seismic_data, velocity_target) in enumerate(train_loader):
            batch_start = time.time()
            
            # Move data to device
            seismic_data = seismic_data.to(self.device, non_blocking=True)
            velocity_target = velocity_target.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            velocity_pred = self.model(seismic_data)
            
            # Compute loss
            if self.use_physics_loss:
                losses = self.criterion(velocity_pred, velocity_target, seismic_data)
                loss = losses['total']
                
                # Accumulate loss components
                for key, value in losses.items():
                    if key not in total_components:
                        total_components[key] = 0.0
                    total_components[key] += value.item()
            else:
                loss = self.criterion(velocity_pred, velocity_target)
                total_components['mse'] = total_components.get('mse', 0.0) + loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute metrics
            with torch.no_grad():
                batch_metrics = EfficientMetrics.compute_metrics(velocity_pred, velocity_target)
                all_metrics.append(batch_metrics)
            
            # Progress logging
            batch_time = time.time() - batch_start
            if batch_idx % max(1, num_batches // 5) == 0:
                print(f"      Batch {batch_idx + 1}/{num_batches} | "
                      f"Loss: {loss.item():.6f} | "
                      f"RMSE: {batch_metrics['rmse']:.1f} | "
                      f"Time: {batch_time:.2f}s")
        
        # Average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_loss = total_loss / num_batches
        
        # Average loss components
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        epoch_time = time.time() - epoch_start
        print(f"    ✅ Training epoch completed in {epoch_time:.2f}s")
        print(f"    📊 Average loss: {avg_loss:.6f}")
        print(f"    📈 RMSE: {avg_metrics['rmse']:.1f} m/s")
        
        return avg_loss, avg_metrics, avg_components, epoch_time
    
    def validate_epoch(self, val_loader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Enhanced validation epoch."""
        if len(val_loader) == 0:
            print(f"    ⚠️  No validation data available")
            return float('inf'), {}
        
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        num_batches = len(val_loader)
        
        print(f"    🔍 Validating epoch {epoch + 1}...")
        val_start = time.time()
        
        with torch.no_grad():
            for seismic_data, velocity_target in val_loader:
                seismic_data = seismic_data.to(self.device, non_blocking=True)
                velocity_target = velocity_target.to(self.device, non_blocking=True)
                
                velocity_pred = self.model(seismic_data)
                
                if self.use_physics_loss:
                    losses = self.criterion(velocity_pred, velocity_target, seismic_data)
                    loss = losses['total']
                else:
                    loss = self.criterion(velocity_pred, velocity_target)
                
                total_loss += loss.item()
                
                # Compute metrics
                batch_metrics = EfficientMetrics.compute_metrics(velocity_pred, velocity_target)
                all_metrics.append(batch_metrics)
        
        avg_metrics = self._average_metrics(all_metrics)
        avg_loss = total_loss / num_batches
        
        val_time = time.time() - val_start
        print(f"    ✅ Validation completed in {val_time:.2f}s")
        print(f"    📊 Validation loss: {avg_loss:.6f}")
        print(f"    📈 Validation RMSE: {avg_metrics['rmse']:.1f} m/s")
        
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
    
    def train_progressive(self, 
                         complexity_schedule: List[Tuple[str, int]] = None,
                         save_checkpoints: bool = True) -> Dict[str, List[float]]:
        """
        Enhanced progressive training with physics-informed losses.
        """
        if complexity_schedule is None:
            complexity_schedule = [
                ('simple', 5),      # More epochs for better convergence
                ('intermediate', 4),
                ('complex', 3),
                ('all', 2)          # Final training on all data
            ]
        
        print(f"🚀 Starting enhanced progressive training")
        print(f"📋 Schedule: {complexity_schedule}")
        print(f"⚖️  Physics-informed losses: {'Enabled' if self.use_physics_loss else 'Disabled'}")
        
        total_start_time = time.time()
        total_epochs = 0
        best_val_loss = float('inf')
        
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
                train_loss, train_metrics, loss_components, train_time = self.train_epoch(train_loader, total_epochs)
                
                # Validation
                val_loss, val_metrics = self.validate_epoch(val_loader, total_epochs)
                
                # Update scheduler
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Store history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_metrics'].append(train_metrics)
                self.history['val_metrics'].append(val_metrics)
                self.history['learning_rates'].append(current_lr)
                self.history['epoch_times'].append(train_time)
                self.history['loss_components'].append(loss_components)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = self.save_dir / 'best_model.pth'
                    self.save_checkpoint(best_model_path, total_epochs, f'{complexity}_best')
                    print(f"    🏆 New best model saved! Val loss: {val_loss:.6f}")
                
                total_epochs += 1
                epoch_total_time = time.time() - epoch_start
                
                print(f"  📊 Epoch {total_epochs} Summary:")
                print(f"    Train Loss: {train_loss:.6f}")
                print(f"    Val Loss: {val_loss:.6f}")
                print(f"    Learning Rate: {current_lr:.2e}")
                print(f"    Total Time: {epoch_total_time:.2f}s")
                
                if self.use_physics_loss and loss_components:
                    print(f"    Loss Components:")
                    for comp, value in loss_components.items():
                        print(f"      {comp}: {value:.6f}")
            
            phase_time = time.time() - phase_start
            print(f"✅ Phase {phase_idx + 1} completed in {phase_time:.2f}s")
            
            # Save checkpoint after each phase
            if save_checkpoints:
                checkpoint_path = self.save_dir / f'checkpoint_{complexity}_epoch_{total_epochs}.pth'
                self.save_checkpoint(checkpoint_path, total_epochs, complexity)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 Enhanced training completed!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Total epochs: {total_epochs}")
        print(f"🏆 Best validation loss: {best_val_loss:.6f}")
        
        # Save final model and history
        final_path = self.save_dir / 'final_enhanced_model.pth'
        self.save_checkpoint(final_path, total_epochs, 'final')
        
        history_path = self.save_dir / 'enhanced_training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
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
        
        print(f"💾 Saved final model and history")
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, phase: str):
        """Save enhanced model checkpoint."""
        torch.save({
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'use_physics_loss': self.use_physics_loss
        }, path)

def main():
    """Main enhanced training function."""
    print("🚀 ENHANCED SEISMIC WAVEFORM INVERSION TRAINING")
    print("="*60)
    
    try:
        # Create scaled model
        print("🏗️  Creating scaled model...")
        model = ScaledSeismicInversionNet(
            temporal_hidden_size=64,
            spatial_output_channels=128,
            num_sources=5,
            output_size=(70, 70)
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")
        print(f"📊 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Create data module
        print("📦 Setting up data module...")
        data_module = SeismicDataModule(
            data_root="waveform-inversion",
            batch_size=2,  # Larger batch size for scaled model
            val_split=0.2,
            num_workers=0,
            augment_train=False
        )
        print("✅ Data module created")
        
        # Create enhanced trainer
        print("🎯 Creating enhanced trainer...")
        trainer = EnhancedTrainer(
            model=model,
            data_module=data_module,
            save_dir='enhanced_checkpoints',
            use_physics_loss=True  # Enable physics-informed losses
        )
        
        # Start enhanced training
        print("🏃 Starting enhanced training...")
        history = trainer.train_progressive(
            complexity_schedule=[
                ('simple', 3),      # Quick test with scaled model
                ('intermediate', 2),
                ('complex', 2),
            ],
            save_checkpoints=True
        )
        
        print("🎉 Enhanced training completed successfully!")
        if history['train_loss']:
            print(f"📊 Final training loss: {history['train_loss'][-1]:.6f}")
            print(f"📊 Final validation loss: {history['val_loss'][-1]:.6f}")
            print(f"⏱️  Average epoch time: {np.mean(history['epoch_times']):.2f}s")
        
    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
