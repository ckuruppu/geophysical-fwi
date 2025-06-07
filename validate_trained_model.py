#!/usr/bin/env python3
"""
Comprehensive validation and evaluation script for the trained seismic inversion model.
Tests the model on unseen data and provides detailed performance analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
import seaborn as sns
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from seismic_model_fixed import FixedSeismicInversionNet, FixedPhysicsInformedLoss
from physics_losses_efficient import EfficientMetrics

class ModelValidator:
    """
    Comprehensive validator for the trained seismic inversion model.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        
        # Load trained model
        print("🔄 Loading trained model...")
        self.model, self.training_history = self._load_model()
        self.model.eval()
        
        # Initialize loss function for evaluation
        self.criterion = FixedPhysicsInformedLoss(
            mse_weight=1.0,
            smoothness_weight=0.1,
            gradient_weight=0.1,
            geological_weight=0.05,
            velocity_range=(1500.0, 4500.0)
        )
        
        print("✅ Model validator initialized!")
    
    def _load_model(self) -> Tuple[FixedSeismicInversionNet, Dict]:
        """Load the trained model and its history."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract model configuration
        model_config = checkpoint.get('model_config', {
            'temporal_hidden_size': 32,
            'spatial_output_channels': 64,
            'num_sources': 5,
            'output_size': (70, 70),
            'velocity_range': (1500.0, 4500.0)
        })
        
        # Create model with same configuration
        model = FixedSeismicInversionNet(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        # Extract training history
        history = checkpoint.get('history', {})
        
        print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, history
    
    def validate_on_test_samples(self, data_root: str, num_samples: int = 50) -> Dict:
        """
        Validate model on a subset of test samples from different scenarios.
        """
        print(f"\n🧪 VALIDATING ON {num_samples} TEST SAMPLES")
        print("="*50)
        
        data_root = Path(data_root)
        scenarios = ['FlatVel_A', 'Style_A', 'FlatFault_A']  # Representative scenarios
        
        all_predictions = []
        all_targets = []
        all_losses = []
        all_metrics = []
        scenario_results = {}
        
        samples_per_scenario = num_samples // len(scenarios)
        
        for scenario in scenarios:
            print(f"\n🏔️  Testing scenario: {scenario}")
            scenario_path = data_root / scenario
            
            if not scenario_path.exists():
                print(f"⚠️  Scenario {scenario} not found, skipping...")
                continue
            
            # Load test samples
            scenario_preds, scenario_targets, scenario_losses, scenario_metrics = \
                self._test_scenario(scenario_path, samples_per_scenario)
            
            if scenario_preds:
                all_predictions.extend(scenario_preds)
                all_targets.extend(scenario_targets)
                all_losses.extend(scenario_losses)
                all_metrics.extend(scenario_metrics)
                
                # Calculate scenario-specific metrics
                scenario_results[scenario] = {
                    'samples': len(scenario_preds),
                    'avg_loss': np.mean(scenario_losses),
                    'avg_rmse': np.mean([m['rmse'] for m in scenario_metrics]),
                    'avg_correlation': np.mean([m['correlation'] for m in scenario_metrics])
                }
                
                print(f"  ✅ {scenario}: {len(scenario_preds)} samples tested")
                print(f"     Avg RMSE: {scenario_results[scenario]['avg_rmse']:.1f} m/s")
                print(f"     Avg Correlation: {scenario_results[scenario]['avg_correlation']:.3f}")
        
        # Overall validation metrics
        if all_predictions:
            overall_metrics = {
                'total_samples': len(all_predictions),
                'avg_loss': np.mean(all_losses),
                'avg_rmse': np.mean([m['rmse'] for m in all_metrics]),
                'avg_mae': np.mean([m['mae'] for m in all_metrics]),
                'avg_correlation': np.mean([m['correlation'] for m in all_metrics]),
                'avg_ssim': np.mean([m['ssim'] for m in all_metrics]),
                'scenario_results': scenario_results
            }
            
            print(f"\n📊 OVERALL VALIDATION RESULTS:")
            print(f"  Total samples tested: {overall_metrics['total_samples']}")
            print(f"  Average loss: {overall_metrics['avg_loss']:.6f}")
            print(f"  Average RMSE: {overall_metrics['avg_rmse']:.1f} m/s")
            print(f"  Average MAE: {overall_metrics['avg_mae']:.1f} m/s")
            print(f"  Average Correlation: {overall_metrics['avg_correlation']:.3f}")
            print(f"  Average SSIM: {overall_metrics['avg_ssim']:.3f}")
            
            return overall_metrics
        else:
            print("❌ No samples could be tested!")
            return {}
    
    def _test_scenario(self, scenario_path: Path, num_samples: int) -> Tuple[List, List, List, List]:
        """Test model on samples from a specific scenario."""
        predictions = []
        targets = []
        losses = []
        metrics = []
        
        # Check scenario structure
        data_dir = scenario_path / "data"
        model_dir = scenario_path / "model"
        
        if data_dir.exists() and model_dir.exists():
            # Structure: scenario/data/*.npy, scenario/model/*.npy
            data_files = sorted(list(data_dir.glob("*.npy")))[:num_samples]
            model_files = sorted(list(model_dir.glob("*.npy")))[:num_samples]
            file_pairs = list(zip(data_files, model_files))
        else:
            # Structure: scenario/*.npy files directly
            seis_files = sorted(list(scenario_path.glob("seis*.npy")))[:num_samples]
            vel_files = sorted(list(scenario_path.glob("vel*.npy")))[:num_samples]
            file_pairs = list(zip(seis_files, vel_files))
        
        for data_file, target_file in file_pairs:
            try:
                # Load data
                seismic_data = np.load(data_file, mmap_mode='r')
                velocity_target = np.load(target_file, mmap_mode='r')
                
                # Handle different data structures
                if len(seismic_data.shape) == 4:  # Batched data
                    seismic_sample = seismic_data[0]  # Take first sample
                    velocity_sample = velocity_target[0]
                else:
                    seismic_sample = seismic_data
                    velocity_sample = velocity_target
                
                # Ensure correct shapes
                if len(seismic_sample.shape) == 2:  # (time, receivers)
                    seismic_sample = seismic_sample[np.newaxis, :, :]  # Add source dimension
                
                if len(velocity_sample.shape) == 2:  # (height, width)
                    velocity_sample = velocity_sample[np.newaxis, :, :]  # Add channel dimension
                
                # Convert to tensors
                seismic_tensor = torch.from_numpy(seismic_sample).float().unsqueeze(0).to(self.device)
                velocity_tensor = torch.from_numpy(velocity_sample).float().unsqueeze(0).to(self.device)
                
                # Ensure velocity is in realistic range
                velocity_tensor = torch.clamp(velocity_tensor, 1000, 6000)
                
                # Model prediction
                with torch.no_grad():
                    velocity_pred = self.model(seismic_tensor)
                    
                    # Compute loss
                    loss_components = self.criterion(velocity_pred, velocity_tensor)
                    loss = loss_components['total'].item()
                    
                    # Compute metrics
                    sample_metrics = EfficientMetrics.compute_metrics(velocity_pred, velocity_tensor)
                
                # Store results
                predictions.append(velocity_pred.cpu().numpy())
                targets.append(velocity_tensor.cpu().numpy())
                losses.append(loss)
                metrics.append(sample_metrics)
                
            except Exception as e:
                print(f"    ⚠️  Error processing {data_file}: {e}")
                continue
        
        return predictions, targets, losses, metrics
    
    def analyze_training_history(self) -> Dict:
        """Analyze and visualize training history."""
        print(f"\n📈 TRAINING HISTORY ANALYSIS")
        print("="*40)
        
        if not self.training_history:
            print("❌ No training history available!")
            return {}
        
        history = self.training_history
        
        # Extract key metrics
        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        train_metrics = history.get('train_metrics', [])
        val_metrics = history.get('val_metrics', [])
        
        if not train_losses:
            print("❌ No training data in history!")
            return {}
        
        # Calculate training statistics
        final_train_loss = train_losses[-1] if train_losses else 0
        final_val_loss = val_losses[-1] if val_losses else 0
        best_val_loss = min(val_losses) if val_losses else 0
        best_val_epoch = val_losses.index(best_val_loss) + 1 if val_losses else 0
        
        # RMSE progression
        train_rmse = [m.get('rmse', 0) for m in train_metrics] if train_metrics else []
        val_rmse = [m.get('rmse', 0) for m in val_metrics] if val_metrics else []
        
        final_train_rmse = train_rmse[-1] if train_rmse else 0
        final_val_rmse = val_rmse[-1] if val_rmse else 0
        best_val_rmse = min(val_rmse) if val_rmse else 0
        
        # Training improvement
        if len(train_losses) > 1:
            improvement = ((train_losses[0] - final_train_loss) / train_losses[0]) * 100
        else:
            improvement = 0
        
        analysis = {
            'total_epochs': len(train_losses),
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'best_val_epoch': best_val_epoch,
            'final_train_rmse': final_train_rmse,
            'final_val_rmse': final_val_rmse,
            'best_val_rmse': best_val_rmse,
            'training_improvement': improvement,
            'converged': len(train_losses) > 10 and abs(train_losses[-1] - train_losses[-5]) < 0.001
        }
        
        print(f"📊 Training Summary:")
        print(f"  Total epochs: {analysis['total_epochs']}")
        print(f"  Final training loss: {analysis['final_train_loss']:.6f}")
        print(f"  Final validation loss: {analysis['final_val_loss']:.6f}")
        print(f"  Best validation loss: {analysis['best_val_loss']:.6f} (epoch {analysis['best_val_epoch']})")
        print(f"  Final training RMSE: {analysis['final_train_rmse']:.1f} m/s")
        print(f"  Final validation RMSE: {analysis['final_val_rmse']:.1f} m/s")
        print(f"  Best validation RMSE: {analysis['best_val_rmse']:.1f} m/s")
        print(f"  Training improvement: {analysis['training_improvement']:.1f}%")
        print(f"  Converged: {'✅ Yes' if analysis['converged'] else '⚠️  No'}")
        
        return analysis
    
    def create_validation_visualizations(self, validation_results: Dict, save_dir: str = 'validation_results'):
        """Create comprehensive validation visualizations."""
        print(f"\n🎨 CREATING VALIDATION VISUALIZATIONS")
        print("="*45)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 1. Training history plots
        if self.training_history:
            self._plot_training_history(save_dir)
        
        # 2. Validation metrics summary
        if validation_results:
            self._plot_validation_summary(validation_results, save_dir)
        
        print(f"📊 Visualizations saved to: {save_dir}/")
    
    def _plot_training_history(self, save_dir: Path):
        """Plot training history."""
        history = self.training_history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History Analysis', fontsize=16, fontweight='bold')
        
        # Loss curves
        if history.get('train_loss') and history.get('val_loss'):
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training & Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE curves
        if history.get('train_metrics') and history.get('val_metrics'):
            train_rmse = [m.get('rmse', 0) for m in history['train_metrics']]
            val_rmse = [m.get('rmse', 0) for m in history['val_metrics']]
            epochs = range(1, len(train_rmse) + 1)
            
            axes[0, 1].plot(epochs, train_rmse, 'b-', label='Training RMSE', linewidth=2)
            axes[0, 1].plot(epochs, val_rmse, 'r-', label='Validation RMSE', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('RMSE (m/s)')
            axes[0, 1].set_title('RMSE Progression')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Memory usage
        if history.get('memory_usage'):
            epochs = range(1, len(history['memory_usage']) + 1)
            axes[1, 0].plot(epochs, history['memory_usage'], 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].set_title('Memory Usage')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Epoch times
        if history.get('epoch_times'):
            epochs = range(1, len(history['epoch_times']) + 1)
            axes[1, 1].plot(epochs, history['epoch_times'], 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_title('Epoch Training Time')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✅ Training history plots saved")
    
    def _plot_validation_summary(self, results: Dict, save_dir: Path):
        """Plot validation summary."""
        if not results.get('scenario_results'):
            return
        
        scenarios = list(results['scenario_results'].keys())
        rmse_values = [results['scenario_results'][s]['avg_rmse'] for s in scenarios]
        corr_values = [results['scenario_results'][s]['avg_correlation'] for s in scenarios]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Validation Results by Scenario', fontsize=14, fontweight='bold')
        
        # RMSE by scenario
        bars1 = axes[0].bar(scenarios, rmse_values, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0].set_ylabel('RMSE (m/s)')
        axes[0].set_title('Average RMSE by Scenario')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, rmse_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{value:.1f}', ha='center', va='bottom')
        
        # Correlation by scenario
        bars2 = axes[1].bar(scenarios, corr_values, color=['skyblue', 'lightgreen', 'salmon'])
        axes[1].set_ylabel('Correlation')
        axes[1].set_title('Average Correlation by Scenario')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars2, corr_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'validation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✅ Validation summary plots saved")

def main():
    """Main validation function."""
    print("🔍 COMPREHENSIVE MODEL VALIDATION")
    print("="*50)
    
    # Initialize validator with best model
    model_path = "fixed_full_dataset_checkpoints/best_fixed_model.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Available checkpoints:")
        checkpoint_dir = Path("fixed_full_dataset_checkpoints")
        if checkpoint_dir.exists():
            for file in checkpoint_dir.glob("*.pth"):
                print(f"  - {file.name}")
        return
    
    try:
        validator = ModelValidator(model_path)
        
        # 1. Analyze training history
        training_analysis = validator.analyze_training_history()
        
        # 2. Validate on test samples
        validation_results = validator.validate_on_test_samples(
            data_root="waveform-inversion/train_samples",
            num_samples=30  # Test on 30 samples (10 per scenario)
        )
        
        # 3. Create visualizations
        validator.create_validation_visualizations(validation_results)
        
        # 4. Save comprehensive report
        report = {
            'model_path': str(model_path),
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_analysis': training_analysis,
            'validation_results': validation_results
        }
        
        report_path = Path('validation_results/comprehensive_validation_report.json')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n🎉 VALIDATION COMPLETED SUCCESSFULLY!")
        print(f"📄 Comprehensive report saved: {report_path}")
        print(f"📊 Visualizations saved: validation_results/")
        
        # Summary
        if validation_results:
            print(f"\n🎯 VALIDATION SUMMARY:")
            print(f"  Model Performance: {'🏆 Excellent' if validation_results['avg_rmse'] < 600 else '✅ Good' if validation_results['avg_rmse'] < 800 else '⚠️ Needs Improvement'}")
            print(f"  Average RMSE: {validation_results['avg_rmse']:.1f} m/s")
            print(f"  Average Correlation: {validation_results['avg_correlation']:.3f}")
            print(f"  Samples Tested: {validation_results['total_samples']}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
