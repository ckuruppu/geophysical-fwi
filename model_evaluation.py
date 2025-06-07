#!/usr/bin/env python3
"""
Comprehensive evaluation script for trained seismic inversion models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from seismic_model_scaled import ScaledSeismicInversionNet
from seismic_model_efficient import EfficientSeismicInversionNet
from physics_losses_efficient import EfficientPhysicsInformedLoss, EfficientMetrics
from seismic_data_loader import SeismicDataModule

class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, model_path: str, model_type: str = 'scaled', data_module: SeismicDataModule = None):
        print(f"🔍 Initializing model evaluator...")
        
        self.device = torch.device('cpu')  # Use CPU for evaluation
        print(f"🚀 Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        self.model.to(self.device)
        self.model.eval()
        
        self.data_module = data_module
        self.criterion = EfficientPhysicsInformedLoss()
        
        print("✅ Model evaluator initialized!")
    
    def _load_model(self, model_path: str, model_type: str):
        """Load trained model from checkpoint."""
        print(f"📦 Loading {model_type} model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if model_type == 'scaled':
            model = ScaledSeismicInversionNet(
                temporal_hidden_size=64,
                spatial_output_channels=128,
                num_sources=5,
                output_size=(70, 70)
            )
        elif model_type == 'efficient':
            model = EfficientSeismicInversionNet(
                temporal_hidden_size=16,  # Match the efficient model
                spatial_output_channels=32,
                num_sources=5,
                output_size=(70, 70)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded with {total_params:,} parameters")
        
        return model
    
    def evaluate_dataset(self, complexity: str = 'simple') -> Dict[str, float]:
        """Evaluate model on a specific dataset complexity."""
        print(f"📊 Evaluating on {complexity} dataset...")
        
        if self.data_module is None:
            raise ValueError("Data module not provided")
        
        # Get data loaders
        train_loader, val_loader = self.data_module.get_dataloaders(complexity)
        data_loader = val_loader if len(val_loader) > 0 else train_loader
        
        all_metrics = []
        all_losses = []
        inference_times = []
        
        print(f"  Processing {len(data_loader)} batches...")
        
        with torch.no_grad():
            for batch_idx, (seismic_data, velocity_target) in enumerate(data_loader):
                seismic_data = seismic_data.to(self.device)
                velocity_target = velocity_target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                velocity_pred = self.model(seismic_data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Compute losses
                losses = self.criterion(velocity_pred, velocity_target, seismic_data)
                all_losses.append({k: v.item() for k, v in losses.items()})
                
                # Compute metrics
                batch_metrics = EfficientMetrics.compute_metrics(velocity_pred, velocity_target)
                all_metrics.append(batch_metrics)
        
        # Average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_losses = self._average_losses(all_losses)
        
        # Add timing information
        avg_metrics['inference_time_mean'] = np.mean(inference_times)
        avg_metrics['throughput'] = 1.0 / np.mean(inference_times)
        
        # Combine metrics and losses
        evaluation_results = {**avg_metrics, **avg_losses}
        
        print(f"✅ Evaluation completed!")
        print(f"  📊 RMSE: {avg_metrics['rmse']:.1f} m/s")
        print(f"  📊 Correlation: {avg_metrics['correlation']:.3f}")
        print(f"  ⏱️  Avg inference time: {avg_metrics['inference_time_mean']:.3f}s")
        
        return evaluation_results
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Average metrics across batches."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key not in ['predicted_stats', 'target_stats']:
                avg_metrics[key] = np.mean([m[key] for m in metrics_list])
        
        return avg_metrics
    
    def _average_losses(self, losses_list: List[Dict]) -> Dict[str, float]:
        """Average losses across batches."""
        if not losses_list:
            return {}
        
        avg_losses = {}
        for key in losses_list[0].keys():
            avg_losses[f'loss_{key}'] = np.mean([l[key] for l in losses_list])
        
        return avg_losses
    
    def create_prediction_visualization(self, complexity: str = 'simple', save_path: str = 'evaluation_results.png'):
        """Create comprehensive visualization of model predictions."""
        print(f"🎨 Creating prediction visualization...")
        
        # Get sample data
        train_loader, val_loader = self.data_module.get_dataloaders(complexity)
        data_loader = val_loader if len(val_loader) > 0 else train_loader
        
        # Get first batch
        seismic_data, velocity_target = next(iter(data_loader))
        seismic_data = seismic_data.to(self.device)
        velocity_target = velocity_target.to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            velocity_pred = self.model(seismic_data)
        
        # Move to CPU for plotting
        seismic_data = seismic_data.cpu().numpy()
        velocity_target = velocity_target.cpu().numpy()
        velocity_pred = velocity_pred.cpu().numpy()
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(16, 10))
        
        # Sample index to visualize
        sample_idx = 0
        
        # 1. Target velocity model
        ax1 = plt.subplot(2, 4, 1)
        target_sample = velocity_target[sample_idx, 0, :, :]
        im1 = plt.imshow(target_sample, cmap='jet', vmin=1500, vmax=4500)
        plt.title('Target Velocity Model')
        plt.xlabel('X (grid points)')
        plt.ylabel('Depth (grid points)')
        plt.colorbar(im1, label='Velocity (m/s)')
        
        # 2. Predicted velocity model
        ax2 = plt.subplot(2, 4, 2)
        pred_sample = velocity_pred[sample_idx, 0, :, :]
        im2 = plt.imshow(pred_sample, cmap='jet', vmin=1500, vmax=4500)
        plt.title('Predicted Velocity Model')
        plt.xlabel('X (grid points)')
        plt.ylabel('Depth (grid points)')
        plt.colorbar(im2, label='Velocity (m/s)')
        
        # 3. Difference map
        ax3 = plt.subplot(2, 4, 3)
        diff = pred_sample - target_sample
        im3 = plt.imshow(diff, cmap='RdBu_r', vmin=-500, vmax=500)
        plt.title('Prediction Error')
        plt.xlabel('X (grid points)')
        plt.ylabel('Depth (grid points)')
        plt.colorbar(im3, label='Error (m/s)')
        
        # 4. Scatter plot: predicted vs target
        ax4 = plt.subplot(2, 4, 4)
        target_flat = target_sample.flatten()
        pred_flat = pred_sample.flatten()
        
        plt.scatter(target_flat, pred_flat, alpha=0.5, s=1)
        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2)
        plt.xlabel('Target Velocity (m/s)')
        plt.ylabel('Predicted Velocity (m/s)')
        plt.title('Predicted vs Target')
        plt.grid(True, alpha=0.3)
        
        # Calculate R²
        correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
        plt.text(0.05, 0.95, f'R² = {correlation**2:.3f}', transform=ax4.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Depth profiles comparison
        ax5 = plt.subplot(2, 4, 5)
        center_x = target_sample.shape[1] // 2
        target_profile = target_sample[:, center_x]
        pred_profile = pred_sample[:, center_x]
        depth = np.arange(len(target_profile))
        
        plt.plot(target_profile, depth, 'b-', label='Target', linewidth=2)
        plt.plot(pred_profile, depth, 'r--', label='Predicted', linewidth=2)
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Depth (grid points)')
        plt.title('Center Depth Profile')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # 6. Histogram of errors
        ax6 = plt.subplot(2, 4, 6)
        errors = (pred_flat - target_flat)
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error (m/s)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        plt.text(0.05, 0.95, f'Mean: {mean_error:.1f}\nStd: {std_error:.1f}', 
                transform=ax6.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 7. Seismic data visualization
        ax7 = plt.subplot(2, 4, 7)
        seismic_sample = seismic_data[sample_idx, 0, :, :]  # First source
        plt.imshow(seismic_sample, aspect='auto', cmap='seismic')
        plt.title('Input Seismic Data\n(Source 1)')
        plt.xlabel('Receiver')
        plt.ylabel('Time Sample')
        plt.colorbar()
        
        # 8. Metrics summary
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((pred_flat - target_flat)**2))
        mae = np.mean(np.abs(pred_flat - target_flat))
        rel_error = (mae / (target_flat.max() - target_flat.min())) * 100
        
        metrics_text = f"""Model Performance:
        
RMSE: {rmse:.1f} m/s
MAE: {mae:.1f} m/s
Relative Error: {rel_error:.1f}%
Correlation: {correlation:.3f}
R²: {correlation**2:.3f}

Data Statistics:
Target: {target_flat.min():.0f} - {target_flat.max():.0f} m/s
Pred: {pred_flat.min():.0f} - {pred_flat.max():.0f} m/s"""
        
        ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to {save_path}")

def main():
    """Main evaluation function."""
    print("🔍 SEISMIC INVERSION MODEL EVALUATION")
    print("="*60)
    
    try:
        # Setup data module
        print("📦 Setting up data module...")
        data_module = SeismicDataModule(
            data_root="waveform-inversion",
            batch_size=2,
            val_split=0.2,
            num_workers=0,
            augment_train=False
        )
        
        # Check for available models
        model_paths = {
            'enhanced': 'enhanced_checkpoints/best_model.pth',
            'efficient': 'efficient_checkpoints/final_efficient_model.pth'
        }
        
        for model_name, model_path in model_paths.items():
            if Path(model_path).exists():
                print(f"\n🔍 Evaluating {model_name} model...")
                
                # Determine model type
                model_type = 'scaled' if model_name == 'enhanced' else 'efficient'
                
                # Create evaluator
                evaluator = ModelEvaluator(
                    model_path=model_path,
                    model_type=model_type,
                    data_module=data_module
                )
                
                # Evaluate on simple dataset
                results = evaluator.evaluate_dataset('simple')
                
                # Create visualization
                evaluator.create_prediction_visualization(
                    complexity='simple',
                    save_path=f'{model_name}_predictions.png'
                )
                
                print(f"✅ {model_name} model evaluation completed!")
                print(f"  📊 RMSE: {results['rmse']:.1f} m/s")
                
            else:
                print(f"⚠️  Model not found: {model_path}")
        
        print("\n🎉 Model evaluation completed!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
