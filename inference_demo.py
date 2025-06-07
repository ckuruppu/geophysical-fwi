#!/usr/bin/env python3
"""
Inference demonstration script for the trained seismic inversion model.
Shows model predictions on test samples with detailed visualizations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

from seismic_model_fixed import FixedSeismicInversionNet
from physics_losses_efficient import EfficientMetrics

class SeismicInversionInference:
    """
    Inference engine for the trained seismic inversion model.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        
        print("🔄 Loading trained model for inference...")
        self.model = self._load_model()
        self.model.eval()
        
        print("✅ Inference engine ready!")
    
    def _load_model(self) -> FixedSeismicInversionNet:
        """Load the trained model."""
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
        
        # Create and load model
        model = FixedSeismicInversionNet(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def predict_velocity_model(self, seismic_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Predict velocity model from seismic data.
        
        Args:
            seismic_data: Seismic data array of shape (sources, time, receivers)
            
        Returns:
            Tuple of (predicted_velocity, prediction_info)
        """
        start_time = time.time()
        
        # Ensure correct input shape
        if len(seismic_data.shape) == 2:  # (time, receivers)
            seismic_data = seismic_data[np.newaxis, :, :]  # Add source dimension
        
        # Convert to tensor
        seismic_tensor = torch.from_numpy(seismic_data).float().unsqueeze(0).to(self.device)
        
        # Model prediction
        with torch.no_grad():
            velocity_pred = self.model(seismic_tensor)
            velocity_pred_np = velocity_pred.cpu().numpy().squeeze()
        
        inference_time = time.time() - start_time
        
        # Prediction info
        pred_info = {
            'inference_time': inference_time,
            'input_shape': seismic_data.shape,
            'output_shape': velocity_pred_np.shape,
            'velocity_range': (velocity_pred_np.min(), velocity_pred_np.max()),
            'velocity_mean': velocity_pred_np.mean(),
            'velocity_std': velocity_pred_np.std()
        }
        
        return velocity_pred_np, pred_info
    
    def run_inference_demo(self, data_root: str, num_samples: int = 3):
        """
        Run inference demonstration on test samples.
        """
        print(f"\n🎯 RUNNING INFERENCE DEMO ON {num_samples} SAMPLES")
        print("="*55)
        
        data_root = Path(data_root)
        scenarios = ['FlatVel_A', 'Style_A', 'FlatFault_A']
        
        demo_results = []
        
        for i, scenario in enumerate(scenarios[:num_samples]):
            print(f"\n🏔️  Demo {i+1}: {scenario}")
            scenario_path = data_root / scenario
            
            if not scenario_path.exists():
                print(f"⚠️  Scenario {scenario} not found, skipping...")
                continue
            
            # Load sample data
            sample_data = self._load_sample_data(scenario_path)
            if sample_data is None:
                continue
            
            seismic_data, velocity_target = sample_data
            
            # Run inference
            velocity_pred, pred_info = self.predict_velocity_model(seismic_data)
            
            # Compute metrics if target is available
            metrics = None
            if velocity_target is not None:
                # Convert to tensors for metrics computation
                pred_tensor = torch.from_numpy(velocity_pred).unsqueeze(0).unsqueeze(0)
                target_tensor = torch.from_numpy(velocity_target).unsqueeze(0).unsqueeze(0)
                metrics = EfficientMetrics.compute_metrics(pred_tensor, target_tensor)
            
            # Store results
            demo_results.append({
                'scenario': scenario,
                'seismic_data': seismic_data,
                'velocity_target': velocity_target,
                'velocity_pred': velocity_pred,
                'pred_info': pred_info,
                'metrics': metrics
            })
            
            # Print results
            print(f"  ✅ Inference completed in {pred_info['inference_time']:.3f}s")
            print(f"  📊 Predicted velocity range: [{pred_info['velocity_range'][0]:.0f}, {pred_info['velocity_range'][1]:.0f}] m/s")
            print(f"  📊 Mean velocity: {pred_info['velocity_mean']:.0f} m/s")
            
            if metrics:
                print(f"  📊 RMSE: {metrics['rmse']:.1f} m/s")
                print(f"  📊 Correlation: {metrics['correlation']:.3f}")
                print(f"  📊 SSIM: {metrics['ssim']:.3f}")
        
        # Create comprehensive visualization
        if demo_results:
            self._create_inference_visualization(demo_results)
        
        return demo_results
    
    def _load_sample_data(self, scenario_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a sample from the scenario."""
        try:
            # Check scenario structure
            data_dir = scenario_path / "data"
            model_dir = scenario_path / "model"
            
            if data_dir.exists() and model_dir.exists():
                # Structure: scenario/data/*.npy, scenario/model/*.npy
                data_files = list(data_dir.glob("*.npy"))
                model_files = list(model_dir.glob("*.npy"))
                
                if data_files and model_files:
                    seismic_data = np.load(data_files[0], mmap_mode='r')
                    velocity_target = np.load(model_files[0], mmap_mode='r')
            else:
                # Structure: scenario/*.npy files directly
                seis_files = list(scenario_path.glob("seis*.npy"))
                vel_files = list(scenario_path.glob("vel*.npy"))
                
                if seis_files and vel_files:
                    seismic_data = np.load(seis_files[0], mmap_mode='r')
                    velocity_target = np.load(vel_files[0], mmap_mode='r')
                else:
                    return None
            
            # Handle batched data
            if len(seismic_data.shape) == 4:  # Batched data
                seismic_sample = seismic_data[0]  # Take first sample
                velocity_sample = velocity_target[0]
            else:
                seismic_sample = seismic_data
                velocity_sample = velocity_target
            
            # Ensure correct shapes
            if len(seismic_sample.shape) == 2:  # (time, receivers)
                seismic_sample = seismic_sample[np.newaxis, :, :]  # Add source dimension
            
            if len(velocity_sample.shape) == 3 and velocity_sample.shape[0] > 1:
                velocity_sample = velocity_sample[0, :, :]  # Take first channel
            elif len(velocity_sample.shape) == 3:
                velocity_sample = velocity_sample.squeeze()
            
            # Clamp velocity to realistic range
            velocity_sample = np.clip(velocity_sample, 1000, 6000)
            
            return seismic_sample, velocity_sample
            
        except Exception as e:
            print(f"    ⚠️  Error loading sample: {e}")
            return None
    
    def _create_inference_visualization(self, demo_results: List[Dict]):
        """Create comprehensive inference visualization."""
        print(f"\n🎨 Creating inference visualization...")
        
        n_samples = len(demo_results)
        fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(demo_results):
            scenario = result['scenario']
            seismic_data = result['seismic_data']
            velocity_target = result['velocity_target']
            velocity_pred = result['velocity_pred']
            metrics = result['metrics']
            
            # 1. Seismic data (first source)
            seismic_plot = seismic_data[0, :, :] if len(seismic_data.shape) == 3 else seismic_data
            im1 = axes[i, 0].imshow(seismic_plot, aspect='auto', cmap='seismic')
            axes[i, 0].set_title(f'{scenario}\nSeismic Data (Source 1)')
            axes[i, 0].set_xlabel('Receiver')
            axes[i, 0].set_ylabel('Time')
            plt.colorbar(im1, ax=axes[i, 0], label='Amplitude')
            
            # 2. Target velocity model
            if velocity_target is not None:
                im2 = axes[i, 1].imshow(velocity_target, cmap='jet', vmin=1500, vmax=4500)
                axes[i, 1].set_title(f'{scenario}\nTrue Velocity Model')
                axes[i, 1].set_xlabel('X')
                axes[i, 1].set_ylabel('Depth')
                plt.colorbar(im2, ax=axes[i, 1], label='Velocity (m/s)')
            else:
                axes[i, 1].text(0.5, 0.5, 'No Target\nAvailable', 
                               ha='center', va='center', transform=axes[i, 1].transAxes)
                axes[i, 1].set_title(f'{scenario}\nNo Target Available')
            
            # 3. Predicted velocity model
            im3 = axes[i, 2].imshow(velocity_pred, cmap='jet', vmin=1500, vmax=4500)
            axes[i, 2].set_title(f'{scenario}\nPredicted Velocity Model')
            axes[i, 2].set_xlabel('X')
            axes[i, 2].set_ylabel('Depth')
            plt.colorbar(im3, ax=axes[i, 2], label='Velocity (m/s)')
            
            # 4. Comparison plot (if target available)
            if velocity_target is not None and metrics:
                # Depth profile comparison
                center_x = velocity_target.shape[1] // 2
                depth = np.arange(velocity_target.shape[0])
                
                target_profile = velocity_target[:, center_x]
                pred_profile = velocity_pred[:, center_x]
                
                axes[i, 3].plot(target_profile, depth, 'b-', linewidth=2, label='True')
                axes[i, 3].plot(pred_profile, depth, 'r--', linewidth=2, label='Predicted')
                axes[i, 3].set_xlabel('Velocity (m/s)')
                axes[i, 3].set_ylabel('Depth')
                axes[i, 3].set_title(f'{scenario}\nDepth Profile Comparison\n'
                                    f'RMSE: {metrics["rmse"]:.1f} m/s\n'
                                    f'Corr: {metrics["correlation"]:.3f}')
                axes[i, 3].legend()
                axes[i, 3].invert_yaxis()
                axes[i, 3].grid(True, alpha=0.3)
            else:
                # Show prediction statistics
                pred_stats = f"""Prediction Statistics:
Range: [{velocity_pred.min():.0f}, {velocity_pred.max():.0f}] m/s
Mean: {velocity_pred.mean():.0f} m/s
Std: {velocity_pred.std():.0f} m/s"""
                
                axes[i, 3].text(0.1, 0.5, pred_stats, 
                               ha='left', va='center', transform=axes[i, 3].transAxes,
                               fontfamily='monospace')
                axes[i, 3].set_title(f'{scenario}\nPrediction Statistics')
                axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        save_path = Path('inference_results')
        save_path.mkdir(exist_ok=True)
        
        plt.savefig(save_path / 'inference_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Inference visualization saved to: {save_path}/inference_demo.png")

def main():
    """Main inference demonstration."""
    print("🎯 SEISMIC INVERSION MODEL - INFERENCE DEMO")
    print("="*50)
    
    # Check for trained model
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
        # Initialize inference engine
        inference_engine = SeismicInversionInference(model_path)
        
        # Run inference demonstration
        demo_results = inference_engine.run_inference_demo(
            data_root="waveform-inversion/train_samples",
            num_samples=3
        )
        
        if demo_results:
            print(f"\n🎉 INFERENCE DEMO COMPLETED!")
            print(f"📊 Processed {len(demo_results)} samples")
            
            # Summary statistics
            avg_inference_time = np.mean([r['pred_info']['inference_time'] for r in demo_results])
            
            valid_metrics = [r['metrics'] for r in demo_results if r['metrics']]
            if valid_metrics:
                avg_rmse = np.mean([m['rmse'] for m in valid_metrics])
                avg_correlation = np.mean([m['correlation'] for m in valid_metrics])
                
                print(f"📊 Average inference time: {avg_inference_time:.3f}s")
                print(f"📊 Average RMSE: {avg_rmse:.1f} m/s")
                print(f"📊 Average correlation: {avg_correlation:.3f}")
            
            print(f"🎨 Visualization saved to: inference_results/inference_demo.png")
        
    except Exception as e:
        print(f"❌ Inference demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
