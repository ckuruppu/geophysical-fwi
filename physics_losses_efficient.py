#!/usr/bin/env python3
"""
Efficient physics-informed loss functions for seismic waveform inversion.
Optimized for CPU training with reduced computational complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class EfficientPhysicsInformedLoss(nn.Module):
    """
    Efficient physics-informed loss function optimized for CPU training.
    Includes geological constraints and wave physics with reduced complexity.
    """
    
    def __init__(self, 
                 mse_weight: float = 1.0,
                 smoothness_weight: float = 0.1,
                 gradient_weight: float = 0.05,
                 geological_weight: float = 0.02,
                 velocity_range: Tuple[float, float] = (1500.0, 4500.0)):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight
        self.gradient_weight = gradient_weight
        self.geological_weight = geological_weight
        self.velocity_range = velocity_range
        
        # Efficient loss components
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, 
                velocity_pred: torch.Tensor, 
                velocity_target: torch.Tensor,
                seismic_data: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute efficient physics-informed loss.
        
        Args:
            velocity_pred: Predicted velocity model (batch, 1, H, W)
            velocity_target: Target velocity model (batch, 1, H, W)
            seismic_data: Input seismic data (optional, for advanced physics)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # 1. Primary reconstruction loss (MSE)
        losses['mse'] = self.mse_loss(velocity_pred, velocity_target)
        
        # 2. Smoothness constraint (geological realism)
        losses['smoothness'] = self._compute_smoothness_loss(velocity_pred)
        
        # 3. Gradient consistency (geological structure)
        losses['gradient'] = self._compute_gradient_loss(velocity_pred, velocity_target)
        
        # 4. Geological constraints (velocity bounds and trends)
        losses['geological'] = self._compute_geological_loss(velocity_pred)
        
        # 5. Total weighted loss
        losses['total'] = (
            self.mse_weight * losses['mse'] +
            self.smoothness_weight * losses['smoothness'] +
            self.gradient_weight * losses['gradient'] +
            self.geological_weight * losses['geological']
        )
        
        return losses
    
    def _compute_smoothness_loss(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness loss to encourage geological realism.
        Penalizes sharp velocity contrasts.
        """
        # Compute gradients in both spatial directions
        grad_x = torch.abs(velocity[:, :, :, 1:] - velocity[:, :, :, :-1])
        grad_y = torch.abs(velocity[:, :, 1:, :] - velocity[:, :, :-1, :])
        
        # L1 norm of gradients (more robust than L2)
        smoothness_x = torch.mean(grad_x)
        smoothness_y = torch.mean(grad_y)
        
        return smoothness_x + smoothness_y
    
    def _compute_gradient_loss(self, velocity_pred: torch.Tensor, velocity_target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient consistency loss.
        Ensures predicted gradients match target gradients.
        """
        # Compute gradients for both predicted and target
        pred_grad_x = velocity_pred[:, :, :, 1:] - velocity_pred[:, :, :, :-1]
        pred_grad_y = velocity_pred[:, :, 1:, :] - velocity_pred[:, :, :-1, :]
        
        target_grad_x = velocity_target[:, :, :, 1:] - velocity_target[:, :, :, :-1]
        target_grad_y = velocity_target[:, :, 1:, :] - velocity_target[:, :, :-1, :]
        
        # L1 loss between gradients
        grad_loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        grad_loss_y = self.l1_loss(pred_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y
    
    def _compute_geological_loss(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute geological constraint loss.
        Enforces realistic velocity ranges and depth trends.
        """
        batch_size, _, height, width = velocity.shape
        
        # 1. Velocity range constraint
        min_vel, max_vel = self.velocity_range
        range_penalty = torch.mean(
            F.relu(min_vel - velocity) + F.relu(velocity - max_vel)
        )
        
        # 2. Depth trend constraint (velocity generally increases with depth)
        # Create depth weights (higher values for deeper layers)
        depth_weights = torch.linspace(0.0, 1.0, height, device=velocity.device)
        depth_weights = depth_weights.view(1, 1, height, 1).expand_as(velocity)
        
        # Compute depth gradient (should be positive on average)
        depth_grad = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]
        depth_trend_penalty = torch.mean(F.relu(-depth_grad))  # Penalize negative gradients
        
        # 3. Lateral continuity (velocity should be relatively continuous laterally)
        lateral_grad = torch.abs(velocity[:, :, :, 1:] - velocity[:, :, :, :-1])
        lateral_penalty = torch.mean(F.relu(lateral_grad - 500.0))  # Penalize large lateral changes
        
        return range_penalty + 0.1 * depth_trend_penalty + 0.05 * lateral_penalty

class EfficientMetrics:
    """
    Efficient evaluation metrics for seismic inversion.
    """
    
    @staticmethod
    def compute_metrics(velocity_pred: torch.Tensor, 
                       velocity_target: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            velocity_pred: Predicted velocity model
            velocity_target: Target velocity model
            
        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            # Flatten tensors for easier computation
            pred_flat = velocity_pred.flatten()
            target_flat = velocity_target.flatten()
            
            # Basic metrics
            mae = torch.mean(torch.abs(pred_flat - target_flat)).item()
            mse = torch.mean((pred_flat - target_flat) ** 2).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()
            
            # Relative error
            target_range = target_flat.max() - target_flat.min()
            relative_error = (mae / target_range.item()) * 100 if target_range > 0 else 0.0
            
            # Correlation coefficient
            pred_centered = pred_flat - torch.mean(pred_flat)
            target_centered = target_flat - torch.mean(target_flat)
            correlation = torch.sum(pred_centered * target_centered) / (
                torch.sqrt(torch.sum(pred_centered ** 2)) * 
                torch.sqrt(torch.sum(target_centered ** 2))
            )
            correlation = correlation.item() if not torch.isnan(correlation) else 0.0
            
            # Structural similarity (simplified SSIM)
            ssim = EfficientMetrics._compute_simple_ssim(velocity_pred, velocity_target)
            
            # Velocity statistics
            pred_stats = {
                'min': pred_flat.min().item(),
                'max': pred_flat.max().item(),
                'mean': pred_flat.mean().item(),
                'std': pred_flat.std().item()
            }
            
            target_stats = {
                'min': target_flat.min().item(),
                'max': target_flat.max().item(),
                'mean': target_flat.mean().item(),
                'std': target_flat.std().item()
            }
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'relative_error': relative_error,
                'correlation': correlation,
                'ssim': ssim,
                'predicted_stats': pred_stats,
                'target_stats': target_stats
            }
    
    @staticmethod
    def _compute_simple_ssim(pred: torch.Tensor, target: torch.Tensor, 
                           window_size: int = 7) -> float:
        """
        Compute simplified structural similarity index.
        """
        # Convert to grayscale if needed and ensure proper dimensions
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # Remove channel dimension
            target = target.squeeze(1)
        
        # Simple local statistics
        mu_pred = F.avg_pool2d(pred.unsqueeze(1), window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target.unsqueeze(1), window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d((pred.unsqueeze(1) ** 2), window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d((target.unsqueeze(1) ** 2), window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d((pred.unsqueeze(1) * target.unsqueeze(1)), window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # SSIM computation
        numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
        denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        
        ssim_map = numerator / (denominator + 1e-8)
        return torch.mean(ssim_map).item()

def test_efficient_physics_loss():
    """Test the efficient physics-informed loss function."""
    print("🧪 Testing Efficient Physics-Informed Loss...")
    
    # Create test data
    batch_size = 2
    height, width = 70, 70
    
    velocity_pred = torch.randn(batch_size, 1, height, width) * 1000 + 3000
    velocity_target = torch.randn(batch_size, 1, height, width) * 1000 + 3000
    seismic_data = torch.randn(batch_size, 5, 1000, 70)
    
    # Create loss function
    loss_fn = EfficientPhysicsInformedLoss()
    
    # Compute losses
    import time
    start_time = time.time()
    losses = loss_fn(velocity_pred, velocity_target, seismic_data)
    loss_time = time.time() - start_time
    
    print(f"✅ Loss computation completed in {loss_time:.4f}s")
    print("📊 Loss components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")
    
    # Test metrics
    start_time = time.time()
    metrics = EfficientMetrics.compute_metrics(velocity_pred, velocity_target)
    metrics_time = time.time() - start_time
    
    print(f"✅ Metrics computation completed in {metrics_time:.4f}s")
    print("📈 Evaluation metrics:")
    for name, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {name}:")
            for k, v in value.items():
                print(f"    {k}: {v:.3f}")
        else:
            print(f"  {name}: {value:.6f}")
    
    print("🎉 Efficient physics loss testing completed!")

if __name__ == "__main__":
    test_efficient_physics_loss()
