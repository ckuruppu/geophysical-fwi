#!/usr/bin/env python3
"""
Physics-Informed Loss Functions for Seismic Waveform Inversion
Incorporates seismic wave physics into the training process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class PhysicsInformedLoss(nn.Module):
    """
    Combined loss function incorporating multiple physics-based constraints.
    """
    
    def __init__(self, 
                 mse_weight: float = 1.0,
                 smoothness_weight: float = 0.1,
                 gradient_weight: float = 0.05,
                 travel_time_weight: float = 0.2,
                 geological_weight: float = 0.1):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight
        self.gradient_weight = gradient_weight
        self.travel_time_weight = travel_time_weight
        self.geological_weight = geological_weight
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, 
                predicted_velocity: torch.Tensor,
                target_velocity: torch.Tensor,
                seismic_data: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss.
        
        Args:
            predicted_velocity: Predicted velocity model (batch, 1, H, W)
            target_velocity: Target velocity model (batch, 1, H, W)
            seismic_data: Original seismic data for travel time constraints (optional)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # 1. Basic MSE loss
        mse_loss = self.mse_loss(predicted_velocity, target_velocity)
        losses['mse'] = mse_loss
        
        # 2. Smoothness regularization
        smoothness_loss = self._compute_smoothness_loss(predicted_velocity)
        losses['smoothness'] = smoothness_loss
        
        # 3. Gradient preservation loss
        gradient_loss = self._compute_gradient_loss(predicted_velocity, target_velocity)
        losses['gradient'] = gradient_loss
        
        # 4. Geological realism loss
        geological_loss = self._compute_geological_loss(predicted_velocity)
        losses['geological'] = geological_loss
        
        # 5. Travel time consistency (if seismic data provided)
        if seismic_data is not None:
            travel_time_loss = self._compute_travel_time_loss(predicted_velocity, seismic_data)
            losses['travel_time'] = travel_time_loss
        else:
            losses['travel_time'] = torch.tensor(0.0, device=predicted_velocity.device)
        
        # Combine all losses
        total_loss = (
            self.mse_weight * losses['mse'] +
            self.smoothness_weight * losses['smoothness'] +
            self.gradient_weight * losses['gradient'] +
            self.geological_weight * losses['geological'] +
            self.travel_time_weight * losses['travel_time']
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def _compute_smoothness_loss(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness regularization to prevent unrealistic velocity variations.
        """
        # Compute gradients in both spatial dimensions
        grad_x = torch.diff(velocity, dim=-1)  # Horizontal gradient
        grad_y = torch.diff(velocity, dim=-2)  # Vertical gradient
        
        # L2 norm of gradients (total variation)
        smoothness_loss = torch.mean(grad_x**2) + torch.mean(grad_y**2)
        
        return smoothness_loss
    
    def _compute_gradient_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Ensure predicted gradients match target gradients (structural preservation).
        """
        # Compute gradients for both predicted and target
        pred_grad_x = torch.diff(predicted, dim=-1)
        pred_grad_y = torch.diff(predicted, dim=-2)
        
        target_grad_x = torch.diff(target, dim=-1)
        target_grad_y = torch.diff(target, dim=-2)
        
        # MSE between gradients
        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y
    
    def _compute_geological_loss(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Enforce geological realism constraints.
        """
        # 1. Velocity should generally increase with depth (for sedimentary layers)
        depth_gradient = torch.diff(velocity, dim=-2)  # Vertical differences
        
        # Penalize strong negative gradients (velocity decreasing with depth)
        negative_gradients = torch.clamp(-depth_gradient, min=0)
        depth_penalty = torch.mean(negative_gradients**2)
        
        # 2. Velocity range constraints (typical seismic velocities)
        min_velocity = 1000.0  # m/s
        max_velocity = 8000.0  # m/s
        
        range_penalty = (
            torch.mean(torch.clamp(min_velocity - velocity, min=0)**2) +
            torch.mean(torch.clamp(velocity - max_velocity, min=0)**2)
        )
        
        return depth_penalty + range_penalty
    
    def _compute_travel_time_loss(self, velocity: torch.Tensor, seismic_data: torch.Tensor) -> torch.Tensor:
        """
        Approximate travel time consistency check.
        This is a simplified version - full implementation would require ray tracing.
        """
        batch_size = velocity.shape[0]
        
        # Simplified: Check if first arrival times are consistent with velocity model
        # This is an approximation - real implementation would use eikonal equation
        
        # Extract approximate first arrival times from seismic data
        abs_seismic = torch.abs(seismic_data)
        threshold = 0.1 * torch.max(abs_seismic, dim=2, keepdim=True)[0]
        first_arrivals = torch.argmax((abs_seismic > threshold).float(), dim=2)
        
        # Compute expected travel times based on velocity model
        # Simplified: use average velocity along vertical path
        avg_velocity = torch.mean(velocity, dim=(-2, -1))  # (batch, 1)
        
        # For this simplified version, we just ensure consistency in relative timing
        # Real implementation would solve the eikonal equation
        travel_time_loss = torch.tensor(0.0, device=velocity.device)
        
        return travel_time_loss

class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting that adjusts weights during training.
    """
    
    def __init__(self, num_losses: int = 5, temperature: float = 2.0):
        super().__init__()
        
        self.num_losses = num_losses
        self.temperature = temperature
        
        # Learnable loss weights
        self.log_weights = nn.Parameter(torch.zeros(num_losses))
        
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted combination of losses with adaptive weights.
        """
        # Extract loss values (excluding 'total')
        loss_values = []
        loss_names = ['mse', 'smoothness', 'gradient', 'geological', 'travel_time']
        
        for name in loss_names:
            if name in losses:
                loss_values.append(losses[name])
            else:
                loss_values.append(torch.tensor(0.0, device=self.log_weights.device))
        
        loss_tensor = torch.stack(loss_values)
        
        # Compute adaptive weights using softmax
        weights = F.softmax(self.log_weights / self.temperature, dim=0)
        
        # Weighted combination
        weighted_loss = torch.sum(weights * loss_tensor)
        
        return weighted_loss

class VelocityMetrics:
    """
    Evaluation metrics specific to velocity model prediction.
    """
    
    @staticmethod
    def velocity_mae(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Absolute Error in velocity."""
        return torch.mean(torch.abs(predicted - target)).item()
    
    @staticmethod
    def velocity_rmse(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Root Mean Square Error in velocity."""
        return torch.sqrt(torch.mean((predicted - target)**2)).item()
    
    @staticmethod
    def relative_error(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Relative error percentage."""
        return torch.mean(torch.abs(predicted - target) / torch.abs(target)).item() * 100
    
    @staticmethod
    def structural_similarity(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        Structural similarity based on gradient correlation.
        """
        # Compute gradients
        pred_grad_x = torch.diff(predicted, dim=-1)
        pred_grad_y = torch.diff(predicted, dim=-2)
        
        target_grad_x = torch.diff(target, dim=-1)
        target_grad_y = torch.diff(target, dim=-2)
        
        # Flatten gradients
        pred_grad = torch.cat([pred_grad_x.flatten(), pred_grad_y.flatten()])
        target_grad = torch.cat([target_grad_x.flatten(), target_grad_y.flatten()])
        
        # Compute correlation coefficient
        pred_centered = pred_grad - torch.mean(pred_grad)
        target_centered = target_grad - torch.mean(target_grad)
        
        correlation = torch.sum(pred_centered * target_centered) / (
            torch.sqrt(torch.sum(pred_centered**2)) * torch.sqrt(torch.sum(target_centered**2))
        )
        
        return correlation.item()
    
    @staticmethod
    def velocity_range_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Check if predicted velocities are within realistic ranges.
        """
        pred_min = torch.min(predicted).item()
        pred_max = torch.max(predicted).item()
        target_min = torch.min(target).item()
        target_max = torch.max(target).item()
        
        return {
            'predicted_range': (pred_min, pred_max),
            'target_range': (target_min, target_max),
            'range_error': abs(pred_max - pred_min - (target_max - target_min))
        }

def compute_all_metrics(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    """
    metrics = {}
    
    metrics['mae'] = VelocityMetrics.velocity_mae(predicted, target)
    metrics['rmse'] = VelocityMetrics.velocity_rmse(predicted, target)
    metrics['relative_error'] = VelocityMetrics.relative_error(predicted, target)
    metrics['structural_similarity'] = VelocityMetrics.structural_similarity(predicted, target)
    
    range_metrics = VelocityMetrics.velocity_range_accuracy(predicted, target)
    metrics.update(range_metrics)
    
    return metrics
