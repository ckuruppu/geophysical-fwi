#!/usr/bin/env python3
"""
Efficient version of the seismic inversion model optimized for CPU training.
Reduces computational complexity while maintaining the hybrid CNN-RNN concept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class EfficientTemporalEncoder(nn.Module):
    """
    Efficient temporal encoder that reduces computational complexity.
    """
    
    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Much smaller LSTM - single layer, smaller hidden size
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False  # Unidirectional for speed
        )
        
        # Simple feature projection instead of attention
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficient forward pass.
        
        Args:
            x: Input tensor of shape (batch, sources, time_steps, receivers)
            
        Returns:
            Temporal features of shape (batch, sources, receivers, hidden_size)
        """
        batch_size, n_sources, n_time, n_receivers = x.shape
        
        # Process fewer traces to reduce computation
        # Sample every 5th receiver to reduce from 70 to 14 receivers
        sampled_receivers = torch.arange(0, n_receivers, 5)
        x_sampled = x[:, :, :, sampled_receivers]  # (batch, sources, time, 14)
        
        # Downsample time dimension by factor of 4 (1000 -> 250)
        x_downsampled = x_sampled[:, :, ::4, :]  # (batch, sources, 250, 14)
        
        batch_size, n_sources, n_time_down, n_receivers_down = x_downsampled.shape
        
        # Reshape for LSTM processing
        x_reshaped = x_downsampled.permute(0, 1, 3, 2).reshape(-1, n_time_down, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_reshaped)  # (batch * sources * receivers_down, time_down, hidden_size)
        
        # Take only the last output (no attention for speed)
        temporal_features = lstm_out[:, -1, :]  # (batch * sources * receivers_down, hidden_size)
        
        # Project features
        temporal_features = self.feature_projection(temporal_features)
        
        # Reshape back and interpolate to original receiver count
        temporal_features = temporal_features.reshape(batch_size, n_sources, n_receivers_down, -1)
        
        # Interpolate back to full receiver count
        temporal_features = F.interpolate(
            temporal_features.permute(0, 3, 1, 2),  # (batch, hidden_size, sources, receivers_down)
            size=(n_sources, n_receivers),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # (batch, sources, receivers, hidden_size)
        
        return temporal_features

class EfficientSpatialEncoder(nn.Module):
    """
    Efficient spatial encoder with reduced complexity.
    """
    
    def __init__(self, input_channels: int = 32, output_channels: int = 64):
        super().__init__()
        
        # Simpler conv blocks
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficient forward pass.
        
        Args:
            x: Input tensor of shape (batch, sources, receivers, features)
            
        Returns:
            Spatial features of shape (batch, sources, output_channels)
        """
        batch_size, n_sources, n_receivers, n_features = x.shape
        
        # Process all sources together for efficiency
        x_flat = x.reshape(batch_size * n_sources, n_receivers, n_features)
        x_flat = x_flat.permute(0, 2, 1)  # (batch * sources, features, receivers)
        
        # Apply convolutions
        features = self.relu(self.conv1(x_flat))
        features = self.dropout(features)
        features = self.relu(self.conv2(features))
        
        # Global average pooling
        pooled_features = torch.mean(features, dim=2)  # (batch * sources, output_channels)
        
        # Reshape back
        spatial_features = pooled_features.reshape(batch_size, n_sources, -1)
        
        return spatial_features

class EfficientSourceFusion(nn.Module):
    """
    Efficient source fusion without attention.
    """
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        
        # Simple fusion without attention
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple source fusion.
        
        Args:
            x: Input tensor of shape (batch, sources, features)
            
        Returns:
            Fused features of shape (batch, features)
        """
        # Simple average across sources
        fused_features = torch.mean(x, dim=1)  # (batch, features)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(fused_features)
        
        return fused_features

class EfficientVelocityDecoder(nn.Module):
    """
    Efficient velocity decoder with reduced complexity.
    """
    
    def __init__(self, input_dim: int = 64, output_size: Tuple[int, int] = (70, 70)):
        super().__init__()
        
        self.output_size = output_size
        
        # Much simpler decoder
        self.initial_size = 16  # Start larger
        self.initial_channels = 64  # Fewer channels
        
        # Project to initial spatial representation
        self.feature_projection = nn.Linear(
            input_dim,
            self.initial_channels * self.initial_size * self.initial_size
        )
        
        # Simpler decoder layers
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(self.initial_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Final layer to get exact size
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficient decoding.
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            Velocity model of shape (batch, 1, height, width)
        """
        batch_size = x.shape[0]
        
        # Project to spatial representation
        x = self.feature_projection(x)
        x = x.reshape(batch_size, self.initial_channels, self.initial_size, self.initial_size)
        
        # Apply decoder
        x = self.decoder(x)
        
        # Interpolate to exact output size if needed
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        return x

class EfficientSeismicInversionNet(nn.Module):
    """
    Efficient version of the hybrid CNN-RNN architecture.
    Optimized for CPU training with reduced computational complexity.
    """
    
    def __init__(self,
                 temporal_hidden_size: int = 32,
                 spatial_output_channels: int = 64,
                 num_sources: int = 5,
                 output_size: Tuple[int, int] = (70, 70)):
        super().__init__()
        
        # Efficient components
        self.temporal_encoder = EfficientTemporalEncoder(
            hidden_size=temporal_hidden_size,
            num_layers=1
        )
        
        self.spatial_encoder = EfficientSpatialEncoder(
            input_channels=temporal_hidden_size,
            output_channels=spatial_output_channels
        )
        
        self.source_fusion = EfficientSourceFusion(
            feature_dim=spatial_output_channels
        )
        
        self.velocity_decoder = EfficientVelocityDecoder(
            input_dim=spatial_output_channels,
            output_size=output_size
        )
        
    def forward(self, seismic_data: torch.Tensor) -> torch.Tensor:
        """
        Efficient forward pass.
        
        Args:
            seismic_data: Input seismic data of shape (batch, sources, time_steps, receivers)
            
        Returns:
            Predicted velocity model of shape (batch, 1, height, width)
        """
        # Simple normalization
        normalized_data = seismic_data / (torch.std(seismic_data, dim=2, keepdim=True) + 1e-8)
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(normalized_data)
        
        # Spatial encoding
        spatial_features = self.spatial_encoder(temporal_features)
        
        # Source fusion
        fused_features = self.source_fusion(spatial_features)
        
        # Velocity decoding
        velocity_model = self.velocity_decoder(fused_features)
        
        return velocity_model

def test_efficient_model():
    """Test the efficient model."""
    print("🧪 Testing efficient model...")
    
    model = EfficientSeismicInversionNet(
        temporal_hidden_size=16,
        spatial_output_channels=32,
        num_sources=5,
        output_size=(70, 70)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Efficient model created with {total_params:,} parameters")
    print(f"📊 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test forward pass
    x = torch.randn(1, 5, 1000, 70)
    print(f"  Input shape: {x.shape}")
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        output = model(x)
    
    forward_time = time.time() - start_time
    print(f"✅ Forward pass successful in {forward_time:.2f}s")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return model

if __name__ == "__main__":
    test_efficient_model()
