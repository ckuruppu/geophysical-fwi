#!/usr/bin/env python3
"""
Scaled-up efficient seismic inversion model.
Balances computational efficiency with model capacity for better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class ScaledTemporalEncoder(nn.Module):
    """
    Scaled temporal encoder with increased capacity while maintaining efficiency.
    """
    
    def __init__(self, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Larger but still efficient LSTM
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Re-enable bidirectional for better capacity
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Lightweight attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=4,  # Fewer heads for efficiency
            dropout=0.1,
            batch_first=True
        )
        
        # Feature projection with residual connection
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficient forward pass with smart downsampling.
        
        Args:
            x: Input tensor of shape (batch, sources, time_steps, receivers)
            
        Returns:
            Temporal features of shape (batch, sources, receivers, hidden_size)
        """
        batch_size, n_sources, n_time, n_receivers = x.shape
        
        # Smart downsampling: reduce time and receivers for efficiency
        # Time: 1000 -> 200 (5x reduction)
        # Receivers: 70 -> 35 (2x reduction)
        time_step = max(1, n_time // 200)
        receiver_step = max(1, n_receivers // 35)
        
        x_downsampled = x[:, :, ::time_step, ::receiver_step]
        batch_size, n_sources, n_time_down, n_receivers_down = x_downsampled.shape
        
        # Reshape for LSTM processing
        x_reshaped = x_downsampled.permute(0, 1, 3, 2).reshape(-1, n_time_down, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_reshaped)  # (batch * sources * receivers_down, time_down, hidden_size * 2)
        
        # Apply lightweight attention (only on a subset for efficiency)
        if n_time_down > 50:  # Only use attention if sequence is long enough
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        else:
            attended_out = lstm_out
        
        # Global average pooling over time
        temporal_features = torch.mean(attended_out, dim=1)  # (batch * sources * receivers_down, hidden_size * 2)
        
        # Project features
        temporal_features = self.feature_projection(temporal_features)
        
        # Reshape back
        temporal_features = temporal_features.reshape(batch_size, n_sources, n_receivers_down, -1)
        
        # Interpolate back to full receiver count
        temporal_features = F.interpolate(
            temporal_features.permute(0, 3, 1, 2),  # (batch, hidden_size, sources, receivers_down)
            size=(n_sources, n_receivers),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # (batch, sources, receivers, hidden_size)
        
        # Apply layer normalization
        temporal_features = self.layer_norm(temporal_features)
        
        return temporal_features

class ScaledSpatialEncoder(nn.Module):
    """
    Scaled spatial encoder with increased capacity.
    """
    
    def __init__(self, input_channels: int = 64, output_channels: int = 128):
        super().__init__()
        
        # Multi-scale convolution blocks
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(input_channels, 64, kernel_size=3),
            self._make_conv_block(64, 96, kernel_size=5),
            self._make_conv_block(96, output_channels, kernel_size=7)
        ])
        
        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(output_channels, output_channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(output_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Residual connection
        self.residual_proj = nn.Conv1d(input_channels, output_channels, 1)
        
    def _make_conv_block(self, in_channels: int, out_channels: int, kernel_size: int) -> nn.Module:
        """Create a convolutional block with normalization and activation."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor of shape (batch, sources, receivers, features)
            
        Returns:
            Spatial features of shape (batch, sources, output_channels)
        """
        batch_size, n_sources, n_receivers, n_features = x.shape
        
        # Process all sources together for efficiency
        x_flat = x.reshape(batch_size * n_sources, n_receivers, n_features)
        x_flat = x_flat.permute(0, 2, 1)  # (batch * sources, features, receivers)
        
        # Store input for residual connection
        residual = self.residual_proj(x_flat)
        
        # Apply convolution blocks
        features = x_flat
        for conv_block in self.conv_blocks:
            features = conv_block(features)
        
        # Add residual connection
        features = features + residual
        
        # Apply spatial attention
        attention_weights = self.spatial_attention(features)
        features = features * attention_weights
        
        # Global average pooling
        pooled_features = torch.mean(features, dim=2)  # (batch * sources, output_channels)
        
        # Reshape back
        spatial_features = pooled_features.reshape(batch_size, n_sources, -1)
        
        return spatial_features

class ScaledSourceFusion(nn.Module):
    """
    Scaled source fusion with attention mechanism.
    """
    
    def __init__(self, feature_dim: int = 128, num_sources: int = 5):
        super().__init__()
        
        self.num_sources = num_sources
        
        # Source attention mechanism
        self.source_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced source fusion with attention.
        
        Args:
            x: Input tensor of shape (batch, sources, features)
            
        Returns:
            Fused features of shape (batch, features)
        """
        # Apply source attention
        attended_features, attention_weights = self.source_attention(x, x, x)
        
        # Weighted average across sources
        fused_features = torch.mean(attended_features, dim=1)  # (batch, features)
        
        # Apply fusion layers
        enhanced_features = self.fusion_layers(fused_features)
        
        # Residual connection and normalization
        fused_features = self.layer_norm(fused_features + enhanced_features)
        
        return fused_features

class ScaledVelocityDecoder(nn.Module):
    """
    Scaled velocity decoder with improved architecture.
    """
    
    def __init__(self, input_dim: int = 128, output_size: Tuple[int, int] = (70, 70)):
        super().__init__()
        
        self.output_size = output_size
        
        # Larger initial representation for better detail
        self.initial_size = 16
        self.initial_channels = 128
        
        # Project to initial spatial representation
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, self.initial_channels * self.initial_size * self.initial_size)
        )
        
        # Enhanced decoder layers with skip connections
        self.decoder_layers = nn.ModuleList([
            # 16x16 -> 32x32
            nn.ConvTranspose2d(self.initial_channels, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64x64 -> 70x70 (with padding adjustment)
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Final layer
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        ])
        
        # Skip connections for detail preservation
        self.skip_connections = nn.ModuleList([
            nn.Conv2d(self.initial_channels, 96, 1),
            nn.Conv2d(96, 64, 1),
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced decoding with skip connections.
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            Velocity model of shape (batch, 1, height, width)
        """
        batch_size = x.shape[0]
        
        # Project to spatial representation
        x = self.feature_projection(x)
        x = x.reshape(batch_size, self.initial_channels, self.initial_size, self.initial_size)
        
        # Store features for skip connections
        skip_features = [x]
        
        # Apply decoder layers with skip connections
        for i in range(0, len(self.decoder_layers), 3):
            if i + 2 < len(self.decoder_layers):
                # Apply conv, bn, relu
                x = self.decoder_layers[i](x)
                x = self.decoder_layers[i + 1](x)
                x = self.decoder_layers[i + 2](x)
                
                # Add skip connection if available
                skip_idx = i // 3
                if skip_idx < len(self.skip_connections) and skip_idx < len(skip_features):
                    skip = self.skip_connections[skip_idx](skip_features[skip_idx])
                    if skip.shape[-2:] == x.shape[-2:]:
                        x = x + skip
                
                skip_features.append(x)
            else:
                # Final layer
                x = self.decoder_layers[i](x)
        
        # Ensure exact output size
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        return x

class ScaledSeismicInversionNet(nn.Module):
    """
    Scaled-up efficient seismic inversion network.
    Balances computational efficiency with increased model capacity.
    """
    
    def __init__(self,
                 temporal_hidden_size: int = 64,
                 spatial_output_channels: int = 128,
                 num_sources: int = 5,
                 output_size: Tuple[int, int] = (70, 70)):
        super().__init__()
        
        # Scaled components
        self.temporal_encoder = ScaledTemporalEncoder(
            hidden_size=temporal_hidden_size,
            num_layers=2
        )
        
        self.spatial_encoder = ScaledSpatialEncoder(
            input_channels=temporal_hidden_size,
            output_channels=spatial_output_channels
        )
        
        self.source_fusion = ScaledSourceFusion(
            feature_dim=spatial_output_channels,
            num_sources=num_sources
        )
        
        self.velocity_decoder = ScaledVelocityDecoder(
            input_dim=spatial_output_channels,
            output_size=output_size
        )
        
    def forward(self, seismic_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the scaled network.
        
        Args:
            seismic_data: Input seismic data of shape (batch, sources, time_steps, receivers)
            
        Returns:
            Predicted velocity model of shape (batch, 1, height, width)
        """
        # Improved normalization
        mean = torch.mean(seismic_data, dim=(2, 3), keepdim=True)
        std = torch.std(seismic_data, dim=(2, 3), keepdim=True) + 1e-8
        normalized_data = (seismic_data - mean) / std
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(normalized_data)
        
        # Spatial encoding
        spatial_features = self.spatial_encoder(temporal_features)
        
        # Source fusion
        fused_features = self.source_fusion(spatial_features)
        
        # Velocity decoding
        velocity_model = self.velocity_decoder(fused_features)
        
        return velocity_model

def test_scaled_model():
    """Test the scaled model."""
    print("🧪 Testing Scaled Seismic Inversion Model...")
    
    model = ScaledSeismicInversionNet(
        temporal_hidden_size=64,
        spatial_output_channels=128,
        num_sources=5,
        output_size=(70, 70)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Scaled model created with {total_params:,} parameters")
    print(f"📊 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test forward pass
    x = torch.randn(2, 5, 1000, 70)
    print(f"  Input shape: {x.shape}")
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        output = model(x)
    
    forward_time = time.time() - start_time
    print(f"✅ Forward pass successful in {forward_time:.2f}s")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Parameter breakdown
    print("\n📊 Parameter breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} parameters")
    
    return model

if __name__ == "__main__":
    test_scaled_model()
