#!/usr/bin/env python3
"""
Hybrid CNN-RNN Architecture for Seismic Waveform Inversion
Designed specifically for the OpenFWI dataset structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class SeismicDataProcessor:
    """
    Preprocesses seismic data for neural network input.
    Handles feature extraction and normalization.
    """
    
    def __init__(self, sampling_rate: float = 250.0, max_freq: float = 50.0):
        self.sampling_rate = sampling_rate
        self.max_freq = max_freq
        self.dt = 1.0 / sampling_rate
        
    def extract_temporal_features(self, seismic_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract temporal features from seismic traces.
        
        Args:
            seismic_data: Shape (batch, sources, time_steps, receivers)
            
        Returns:
            Dictionary of temporal features
        """
        batch_size, n_sources, n_time, n_receivers = seismic_data.shape
        
        features = {}
        
        # 1. Envelope (instantaneous amplitude)
        analytic_signal = torch.fft.fft(seismic_data, dim=2)
        envelope = torch.abs(torch.fft.ifft(analytic_signal, dim=2))
        features['envelope'] = envelope
        
        # 2. First break detection (approximate)
        abs_data = torch.abs(seismic_data)
        threshold = 0.1 * torch.max(abs_data, dim=2, keepdim=True)[0]
        first_breaks = torch.argmax((abs_data > threshold).float(), dim=2)
        features['first_breaks'] = first_breaks.float()
        
        # 3. RMS amplitude in time windows
        window_size = 50  # samples
        n_windows = n_time // window_size
        windowed_data = seismic_data[:, :, :n_windows*window_size, :].reshape(
            batch_size, n_sources, n_windows, window_size, n_receivers
        )
        rms_amplitudes = torch.sqrt(torch.mean(windowed_data**2, dim=3))
        features['rms_amplitudes'] = rms_amplitudes
        
        # 4. Dominant frequency estimation
        fft_data = torch.fft.fft(seismic_data, dim=2)
        power_spectrum = torch.abs(fft_data)**2
        freqs = torch.fft.fftfreq(n_time, d=self.dt)
        
        # Find dominant frequency for each trace
        max_freq_idx = torch.argmax(power_spectrum[:, :, :n_time//2, :], dim=2)
        dominant_freqs = freqs[max_freq_idx]
        features['dominant_frequency'] = dominant_freqs
        
        return features
    
    def normalize_data(self, data: torch.Tensor, method: str = 'trace') -> torch.Tensor:
        """
        Normalize seismic data.
        
        Args:
            data: Input seismic data
            method: 'trace', 'global', or 'rms'
        """
        if method == 'trace':
            # Normalize each trace individually
            trace_max = torch.max(torch.abs(data), dim=2, keepdim=True)[0]
            trace_max = torch.clamp(trace_max, min=1e-8)  # Avoid division by zero
            return data / trace_max
            
        elif method == 'global':
            # Global normalization
            global_max = torch.max(torch.abs(data))
            return data / global_max
            
        elif method == 'rms':
            # RMS normalization
            rms = torch.sqrt(torch.mean(data**2, dim=2, keepdim=True))
            rms = torch.clamp(rms, min=1e-8)
            return data / rms
            
        return data

class TemporalEncoder(nn.Module):
    """
    RNN-based temporal encoder for processing seismic time series.
    Extracts temporal patterns and wave propagation characteristics.
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism for focusing on important time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal encoder.
        
        Args:
            x: Input tensor of shape (batch, sources, time_steps, receivers)
            
        Returns:
            Temporal features of shape (batch, sources, receivers, hidden_size)
        """
        batch_size, n_sources, n_time, n_receivers = x.shape
        
        # Reshape for LSTM processing: (batch * sources * receivers, time_steps, 1)
        x_reshaped = x.permute(0, 1, 3, 2).reshape(-1, n_time, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_reshaped)  # (batch * sources * receivers, time_steps, hidden_size * 2)
        
        # Apply attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over time dimension
        temporal_features = torch.mean(attended_out, dim=1)  # (batch * sources * receivers, hidden_size * 2)
        
        # Project features
        temporal_features = self.feature_projection(temporal_features)
        
        # Reshape back to original structure
        temporal_features = temporal_features.reshape(batch_size, n_sources, n_receivers, -1)
        
        return temporal_features

class SpatialEncoder(nn.Module):
    """
    CNN-based spatial encoder for processing receiver array data.
    Extracts spatial patterns and geological structures.
    """
    
    def __init__(self, input_channels: int = 64, output_channels: int = 128):
        super().__init__()
        
        # Multi-scale spatial processing
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(input_channels, 32, kernel_size=3),
            self._make_conv_block(32, 64, kernel_size=5),
            self._make_conv_block(64, output_channels, kernel_size=7)
        ])
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(output_channels, output_channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(output_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
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
        Forward pass through spatial encoder.
        
        Args:
            x: Input tensor of shape (batch, sources, receivers, features)
            
        Returns:
            Spatial features of shape (batch, sources, output_channels)
        """
        batch_size, n_sources, n_receivers, n_features = x.shape
        
        # Process each source separately
        source_features = []
        
        for source_idx in range(n_sources):
            # Get data for this source: (batch, receivers, features)
            source_data = x[:, source_idx, :, :].permute(0, 2, 1)  # (batch, features, receivers)
            
            # Apply convolutional blocks
            features = source_data
            for conv_block in self.conv_blocks:
                features = conv_block(features)
            
            # Apply spatial attention
            attention_weights = self.spatial_attention(features)
            features = features * attention_weights
            
            # Global average pooling over receivers
            pooled_features = torch.mean(features, dim=2)  # (batch, output_channels)
            source_features.append(pooled_features)
        
        # Stack source features
        spatial_features = torch.stack(source_features, dim=1)  # (batch, sources, output_channels)
        
        return spatial_features

class SourceFusionModule(nn.Module):
    """
    Fuses information from multiple seismic sources using attention mechanism.
    """

    def __init__(self, feature_dim: int = 128, num_sources: int = 5):
        super().__init__()

        self.num_sources = num_sources
        self.feature_dim = feature_dim

        # Source attention mechanism
        self.source_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fuse multi-source features.

        Args:
            x: Input tensor of shape (batch, sources, features)

        Returns:
            Fused features of shape (batch, features)
        """
        # Apply source attention
        attended_features, attention_weights = self.source_attention(x, x, x)

        # Weighted average across sources
        fused_features = torch.mean(attended_features, dim=1)  # (batch, features)

        # Apply fusion layer
        fused_features = self.fusion_layer(fused_features)

        return fused_features

class VelocityDecoder(nn.Module):
    """
    Decodes fused features into velocity models using transpose convolutions.
    """

    def __init__(self, input_dim: int = 128, output_size: Tuple[int, int] = (70, 70)):
        super().__init__()

        self.output_size = output_size
        self.input_dim = input_dim

        # Calculate initial spatial size for transpose convolutions
        # We'll start with a small spatial size and upsample
        self.initial_size = 8
        self.initial_channels = 256

        # Project input features to initial spatial representation
        self.feature_projection = nn.Linear(
            input_dim,
            self.initial_channels * self.initial_size * self.initial_size
        )

        # Transpose convolution layers for upsampling
        self.decoder_layers = nn.ModuleList([
            # 8x8 -> 16x16
            nn.ConvTranspose2d(self.initial_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 64x64 -> 70x70 (final adjustment)
            nn.ConvTranspose2d(32, 16, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Final velocity prediction
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        ])

        # Skip connections for preserving fine details
        self.skip_connections = nn.ModuleList([
            nn.Conv2d(self.initial_channels, 128, 1),
            nn.Conv2d(128, 64, 1),
            nn.Conv2d(64, 32, 1),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode features into velocity model.

        Args:
            x: Input features of shape (batch, input_dim)

        Returns:
            Velocity model of shape (batch, 1, height, width)
        """
        batch_size = x.shape[0]

        # Project to initial spatial representation
        x = self.feature_projection(x)
        x = x.reshape(batch_size, self.initial_channels, self.initial_size, self.initial_size)

        # Store features for skip connections
        skip_features = []

        # Apply decoder layers
        layer_idx = 0
        for i in range(0, len(self.decoder_layers), 3):  # Process in groups of 3 (conv, bn, relu)
            if i + 2 < len(self.decoder_layers):
                # Apply conv, bn, relu
                x = self.decoder_layers[i](x)
                x = self.decoder_layers[i + 1](x)
                x = self.decoder_layers[i + 2](x)

                # Store for skip connection (except for the last layer)
                if layer_idx < len(self.skip_connections):
                    skip_features.append(x)
                    layer_idx += 1
            else:
                # Final layer
                x = self.decoder_layers[i](x)

        # Ensure output size matches target
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)

        return x

class SeismicInversionNet(nn.Module):
    """
    Complete hybrid CNN-RNN architecture for seismic waveform inversion.
    """

    def __init__(self,
                 temporal_hidden_size: int = 64,
                 spatial_output_channels: int = 128,
                 num_sources: int = 5,
                 output_size: Tuple[int, int] = (70, 70)):
        super().__init__()

        # Data preprocessing
        self.data_processor = SeismicDataProcessor()

        # Temporal encoder (RNN)
        self.temporal_encoder = TemporalEncoder(
            input_size=1,
            hidden_size=temporal_hidden_size,
            num_layers=2,
            dropout=0.1
        )

        # Spatial encoder (CNN)
        self.spatial_encoder = SpatialEncoder(
            input_channels=temporal_hidden_size,
            output_channels=spatial_output_channels
        )

        # Source fusion
        self.source_fusion = SourceFusionModule(
            feature_dim=spatial_output_channels,
            num_sources=num_sources
        )

        # Velocity decoder
        self.velocity_decoder = VelocityDecoder(
            input_dim=spatial_output_channels,
            output_size=output_size
        )

    def forward(self, seismic_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete network.

        Args:
            seismic_data: Input seismic data of shape (batch, sources, time_steps, receivers)

        Returns:
            Predicted velocity model of shape (batch, 1, height, width)
        """
        # Normalize input data
        normalized_data = self.data_processor.normalize_data(seismic_data, method='trace')

        # Temporal encoding: Extract temporal features from each trace
        temporal_features = self.temporal_encoder(normalized_data)
        # Shape: (batch, sources, receivers, temporal_hidden_size)

        # Spatial encoding: Process receiver arrays
        spatial_features = self.spatial_encoder(temporal_features)
        # Shape: (batch, sources, spatial_output_channels)

        # Source fusion: Combine information from multiple sources
        fused_features = self.source_fusion(spatial_features)
        # Shape: (batch, spatial_output_channels)

        # Velocity decoding: Generate velocity model
        velocity_model = self.velocity_decoder(fused_features)
        # Shape: (batch, 1, height, width)

        return velocity_model
