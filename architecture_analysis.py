#!/usr/bin/env python3
"""
Comprehensive analysis of the neural network architecture and physics-informed loss.
Identifies issues causing extremely high losses and poor performance.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from seismic_model_efficient import EfficientSeismicInversionNet
from physics_losses_efficient import EfficientPhysicsInformedLoss, EfficientMetrics

def analyze_model_architecture():
    """Analyze the neural network architecture in detail."""
    print("🔍 NEURAL NETWORK ARCHITECTURE ANALYSIS")
    print("="*60)
    
    # Create model
    model = EfficientSeismicInversionNet(
        temporal_hidden_size=16,
        spatial_output_channels=32,
        num_sources=5,
        output_size=(70, 70)
    )
    
    print("🏗️  Model Architecture:")
    print(f"  Input: (batch, 5 sources, 1000 time_steps, 70 receivers)")
    print(f"  Output: (batch, 1, 70, 70) velocity model")
    
    # Analyze each component
    total_params = 0
    
    print(f"\n📊 Component Analysis:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        print(f"  {name}: {params:,} parameters")
        
        # Detailed analysis for each component
        if name == 'temporal_encoder':
            print(f"    - Processes (5×70=350) sequences of length 1000")
            print(f"    - LSTM hidden size: 16")
            print(f"    - Downsamples time: 1000→250, receivers: 70→35")
            print(f"    - Output: (batch, 5, 70, 16)")
            
        elif name == 'spatial_encoder':
            print(f"    - Input: (batch, 5, 70, 16)")
            print(f"    - Conv1D processing across receivers")
            print(f"    - Output: (batch, 5, 32)")
            
        elif name == 'source_fusion':
            print(f"    - Input: (batch, 5, 32)")
            print(f"    - Averages across 5 sources")
            print(f"    - Output: (batch, 32)")
            
        elif name == 'velocity_decoder':
            print(f"    - Input: (batch, 32)")
            print(f"    - Projects to spatial representation")
            print(f"    - Output: (batch, 1, 70, 70)")
    
    print(f"\n📊 Total Parameters: {total_params:,}")
    
    return model

def analyze_data_flow():
    """Analyze data flow through the network with real shapes."""
    print(f"\n🔄 DATA FLOW ANALYSIS")
    print("="*40)
    
    model = EfficientSeismicInversionNet(
        temporal_hidden_size=16,
        spatial_output_channels=32,
        num_sources=5,
        output_size=(70, 70)
    )
    
    # Create sample input
    batch_size = 2
    input_data = torch.randn(batch_size, 5, 1000, 70)
    print(f"🔢 Input shape: {input_data.shape}")
    print(f"  Input range: [{input_data.min():.3f}, {input_data.max():.3f}]")
    print(f"  Input std: {input_data.std():.3f}")
    
    model.eval()
    with torch.no_grad():
        # Step 1: Normalization
        normalized_data = input_data / (torch.std(input_data, dim=2, keepdim=True) + 1e-8)
        print(f"\n1️⃣  After normalization: {normalized_data.shape}")
        print(f"    Range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
        print(f"    Std: {normalized_data.std():.3f}")
        
        # Step 2: Temporal encoding
        temporal_features = model.temporal_encoder(normalized_data)
        print(f"\n2️⃣  After temporal encoding: {temporal_features.shape}")
        print(f"    Range: [{temporal_features.min():.3f}, {temporal_features.max():.3f}]")
        print(f"    Std: {temporal_features.std():.3f}")
        
        # Step 3: Spatial encoding
        spatial_features = model.spatial_encoder(temporal_features)
        print(f"\n3️⃣  After spatial encoding: {spatial_features.shape}")
        print(f"    Range: [{spatial_features.min():.3f}, {spatial_features.max():.3f}]")
        print(f"    Std: {spatial_features.std():.3f}")
        
        # Step 4: Source fusion
        fused_features = model.source_fusion(spatial_features)
        print(f"\n4️⃣  After source fusion: {fused_features.shape}")
        print(f"    Range: [{fused_features.min():.3f}, {fused_features.max():.3f}]")
        print(f"    Std: {fused_features.std():.3f}")
        
        # Step 5: Velocity decoding
        velocity_output = model.velocity_decoder(fused_features)
        print(f"\n5️⃣  Final output: {velocity_output.shape}")
        print(f"    Range: [{velocity_output.min():.3f}, {velocity_output.max():.3f}]")
        print(f"    Std: {velocity_output.std():.3f}")
        print(f"    Mean: {velocity_output.mean():.3f}")

def analyze_physics_loss():
    """Analyze the physics-informed loss components."""
    print(f"\n⚖️  PHYSICS-INFORMED LOSS ANALYSIS")
    print("="*50)
    
    # Create realistic test data
    batch_size = 2
    
    # Target velocities (realistic seismic velocities in m/s)
    velocity_target = torch.zeros(batch_size, 1, 70, 70)
    for i in range(70):  # Depth-dependent velocity
        velocity_target[:, :, i, :] = 1500 + i * 30  # 1500-3600 m/s
    
    # Add some noise and structure
    velocity_target += torch.randn_like(velocity_target) * 100
    
    print(f"🎯 Target velocity statistics:")
    print(f"  Shape: {velocity_target.shape}")
    print(f"  Range: [{velocity_target.min():.1f}, {velocity_target.max():.1f}] m/s")
    print(f"  Mean: {velocity_target.mean():.1f} m/s")
    print(f"  Std: {velocity_target.std():.1f} m/s")
    
    # Predicted velocities (what our model might output initially)
    velocity_pred = torch.randn_like(velocity_target) * 2000 + 2500  # Random around 2500 m/s
    
    print(f"\n🔮 Predicted velocity statistics:")
    print(f"  Shape: {velocity_pred.shape}")
    print(f"  Range: [{velocity_pred.min():.1f}, {velocity_pred.max():.1f}] m/s")
    print(f"  Mean: {velocity_pred.mean():.1f} m/s")
    print(f"  Std: {velocity_pred.std():.1f} m/s")
    
    # Analyze loss components
    loss_fn = EfficientPhysicsInformedLoss(
        mse_weight=1.0,
        smoothness_weight=0.1,
        gradient_weight=0.05,
        geological_weight=0.02,
        velocity_range=(1500.0, 4500.0)
    )
    
    losses = loss_fn(velocity_pred, velocity_target)
    
    print(f"\n📊 Loss Component Analysis:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.2e}")
        
        if name == 'mse':
            # Analyze MSE in detail
            mse_per_pixel = torch.mean((velocity_pred - velocity_target) ** 2, dim=(0, 1))
            print(f"    MSE per pixel range: [{mse_per_pixel.min():.2e}, {mse_per_pixel.max():.2e}]")
            print(f"    RMSE: {torch.sqrt(value).item():.1f} m/s")
            
        elif name == 'smoothness':
            # Analyze smoothness
            grad_x = torch.abs(velocity_pred[:, :, :, 1:] - velocity_pred[:, :, :, :-1])
            grad_y = torch.abs(velocity_pred[:, :, 1:, :] - velocity_pred[:, :, :-1, :])
            print(f"    Avg horizontal gradient: {torch.mean(grad_x).item():.1f} m/s")
            print(f"    Avg vertical gradient: {torch.mean(grad_y).item():.1f} m/s")
    
    # Compute metrics
    metrics = EfficientMetrics.compute_metrics(velocity_pred, velocity_target)
    
    print(f"\n📈 Performance Metrics:")
    for name, value in metrics.items():
        if name not in ['predicted_stats', 'target_stats']:
            print(f"  {name}: {value:.6f}")

def identify_issues():
    """Identify specific issues causing high losses."""
    print(f"\n🚨 ISSUE IDENTIFICATION")
    print("="*40)
    
    issues = []
    
    # Issue 1: Output range mismatch
    model = EfficientSeismicInversionNet(temporal_hidden_size=16, spatial_output_channels=32)
    input_data = torch.randn(1, 5, 1000, 70)
    
    with torch.no_grad():
        output = model(input_data)
    
    if output.min() < 0 or output.max() > 10000:
        issues.append("❌ Output range issue: Model outputs unrealistic velocities")
        print(f"  Model output range: [{output.min():.1f}, {output.max():.1f}]")
        print(f"  Expected range: [1500, 4500] m/s")
    
    # Issue 2: No output activation
    decoder_layers = list(model.velocity_decoder.children())
    has_final_activation = any(isinstance(layer, (nn.Sigmoid, nn.Tanh, nn.ReLU)) for layer in decoder_layers[-3:])
    
    if not has_final_activation:
        issues.append("❌ No output activation: Model can output any range")
        print(f"  Final decoder layers: {[type(layer).__name__ for layer in decoder_layers[-3:]]}")
    
    # Issue 3: Loss scale mismatch
    target_vel = torch.ones(1, 1, 70, 70) * 3000  # 3000 m/s
    pred_vel = torch.ones(1, 1, 70, 70) * 2000    # 2000 m/s (1000 m/s error)
    
    mse_loss = nn.MSELoss()(pred_vel, target_vel)
    print(f"\n📊 Loss scale analysis:")
    print(f"  1000 m/s error → MSE loss: {mse_loss.item():.2e}")
    print(f"  This explains why losses are in millions!")
    
    if mse_loss.item() > 100000:
        issues.append("❌ Loss scale issue: MSE on velocity values is too large")
    
    # Issue 4: Normalization issues
    if abs(output.mean()) > 1000:
        issues.append("❌ Output not normalized: Large mean values")
    
    print(f"\n🔧 IDENTIFIED ISSUES:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    return issues

def propose_solutions():
    """Propose solutions to fix the identified issues."""
    print(f"\n💡 PROPOSED SOLUTIONS")
    print("="*40)
    
    solutions = [
        "1️⃣  Add output activation to constrain velocity range:",
        "   - Use Sigmoid + scaling: output = 1500 + 3000 * sigmoid(x)",
        "   - Or use Tanh + scaling: output = 3000 + 1500 * tanh(x)",
        "",
        "2️⃣  Normalize loss computation:",
        "   - Scale velocities to [0, 1] range before computing MSE",
        "   - Or use relative MSE: MSE / (target_mean^2)",
        "",
        "3️⃣  Adjust loss weights:",
        "   - Reduce MSE weight or increase physics weights",
        "   - Current: MSE=1.0, others=0.02-0.1",
        "   - Suggested: MSE=0.1, others=0.01-0.05",
        "",
        "4️⃣  Add proper initialization:",
        "   - Initialize final layer to output realistic velocities",
        "   - Use Xavier/He initialization for better convergence",
        "",
        "5️⃣  Implement progressive training:",
        "   - Start with simple MSE loss",
        "   - Gradually add physics constraints",
        "   - Use curriculum learning"
    ]
    
    for solution in solutions:
        print(solution)

def main():
    """Main analysis function."""
    print("🔍 COMPREHENSIVE ARCHITECTURE & LOSS ANALYSIS")
    print("="*60)
    
    # Analyze model architecture
    model = analyze_model_architecture()
    
    # Analyze data flow
    analyze_data_flow()
    
    # Analyze physics loss
    analyze_physics_loss()
    
    # Identify issues
    issues = identify_issues()
    
    # Propose solutions
    propose_solutions()
    
    print(f"\n🎯 SUMMARY:")
    print(f"  - Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  - Main issue: No output range constraint")
    print(f"  - Loss scale: Velocities in thousands → MSE in millions")
    print(f"  - Solution: Add output activation + normalize losses")

if __name__ == "__main__":
    main()
