#!/usr/bin/env python3
"""
Simplified training script for debugging and CPU training.
Focuses on identifying bottlenecks and ensuring the model works.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from pathlib import Path

from seismic_inversion_model import SeismicInversionNet
from physics_informed_losses import PhysicsInformedLoss, compute_all_metrics
from seismic_data_loader import SeismicDataModule

def create_simple_model():
    """Create a much smaller model for testing."""
    print("🔧 Creating simplified model...")
    
    # Much smaller model for CPU training
    model = SeismicInversionNet(
        temporal_hidden_size=16,  # Reduced from 64
        spatial_output_channels=32,  # Reduced from 128
        num_sources=5,
        output_size=(70, 70)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Simplified model created with {total_params:,} parameters")
    print(f"📊 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model

def test_model_forward_pass(model, batch_size=1):
    """Test the model forward pass with synthetic data."""
    print(f"🧪 Testing model forward pass with batch size {batch_size}...")
    
    # Create small synthetic batch
    synthetic_data = torch.randn(batch_size, 5, 1000, 70)
    print(f"  Input shape: {synthetic_data.shape}")
    
    model.eval()
    start_time = time.time()
    
    try:
        with torch.no_grad():
            output = model(synthetic_data)
        
        forward_time = time.time() - start_time
        print(f"✅ Forward pass successful in {forward_time:.2f}s")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True, forward_time
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_loss_computation(model):
    """Test loss computation."""
    print("🧪 Testing loss computation...")
    
    batch_size = 1
    synthetic_seismic = torch.randn(batch_size, 5, 1000, 70)
    synthetic_velocity = torch.randn(batch_size, 1, 70, 70) * 1000 + 3000  # Realistic velocity range
    
    model.eval()
    criterion = PhysicsInformedLoss()
    
    try:
        with torch.no_grad():
            velocity_pred = model(synthetic_seismic)
            losses = criterion(velocity_pred, synthetic_velocity, synthetic_seismic)
        
        print("✅ Loss computation successful")
        print("  Loss components:")
        for name, value in losses.items():
            print(f"    {name}: {value.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading with minimal settings."""
    print("🧪 Testing data loading...")
    
    try:
        data_module = SeismicDataModule(
            data_root="waveform-inversion",
            batch_size=1,  # Single sample
            val_split=0.2,
            num_workers=0,  # No multiprocessing
            augment_train=False
        )
        
        train_loader, val_loader = data_module.get_dataloaders('simple')
        print(f"✅ Data loading successful")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        
        # Test loading one batch
        for seismic_batch, velocity_batch in train_loader:
            print(f"  Sample batch shapes: {seismic_batch.shape}, {velocity_batch.shape}")
            return True, train_loader, val_loader
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_single_training_step(model, train_loader):
    """Test a single training step to identify bottlenecks."""
    print("🧪 Testing single training step...")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Smaller learning rate
    criterion = PhysicsInformedLoss()
    
    try:
        for seismic_data, velocity_target in train_loader:
            print(f"  📊 Batch shapes: {seismic_data.shape}, {velocity_target.shape}")
            
            # Forward pass timing
            print("  🔄 Forward pass...")
            start_time = time.time()
            optimizer.zero_grad()
            velocity_pred = model(seismic_data)
            forward_time = time.time() - start_time
            print(f"    ✅ Forward pass: {forward_time:.2f}s")
            
            # Loss computation timing
            print("  ⚖️  Loss computation...")
            start_time = time.time()
            losses = criterion(velocity_pred, velocity_target, seismic_data)
            loss = losses['total']
            loss_time = time.time() - start_time
            print(f"    ✅ Loss computation: {loss_time:.2f}s")
            print(f"    📊 Loss value: {loss.item():.6f}")
            
            # Backward pass timing
            print("  🔙 Backward pass...")
            start_time = time.time()
            loss.backward()
            backward_time = time.time() - start_time
            print(f"    ✅ Backward pass: {backward_time:.2f}s")
            
            # Optimizer step timing
            print("  🎯 Optimizer step...")
            start_time = time.time()
            optimizer.step()
            step_time = time.time() - start_time
            print(f"    ✅ Optimizer step: {step_time:.2f}s")
            
            total_time = forward_time + loss_time + backward_time + step_time
            print(f"  ⏱️  Total step time: {total_time:.2f}s")
            
            return True
            
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_mini_training(model, train_loader, num_steps=3):
    """Run a few training steps to test stability."""
    print(f"🏃 Running mini training ({num_steps} steps)...")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = PhysicsInformedLoss()
    
    try:
        step_count = 0
        for seismic_data, velocity_target in train_loader:
            if step_count >= num_steps:
                break
                
            print(f"  Step {step_count + 1}/{num_steps}")
            
            start_time = time.time()
            
            optimizer.zero_grad()
            velocity_pred = model(seismic_data)
            losses = criterion(velocity_pred, velocity_target, seismic_data)
            loss = losses['total']
            loss.backward()
            optimizer.step()
            
            step_time = time.time() - start_time
            print(f"    Loss: {loss.item():.6f}, Time: {step_time:.2f}s")
            
            step_count += 1
        
        print("✅ Mini training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Mini training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debugging function."""
    print("🔍 SIMPLIFIED TRAINING SCRIPT - DEBUGGING MODE")
    print("="*60)
    
    # Test 1: Create simplified model
    model = create_simple_model()
    
    # Test 2: Test forward pass with synthetic data
    success, forward_time = test_model_forward_pass(model, batch_size=1)
    if not success:
        print("❌ Forward pass test failed. Exiting.")
        return
    
    # Test 3: Test loss computation
    success = test_loss_computation(model)
    if not success:
        print("❌ Loss computation test failed. Exiting.")
        return
    
    # Test 4: Test data loading
    success, train_loader, val_loader = test_data_loading()
    if not success:
        print("❌ Data loading test failed. Exiting.")
        return
    
    # Test 5: Test single training step
    success = test_single_training_step(model, train_loader)
    if not success:
        print("❌ Single training step test failed. Exiting.")
        return
    
    # Test 6: Run mini training
    success = run_mini_training(model, train_loader, num_steps=3)
    if not success:
        print("❌ Mini training test failed. Exiting.")
        return
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ The model architecture works correctly")
    print("✅ Training pipeline is functional")
    print("✅ Ready for full training with larger model")
    
    print(f"\n📊 Performance Summary:")
    print(f"  Forward pass time: {forward_time:.2f}s per sample")
    print(f"  Estimated time for 100 samples: {forward_time * 100:.1f}s")
    print(f"  Recommended batch size for CPU: 1-2")

if __name__ == "__main__":
    main()
