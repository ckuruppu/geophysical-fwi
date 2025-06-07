#!/usr/bin/env python3
"""
Ultra-simple debugging script to isolate the hanging issue.
"""

import torch
import time
import sys

def test_basic_torch():
    """Test basic PyTorch functionality."""
    print("🧪 Testing basic PyTorch...")
    try:
        x = torch.randn(10, 10)
        y = torch.mm(x, x.t())
        print(f"✅ Basic PyTorch works: {y.shape}")
        return True
    except Exception as e:
        print(f"❌ Basic PyTorch failed: {e}")
        return False

def test_model_components():
    """Test individual model components."""
    print("🧪 Testing model components...")
    
    try:
        # Test LSTM
        print("  Testing LSTM...")
        lstm = torch.nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        x = torch.randn(1, 100, 1)  # Small input
        output, _ = lstm(x)
        print(f"    ✅ LSTM works: {output.shape}")
        
        # Test Conv1D
        print("  Testing Conv1D...")
        conv = torch.nn.Conv1d(16, 32, kernel_size=3, padding=1)
        x = torch.randn(1, 16, 70)
        output = conv(x)
        print(f"    ✅ Conv1D works: {output.shape}")
        
        # Test ConvTranspose2D
        print("  Testing ConvTranspose2D...")
        conv_t = torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        x = torch.randn(1, 32, 8, 8)
        output = conv_t(x)
        print(f"    ✅ ConvTranspose2D works: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model components failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_model():
    """Test a minimal version of our model."""
    print("🧪 Testing minimal model...")
    
    try:
        class MinimalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(1, 8, batch_first=True)
                self.conv = torch.nn.Conv1d(8, 16, 3, padding=1)
                self.fc = torch.nn.Linear(16, 70*70)
                
            def forward(self, x):
                # x: (batch, sources, time, receivers)
                batch_size, n_sources, n_time, n_receivers = x.shape
                
                # Process first source only for simplicity
                x_source = x[:, 0, :, 0].unsqueeze(-1)  # (batch, time, 1)
                
                # LSTM
                lstm_out, _ = self.lstm(x_source)  # (batch, time, 8)
                
                # Take mean over time
                features = torch.mean(lstm_out, dim=1)  # (batch, 8)
                
                # Simple linear projection
                output = self.fc(features)  # (batch, 70*70)
                output = output.view(batch_size, 1, 70, 70)
                
                return output
        
        model = MinimalModel()
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        print("  Testing forward pass...")
        x = torch.randn(1, 5, 100, 70)  # Much smaller time dimension
        
        start_time = time.time()
        with torch.no_grad():
            output = model(x)
        forward_time = time.time() - start_time
        
        print(f"    ✅ Minimal model works: {output.shape}")
        print(f"    ⏱️  Forward time: {forward_time:.3f}s")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Minimal model failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_data_loading_simple():
    """Test very simple data loading."""
    print("🧪 Testing simple data loading...")
    
    try:
        import numpy as np
        from pathlib import Path
        
        # Check if data exists
        data_path = Path("waveform-inversion/train_samples/FlatVel_A")
        if not data_path.exists():
            print(f"❌ Data path not found: {data_path}")
            return False, None, None
        
        # Load one file
        data_files = list((data_path / "data").glob("*.npy"))
        model_files = list((data_path / "model").glob("*.npy"))
        
        if not data_files or not model_files:
            print(f"❌ No data files found")
            return False, None, None
        
        print(f"  Loading {data_files[0]} and {model_files[0]}")
        
        seismic_data = np.load(data_files[0])
        velocity_data = np.load(model_files[0])
        
        print(f"  ✅ Data loaded: seismic {seismic_data.shape}, velocity {velocity_data.shape}")
        
        # Convert to tensors
        seismic_tensor = torch.from_numpy(seismic_data[0]).float()  # First sample
        velocity_tensor = torch.from_numpy(velocity_data[0]).float()  # First sample
        
        print(f"  ✅ Tensors created: seismic {seismic_tensor.shape}, velocity {velocity_tensor.shape}")
        
        return True, seismic_tensor.unsqueeze(0), velocity_tensor.unsqueeze(0)
        
    except Exception as e:
        print(f"❌ Simple data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_full_pipeline():
    """Test the complete pipeline with minimal model and real data."""
    print("🧪 Testing full pipeline...")
    
    # Get minimal model
    success, model = test_minimal_model()
    if not success:
        return False
    
    # Get real data
    success, seismic_data, velocity_data = test_data_loading_simple()
    if not success:
        print("  Using synthetic data instead...")
        seismic_data = torch.randn(1, 5, 100, 70)
        velocity_data = torch.randn(1, 1, 70, 70) * 1000 + 3000
    
    try:
        print("  Testing training step...")
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        # Forward pass
        print("    Forward pass...")
        start_time = time.time()
        optimizer.zero_grad()
        output = model(seismic_data)
        forward_time = time.time() - start_time
        print(f"      ✅ Forward: {forward_time:.3f}s")
        
        # Loss computation
        print("    Loss computation...")
        start_time = time.time()
        loss = criterion(output, velocity_data)
        loss_time = time.time() - start_time
        print(f"      ✅ Loss: {loss_time:.3f}s, value: {loss.item():.6f}")
        
        # Backward pass
        print("    Backward pass...")
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        print(f"      ✅ Backward: {backward_time:.3f}s")
        
        # Optimizer step
        print("    Optimizer step...")
        start_time = time.time()
        optimizer.step()
        step_time = time.time() - start_time
        print(f"      ✅ Step: {step_time:.3f}s")
        
        total_time = forward_time + loss_time + backward_time + step_time
        print(f"    ⏱️  Total: {total_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debugging function."""
    print("🔍 ULTRA-SIMPLE DEBUGGING SCRIPT")
    print("="*50)
    
    # Test 1: Basic PyTorch
    if not test_basic_torch():
        print("❌ Basic PyTorch failed. Check installation.")
        return
    
    # Test 2: Model components
    if not test_model_components():
        print("❌ Model components failed.")
        return
    
    # Test 3: Minimal model
    if not test_minimal_model():
        print("❌ Minimal model failed.")
        return
    
    # Test 4: Full pipeline
    if not test_full_pipeline():
        print("❌ Full pipeline failed.")
        return
    
    print("\n🎉 ALL DEBUGGING TESTS PASSED!")
    print("✅ PyTorch is working correctly")
    print("✅ Model components are functional")
    print("✅ Training pipeline works")
    print("\n💡 The issue might be in the complex model architecture.")
    print("   Try reducing model size or using GPU for full model.")

if __name__ == "__main__":
    main()
