#!/usr/bin/env python3
"""
Demo script to test the hybrid CNN-RNN architecture for seismic waveform inversion.
This script demonstrates the model architecture and runs a quick test with parallelism and debug output.
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from seismic_inversion_model import SeismicInversionNet, SeismicDataProcessor
from seismic_data_loader import SeismicDataModule, get_data_statistics
from physics_informed_losses import PhysicsInformedLoss, compute_all_metrics

# Set multiprocessing start method
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

def test_model_architecture():
    """
    Test the hybrid CNN-RNN architecture with synthetic data.
    """
    print("="*60)
    print("TESTING HYBRID CNN-RNN ARCHITECTURE")
    print("="*60)

    print("🔧 Creating model...")
    start_time = time.time()

    # Create model
    model = SeismicInversionNet(
        temporal_hidden_size=64,
        spatial_output_channels=128,
        num_sources=5,
        output_size=(70, 70)
    )

    print(f"✅ Model created in {time.time() - start_time:.2f}s")

    # Count parameters
    print("📊 Analyzing model parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model Architecture Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")

    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Move model to device
    print(f"🚀 Moving model to {device}...")
    model.to(device)

    # Test with synthetic data
    batch_size = 2  # Reduced for faster testing
    n_sources = 5
    n_time = 1000
    n_receivers = 70

    print(f"\n🎯 Testing with synthetic data:")
    print(f"  Input shape: ({batch_size}, {n_sources}, {n_time}, {n_receivers})")
    print(f"  Memory requirement: ~{batch_size * n_sources * n_time * n_receivers * 4 / 1024**2:.1f} MB")

    # Generate realistic synthetic seismic data with progress
    print("🔄 Generating synthetic seismic data...")
    start_time = time.time()
    synthetic_seismic = generate_synthetic_seismic_data_parallel(batch_size, n_sources, n_time, n_receivers)
    synthetic_seismic = synthetic_seismic.to(device)
    print(f"✅ Synthetic data generated in {time.time() - start_time:.2f}s")

    # Forward pass with detailed timing
    print(f"\n🧠 Running forward pass...")
    model.eval()

    # Warm up (first run is often slower)
    print("  🔥 Warming up model...")
    with torch.no_grad():
        _ = model(synthetic_seismic[:1])  # Single sample warmup

    # Actual timing
    print("  ⏱️  Measuring inference time...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with torch.no_grad():
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_cpu = time.time()

        predicted_velocity = model(synthetic_seismic)

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event)
        else:
            inference_time = (time.time() - start_cpu) * 1000  # Convert to ms

    print(f"✅ Forward pass completed!")
    print(f"  Output shape: {predicted_velocity.shape}")
    print(f"  Expected shape: ({batch_size}, 1, 70, 70)")
    print(f"  Inference time: {inference_time:.2f} ms")
    print(f"  Throughput: {batch_size / (inference_time / 1000):.1f} samples/sec")

    # Check output statistics
    print("📈 Analyzing output statistics...")
    pred_min = torch.min(predicted_velocity).item()
    pred_max = torch.max(predicted_velocity).item()
    pred_mean = torch.mean(predicted_velocity).item()
    pred_std = torch.std(predicted_velocity).item()

    print(f"Output Statistics:")
    print(f"  Min velocity: {pred_min:.1f} m/s")
    print(f"  Max velocity: {pred_max:.1f} m/s")
    print(f"  Mean velocity: {pred_mean:.1f} m/s")
    print(f"  Std velocity: {pred_std:.1f} m/s")

    # Check for reasonable velocity range
    if 1000 <= pred_mean <= 6000:
        print("  ✅ Velocity range looks realistic for seismic data")
    else:
        print("  ⚠️  Velocity range may need adjustment")

    return model, synthetic_seismic, predicted_velocity

def generate_synthetic_seismic_data_parallel(batch_size, n_sources, n_time, n_receivers):
    """
    Generate realistic synthetic seismic data for testing with parallel processing.
    """
    print(f"  📊 Generating {batch_size} samples with {n_sources} sources each...")

    def generate_single_sample(args):
        batch_idx, n_sources, n_time, n_receivers = args

        # Create time axis
        dt = 0.004  # 4ms sampling
        t = torch.arange(n_time) * dt

        # Create offset axis (receiver positions)
        offsets = torch.linspace(0, 3500, n_receivers)  # 0-3.5km offset range

        sample_data = torch.zeros(n_sources, n_time, n_receivers)

        for source in range(n_sources):
            # Source position (varies by source)
            source_pos = source * 700  # Sources every 700m

            for receiver in range(n_receivers):
                offset = abs(offsets[receiver] - source_pos)

                # Simulate direct arrival
                velocity = 2000 + batch_idx * 200  # Vary velocity by batch
                travel_time = offset / velocity

                if travel_time < t[-1]:
                    # Find time index
                    time_idx = int(travel_time / dt)
                    if time_idx < n_time - 50:
                        # Create Ricker wavelet
                        freq = 25  # 25 Hz dominant frequency
                        wavelet_time = t[time_idx:time_idx+50] - travel_time
                        ricker = (1 - 2 * (np.pi * freq * wavelet_time)**2) * torch.exp(-(np.pi * freq * wavelet_time)**2)

                        # Add to trace with geometric spreading
                        amplitude = 1.0 / (1 + offset / 1000)  # Geometric spreading
                        sample_data[source, time_idx:time_idx+50, receiver] += amplitude * ricker

                # Add some noise
                noise_level = 0.05
                sample_data[source, :, receiver] += noise_level * torch.randn(n_time)

        return sample_data

    # Use ThreadPoolExecutor for parallel generation
    with ThreadPoolExecutor(max_workers=min(4, batch_size)) as executor:
        # Prepare arguments for each batch
        args_list = [(batch_idx, n_sources, n_time, n_receivers) for batch_idx in range(batch_size)]

        # Submit all tasks
        futures = [executor.submit(generate_single_sample, args) for args in args_list]

        # Collect results with progress
        synthetic_data = []
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            synthetic_data.append(result)
            print(f"    ✅ Sample {i+1}/{batch_size} completed")

    # Stack all samples
    synthetic_batch = torch.stack(synthetic_data, dim=0)
    print(f"  🎯 Generated synthetic data shape: {synthetic_batch.shape}")

    return synthetic_batch

def generate_synthetic_seismic_data(batch_size, n_sources, n_time, n_receivers):
    """
    Generate realistic synthetic seismic data for testing.
    """
    # Create time axis
    dt = 0.004  # 4ms sampling
    t = torch.arange(n_time) * dt
    
    # Create offset axis (receiver positions)
    offsets = torch.linspace(0, 3500, n_receivers)  # 0-3.5km offset range
    
    synthetic_data = torch.zeros(batch_size, n_sources, n_time, n_receivers)
    
    for batch in range(batch_size):
        for source in range(n_sources):
            # Source position (varies by source)
            source_pos = source * 700  # Sources every 700m
            
            for receiver in range(n_receivers):
                offset = abs(offsets[receiver] - source_pos)
                
                # Simulate direct arrival
                velocity = 2000 + batch * 200  # Vary velocity by batch
                travel_time = offset / velocity
                
                if travel_time < t[-1]:
                    # Find time index
                    time_idx = int(travel_time / dt)
                    if time_idx < n_time - 50:
                        # Create Ricker wavelet
                        freq = 25  # 25 Hz dominant frequency
                        wavelet_time = t[time_idx:time_idx+50] - travel_time
                        ricker = (1 - 2 * (np.pi * freq * wavelet_time)**2) * torch.exp(-(np.pi * freq * wavelet_time)**2)
                        
                        # Add to trace with geometric spreading
                        amplitude = 1.0 / (1 + offset / 1000)  # Geometric spreading
                        synthetic_data[batch, source, time_idx:time_idx+50, receiver] += amplitude * ricker
                
                # Add some noise
                noise_level = 0.05
                synthetic_data[batch, source, :, receiver] += noise_level * torch.randn(n_time)
    
    return synthetic_data

def test_data_loading():
    """
    Test data loading functionality with detailed progress.
    """
    print("\n" + "="*60)
    print("TESTING DATA LOADING")
    print("="*60)

    data_root = "waveform-inversion"

    print(f"🔍 Checking data directory: {data_root}")
    if not Path(data_root).exists():
        print(f"❌ Data directory not found: {data_root}")
        print("Please ensure the OpenFWI dataset is available.")
        return None

    print(f"✅ Data directory found")

    # Check available families
    train_samples_dir = Path(data_root) / "train_samples"
    if train_samples_dir.exists():
        families = [d.name for d in train_samples_dir.iterdir() if d.is_dir()]
        print(f"📁 Available data families: {families}")
    else:
        print(f"❌ Train samples directory not found")
        return None

    # Test data statistics
    print("\n📊 Computing data statistics...")
    start_time = time.time()
    try:
        stats = get_data_statistics(data_root)
        print(f"✅ Statistics computed in {time.time() - start_time:.2f}s")
        print("Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
    except Exception as e:
        print(f"❌ Error computing statistics: {e}")
        return None

    # Test data module
    print("\n🔧 Creating data module...")
    data_module = SeismicDataModule(
        data_root=data_root,
        batch_size=2,  # Smaller batch for testing
        val_split=0.2,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        augment_train=False
    )

    try:
        print("📦 Getting dataloaders...")
        start_time = time.time()
        train_loader, val_loader = data_module.get_dataloaders('simple')
        print(f"✅ Dataloaders created in {time.time() - start_time:.2f}s")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")

        # Test loading one batch
        print("\n🔄 Loading test batch...")
        start_time = time.time()
        for i, (seismic_batch, velocity_batch) in enumerate(train_loader):
            load_time = time.time() - start_time
            print(f"✅ Batch {i+1} loaded in {load_time:.2f}s")
            print(f"  Seismic shape: {seismic_batch.shape}")
            print(f"  Velocity shape: {velocity_batch.shape}")
            print(f"  Seismic range: [{seismic_batch.min():.3f}, {seismic_batch.max():.3f}]")
            print(f"  Velocity range: [{velocity_batch.min():.1f}, {velocity_batch.max():.1f}] m/s")
            return seismic_batch, velocity_batch

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_loss_functions(model, seismic_data, velocity_target=None):
    """
    Test physics-informed loss functions with detailed progress.
    """
    print("\n" + "="*60)
    print("TESTING PHYSICS-INFORMED LOSS FUNCTIONS")
    print("="*60)

    # Generate prediction
    print("🧠 Generating model prediction...")
    start_time = time.time()
    model.eval()
    device = next(model.parameters()).device
    seismic_data = seismic_data.to(device)

    with torch.no_grad():
        velocity_pred = model(seismic_data)

    pred_time = time.time() - start_time
    print(f"✅ Prediction generated in {pred_time:.2f}s")
    print(f"  Prediction shape: {velocity_pred.shape}")

    # Create synthetic target if not provided
    if velocity_target is None:
        print("🎯 Creating synthetic target velocity...")
        velocity_target = torch.randn_like(velocity_pred) * 500 + 3000  # Realistic velocity range
        velocity_target = velocity_target.to(device)
        print(f"  Target range: [{velocity_target.min():.1f}, {velocity_target.max():.1f}] m/s")
    else:
        velocity_target = velocity_target.to(device)
        print(f"🎯 Using provided target velocity")
        print(f"  Target range: [{velocity_target.min():.1f}, {velocity_target.max():.1f}] m/s")

    # Test loss function
    print("\n⚖️  Creating physics-informed loss function...")
    criterion = PhysicsInformedLoss(
        mse_weight=1.0,
        smoothness_weight=0.1,
        gradient_weight=0.05,
        travel_time_weight=0.2,
        geological_weight=0.1
    )

    print("🔄 Computing loss components...")
    start_time = time.time()
    losses = criterion(velocity_pred, velocity_target, seismic_data)
    loss_time = time.time() - start_time
    print(f"✅ Loss computation completed in {loss_time:.3f}s")

    print("\nLoss Components:")
    total_loss = 0
    for loss_name, loss_value in losses.items():
        loss_val = loss_value.item()
        print(f"  {loss_name}: {loss_val:.6f}")
        if loss_name != 'total':
            total_loss += loss_val

    print(f"\n📊 Loss Analysis:")
    print(f"  Largest component: {max(losses.items(), key=lambda x: x[1].item() if x[0] != 'total' else 0)[0]}")
    print(f"  Total loss: {losses['total'].item():.6f}")

    # Test metrics
    print("\n📈 Computing evaluation metrics...")
    start_time = time.time()
    metrics = compute_all_metrics(velocity_pred, velocity_target)
    metrics_time = time.time() - start_time
    print(f"✅ Metrics computed in {metrics_time:.3f}s")

    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"  {metric_name}: {metric_value:.4f}")
        elif isinstance(metric_value, tuple):
            print(f"  {metric_name}: {metric_value}")

    # Performance summary
    print(f"\n⚡ Performance Summary:")
    print(f"  Model inference: {pred_time:.3f}s")
    print(f"  Loss computation: {loss_time:.3f}s")
    print(f"  Metrics computation: {metrics_time:.3f}s")
    print(f"  Total evaluation time: {pred_time + loss_time + metrics_time:.3f}s")

def visualize_architecture_test(seismic_data, velocity_pred):
    """
    Create visualization of the architecture test with progress tracking.
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)

    print("🎨 Setting up visualization...")

    # Move data to CPU for plotting
    if hasattr(seismic_data, 'cpu'):
        seismic_data = seismic_data.cpu()
    if hasattr(velocity_pred, 'cpu'):
        velocity_pred = velocity_pred.cpu()

    print("📊 Creating subplot layout...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    print("🔄 Plotting seismic data...")
    # Plot input seismic data (first source, first sample)
    seismic_sample = seismic_data[0, 0, :, :].numpy()  # (time, receivers)
    im1 = axes[0, 0].imshow(seismic_sample.T, aspect='auto', cmap='RdBu')
    axes[0, 0].set_title('Input Seismic Data\n(First Source)')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Receiver Number')
    plt.colorbar(im1, ax=axes[0, 0])

    print("🔄 Plotting seismic trace...")
    # Plot single trace
    trace = seismic_sample[:, 35]  # Middle receiver
    axes[0, 1].plot(trace)
    axes[0, 1].set_title('Sample Seismic Trace\n(Middle Receiver)')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)

    print("🔄 Computing frequency spectrum...")
    # Plot frequency spectrum of trace
    fft_trace = np.fft.fft(trace)
    freqs = np.fft.fftfreq(len(trace))
    axes[0, 2].semilogy(freqs[:len(freqs)//2], np.abs(fft_trace[:len(fft_trace)//2]))
    axes[0, 2].set_title('Frequency Spectrum')
    axes[0, 2].set_xlabel('Normalized Frequency')
    axes[0, 2].set_ylabel('Amplitude')
    axes[0, 2].grid(True, alpha=0.3)

    print("🔄 Plotting velocity model...")
    # Plot predicted velocity model
    velocity_sample = velocity_pred[0, 0, :, :].detach().numpy()
    im2 = axes[1, 0].imshow(velocity_sample, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Predicted Velocity Model')
    axes[1, 0].set_xlabel('Width')
    axes[1, 0].set_ylabel('Depth')
    plt.colorbar(im2, ax=axes[1, 0], label='Velocity (m/s)')

    print("🔄 Creating depth profile...")
    # Plot velocity depth profile
    depth_profile = np.mean(velocity_sample, axis=1)
    depths = np.arange(len(depth_profile))
    axes[1, 1].plot(depth_profile, depths)
    axes[1, 1].set_title('Velocity Depth Profile')
    axes[1, 1].set_xlabel('Velocity (m/s)')
    axes[1, 1].set_ylabel('Depth')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3)

    print("🔄 Creating histogram...")
    # Plot velocity histogram
    axes[1, 2].hist(velocity_sample.flatten(), bins=30, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Velocity Distribution')
    axes[1, 2].set_xlabel('Velocity (m/s)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)

    print("💾 Saving visualization...")
    plt.tight_layout()
    plt.savefig('hybrid_architecture_demo.png', dpi=300, bbox_inches='tight')

    print("🖼️  Displaying plot...")
    plt.show()

    print("✅ Saved visualization as 'hybrid_architecture_demo.png'")

def main():
    """
    Main demo function with comprehensive progress tracking.
    """
    print("🚀 HYBRID CNN-RNN ARCHITECTURE DEMO")
    print("🌊 Seismic Waveform Inversion Neural Network")
    print("="*60)

    total_start_time = time.time()

    try:
        # Test 1: Model architecture
        print("🏗️  PHASE 1: Testing Model Architecture")
        phase1_start = time.time()
        model, synthetic_seismic, velocity_pred = test_model_architecture()
        phase1_time = time.time() - phase1_start
        print(f"✅ Phase 1 completed in {phase1_time:.2f}s")

        # Test 2: Data loading (if data available)
        print("\n📁 PHASE 2: Testing Data Loading")
        phase2_start = time.time()
        real_data = test_data_loading()
        phase2_time = time.time() - phase2_start
        print(f"✅ Phase 2 completed in {phase2_time:.2f}s")

        # Test 3: Loss functions and evaluation
        print("\n⚖️  PHASE 3: Testing Loss Functions")
        phase3_start = time.time()

        # Use real data if available, otherwise synthetic
        if real_data is not None:
            seismic_data, velocity_target = real_data
            print(f"🎯 Using real data for testing...")

            # Run model on real data
            print("🧠 Running model on real data...")
            model.eval()
            device = next(model.parameters()).device
            seismic_data = seismic_data.to(device)

            with torch.no_grad():
                velocity_pred_real = model(seismic_data)

            # Test loss functions with real data
            test_loss_functions(model, seismic_data, velocity_target)

            # Use real data for visualization
            viz_seismic = seismic_data
            viz_velocity = velocity_pred_real

        else:
            print(f"🎯 Using synthetic data for testing...")

            # Test loss functions with synthetic data
            test_loss_functions(model, synthetic_seismic)

            # Use synthetic data for visualization
            viz_seismic = synthetic_seismic
            viz_velocity = velocity_pred

        phase3_time = time.time() - phase3_start
        print(f"✅ Phase 3 completed in {phase3_time:.2f}s")

        # Test 4: Visualization
        print("\n🎨 PHASE 4: Creating Visualizations")
        phase4_start = time.time()
        visualize_architecture_test(viz_seismic, viz_velocity)
        phase4_time = time.time() - phase4_start
        print(f"✅ Phase 4 completed in {phase4_time:.2f}s")

        # Summary
        total_time = time.time() - total_start_time
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)

        print(f"\n⏱️  Performance Summary:")
        print(f"  Phase 1 (Architecture): {phase1_time:.2f}s")
        print(f"  Phase 2 (Data Loading): {phase2_time:.2f}s")
        print(f"  Phase 3 (Loss Functions): {phase3_time:.2f}s")
        print(f"  Phase 4 (Visualization): {phase4_time:.2f}s")
        print(f"  Total Demo Time: {total_time:.2f}s")

        print(f"\n🎯 Next Steps:")
        print("1. 🏃 Run 'python train_seismic_inversion.py' to start training")
        print("2. 📊 Monitor training progress with TensorBoard")
        print("3. 📈 Evaluate trained model with 'python evaluate_model.py'")
        print("4. 🔬 Experiment with different architectures and hyperparameters")
        print("5. 🌐 Try different data families and complexity levels")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure PyTorch is properly installed")
        print("2. Check if CUDA is available and working")
        print("3. Verify data directory structure")
        print("4. Try reducing batch size if memory issues occur")

if __name__ == "__main__":
    main()
