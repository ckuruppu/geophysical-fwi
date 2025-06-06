#!/usr/bin/env python3
"""
Example script showing how to load and work with the geophysical waveform data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_vel_style_data(family_name, data_root="waveform-inversion"):
    """
    Load data from Vel or Style families (data/model structure).
    
    Args:
        family_name: One of ['FlatVel_A', 'FlatVel_B', 'CurveVel_A', 'CurveVel_B', 'Style_A', 'Style_B']
        data_root: Root directory containing the data
    
    Returns:
        tuple: (seismic_data, velocity_models) as numpy arrays
    """
    data_path = Path(data_root) / "train_samples" / family_name
    
    # Load all data files
    data_files = sorted((data_path / "data").glob("*.npy"))
    model_files = sorted((data_path / "model").glob("*.npy"))
    
    seismic_data = []
    velocity_models = []
    
    for data_file, model_file in zip(data_files, model_files):
        seismic_data.append(np.load(data_file))
        velocity_models.append(np.load(model_file))
    
    return np.concatenate(seismic_data), np.concatenate(velocity_models)

def load_fault_data(family_name, data_root="waveform-inversion"):
    """
    Load data from Fault families (seis_/vel_ structure).
    
    Args:
        family_name: One of ['FlatFault_A', 'FlatFault_B', 'CurveFault_A', 'CurveFault_B']
        data_root: Root directory containing the data
    
    Returns:
        tuple: (seismic_data, velocity_models) as numpy arrays
    """
    data_path = Path(data_root) / "train_samples" / family_name
    
    # Load all seis and vel files
    seis_files = sorted(data_path.glob("seis_*.npy"))
    vel_files = sorted(data_path.glob("vel_*.npy"))
    
    seismic_data = []
    velocity_models = []
    
    for seis_file, vel_file in zip(seis_files, vel_files):
        seismic_data.append(np.load(seis_file))
        velocity_models.append(np.load(vel_file))
    
    return np.concatenate(seismic_data), np.concatenate(velocity_models)

def load_test_data(data_root="waveform-inversion", num_samples=10):
    """
    Load test data samples.
    
    Args:
        data_root: Root directory containing the data
        num_samples: Number of test samples to load
    
    Returns:
        list: List of test data arrays
    """
    test_path = Path(data_root) / "test"
    test_files = sorted(test_path.glob("*.npy"))[:num_samples]
    
    test_data = []
    for test_file in test_files:
        test_data.append(np.load(test_file))
    
    return test_data

def visualize_sample(seismic_data, velocity_model, sample_idx=0):
    """
    Visualize a sample from the dataset.
    
    Args:
        seismic_data: Seismic data array
        velocity_model: Velocity model array
        sample_idx: Index of sample to visualize
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract sample
    if len(seismic_data.shape) == 4:  # Training data
        seismic_sample = seismic_data[sample_idx]  # (sources, time, receivers)
        velocity_sample = velocity_model[sample_idx, 0]  # (height, width)
    else:  # Test data
        seismic_sample = seismic_data  # Already single sample
        velocity_sample = velocity_model[sample_idx, 0] if len(velocity_model.shape) == 4 else velocity_model
    
    # Plot seismic shot gather (first source)
    axes[0, 0].imshow(seismic_sample[0].T, aspect='auto', cmap='RdBu')
    axes[0, 0].set_title('Seismic Shot Gather (Source 1)')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Receiver Number')
    
    # Plot velocity model
    im1 = axes[0, 1].imshow(velocity_sample, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('Velocity Model')
    axes[0, 1].set_xlabel('Width')
    axes[0, 1].set_ylabel('Depth')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot single trace
    trace = seismic_sample[0, :, 35]  # Middle receiver
    axes[0, 2].plot(trace)
    axes[0, 2].set_title('Sample Trace (Middle Receiver)')
    axes[0, 2].set_xlabel('Time Steps')
    axes[0, 2].set_ylabel('Amplitude')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot velocity depth profile
    depth_profile = np.mean(velocity_sample, axis=1)
    depths = np.arange(len(depth_profile))
    axes[1, 0].plot(depth_profile, depths)
    axes[1, 0].set_title('Velocity vs Depth')
    axes[1, 0].set_xlabel('Velocity (m/s)')
    axes[1, 0].set_ylabel('Depth')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot receiver gather (all sources)
    receiver_gather = seismic_sample[:, :, 35].T  # Middle receiver, all sources
    axes[1, 1].imshow(receiver_gather, aspect='auto', cmap='RdBu')
    axes[1, 1].set_title('Receiver Gather (Middle Receiver)')
    axes[1, 1].set_xlabel('Source Number')
    axes[1, 1].set_ylabel('Time Steps')
    
    # Plot amplitude vs offset
    time_slice = 200
    amplitudes = np.abs(seismic_sample[0, time_slice, :])
    axes[1, 2].plot(amplitudes)
    axes[1, 2].set_title(f'Amplitude vs Offset (t={time_slice})')
    axes[1, 2].set_xlabel('Receiver Number')
    axes[1, 2].set_ylabel('Amplitude')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Example usage of the data loading functions."""
    print("Loading geophysical waveform data...")
    
    # Load Vel family data
    print("Loading Vel family data...")
    vel_seismic, vel_velocity = load_vel_style_data("FlatVel_A")
    print(f"Vel data shapes: Seismic {vel_seismic.shape}, Velocity {vel_velocity.shape}")
    
    # Load Fault family data
    print("Loading Fault family data...")
    fault_seismic, fault_velocity = load_fault_data("FlatFault_A")
    print(f"Fault data shapes: Seismic {fault_seismic.shape}, Velocity {fault_velocity.shape}")
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(num_samples=5)
    print(f"Loaded {len(test_data)} test samples, first shape: {test_data[0].shape}")
    
    # Visualize samples
    print("Creating visualizations...")
    
    print("Visualizing Vel family sample...")
    visualize_sample(vel_seismic, vel_velocity, sample_idx=0)
    
    print("Visualizing Fault family sample...")
    visualize_sample(fault_seismic, fault_velocity, sample_idx=0)
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"Vel seismic range: [{vel_seismic.min():.3f}, {vel_seismic.max():.3f}]")
    print(f"Vel velocity range: [{vel_velocity.min():.0f}, {vel_velocity.max():.0f}] m/s")
    print(f"Fault seismic range: [{fault_seismic.min():.3f}, {fault_seismic.max():.3f}]")
    print(f"Fault velocity range: [{fault_velocity.min():.0f}, {fault_velocity.max():.0f}] m/s")

if __name__ == "__main__":
    main()
