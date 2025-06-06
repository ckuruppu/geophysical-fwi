#!/usr/bin/env python3
"""
Enhanced analysis script for geophysical waveform data with detailed insights.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

def analyze_waveform_characteristics():
    """Detailed analysis of waveform characteristics."""
    print("="*80)
    print("ENHANCED GEOPHYSICAL WAVEFORM DATA ANALYSIS")
    print("="*80)
    
    # Load sample data
    data_root = Path("waveform-inversion")
    
    # Load Vel family data
    vel_seismic = np.load(data_root / "train_samples/FlatVel_A/data/data1.npy")
    vel_velocity = np.load(data_root / "train_samples/FlatVel_A/model/model1.npy")
    
    # Load Fault family data  
    fault_seismic = np.load(data_root / "train_samples/FlatFault_A/seis2_1_0.npy")
    fault_velocity = np.load(data_root / "train_samples/FlatFault_A/vel2_1_0.npy")
    
    # Load test data
    test_files = list((data_root / "test").glob("*.npy"))
    test_data = np.load(test_files[0])
    
    print(f"\nDATA STRUCTURE ANALYSIS:")
    print(f"Vel Seismic: {vel_seismic.shape} - (batch, sources, time_steps, receivers)")
    print(f"Vel Velocity: {vel_velocity.shape} - (batch, depth_layers, height, width)")
    print(f"Test Data: {test_data.shape} - (sources, time_steps, receivers)")
    
    # Time scale analysis
    print(f"\nTIME SCALE CHARACTERISTICS:")
    time_steps = vel_seismic.shape[2]
    print(f"Time steps: {time_steps}")
    print(f"If sampling rate is 4ms (typical for seismic): Total time = {time_steps * 4}ms = {time_steps * 4 / 1000:.1f}s")
    print(f"If sampling rate is 2ms: Total time = {time_steps * 2}ms = {time_steps * 2 / 1000:.1f}s")
    
    # Frequency analysis
    print(f"\nFREQUENCY ANALYSIS:")
    sample_trace = vel_seismic[0, 0, :, 0]
    
    # Compute power spectral density
    freqs, psd = signal.welch(sample_trace, nperseg=256)
    dominant_freq_idx = np.argmax(psd)
    dominant_freq = freqs[dominant_freq_idx]
    
    print(f"Dominant frequency (normalized): {dominant_freq:.4f}")
    print(f"If sampling rate is 250Hz: Dominant frequency = {dominant_freq * 250:.1f}Hz")
    print(f"If sampling rate is 500Hz: Dominant frequency = {dominant_freq * 500:.1f}Hz")
    
    # Velocity analysis
    print(f"\nVELOCITY CHARACTERISTICS:")
    vel_sample = vel_velocity[0, 0]  # First sample, first layer
    fault_vel_sample = fault_velocity[0, 0]
    
    print(f"Vel family velocity range: {vel_sample.min():.0f} - {vel_sample.max():.0f} m/s")
    print(f"Fault family velocity range: {fault_vel_sample.min():.0f} - {fault_vel_sample.max():.0f} m/s")
    
    # Typical seismic velocities for reference
    print(f"\nTYPICAL SEISMIC VELOCITIES (for reference):")
    print(f"Water: ~1500 m/s")
    print(f"Sediments: 1500-3000 m/s") 
    print(f"Sandstone: 2000-4000 m/s")
    print(f"Limestone: 3000-6000 m/s")
    print(f"Crystalline rocks: 4000-7000 m/s")
    
    # Spatial analysis
    print(f"\nSPATIAL CHARACTERISTICS:")
    height, width = vel_sample.shape
    print(f"Model dimensions: {height} x {width} grid points")
    print(f"If grid spacing is 10m: Model size = {height*10/1000:.1f}km x {width*10/1000:.1f}km")
    print(f"If grid spacing is 25m: Model size = {height*25/1000:.1f}km x {width*25/1000:.1f}km")
    
    # Receiver geometry analysis
    print(f"\nRECEIVER GEOMETRY:")
    num_receivers = vel_seismic.shape[3]
    num_sources = vel_seismic.shape[1]
    print(f"Number of receivers: {num_receivers}")
    print(f"Number of sources: {num_sources}")
    print(f"If receiver spacing is 50m: Array length = {(num_receivers-1)*50/1000:.1f}km")
    print(f"If receiver spacing is 25m: Array length = {(num_receivers-1)*25/1000:.1f}km")
    
    return {
        'vel_seismic': vel_seismic,
        'vel_velocity': vel_velocity,
        'fault_seismic': fault_seismic,
        'fault_velocity': fault_velocity,
        'test_data': test_data
    }

def analyze_signal_quality(data_dict):
    """Analyze signal quality and noise characteristics."""
    print(f"\nSIGNAL QUALITY ANALYSIS:")
    print("="*50)
    
    vel_seismic = data_dict['vel_seismic']
    
    # Signal-to-noise ratio estimation
    sample_trace = vel_seismic[0, 0, :, 35]  # Middle receiver
    
    # Estimate noise from late time (assuming signal dies out)
    signal_window = sample_trace[:200]  # Early time (signal)
    noise_window = sample_trace[-200:]  # Late time (noise)
    
    signal_power = np.var(signal_window)
    noise_power = np.var(noise_window)
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    print(f"Estimated SNR: {snr_db:.1f} dB")
    
    # First break analysis (approximate)
    abs_trace = np.abs(sample_trace)
    threshold = 0.1 * abs_trace.max()
    first_break_idx = np.where(abs_trace > threshold)[0]
    if len(first_break_idx) > 0:
        first_break = first_break_idx[0]
        print(f"Approximate first break: time step {first_break}")
        print(f"If 4ms sampling: First break at {first_break * 4}ms")

def analyze_velocity_models(data_dict):
    """Analyze velocity model characteristics."""
    print(f"\nVELOCITY MODEL ANALYSIS:")
    print("="*50)
    
    vel_velocity = data_dict['vel_velocity']
    fault_velocity = data_dict['fault_velocity']
    
    # Analyze velocity gradients
    for name, vel_data in [("Vel Family", vel_velocity), ("Fault Family", fault_velocity)]:
        sample = vel_data[0, 0]
        
        # Vertical gradient (with depth)
        depth_profile = np.mean(sample, axis=1)
        depth_gradient = np.gradient(depth_profile)
        
        print(f"\n{name}:")
        print(f"  Surface velocity: {depth_profile[0]:.0f} m/s")
        print(f"  Bottom velocity: {depth_profile[-1]:.0f} m/s")
        print(f"  Average velocity gradient: {np.mean(depth_gradient):.2f} m/s per grid point")
        print(f"  Max velocity contrast: {sample.max() - sample.min():.0f} m/s")
        
        # Lateral heterogeneity
        lateral_std = np.std(sample, axis=1)  # Std across width for each depth
        print(f"  Average lateral heterogeneity: {np.mean(lateral_std):.1f} m/s")
        print(f"  Max lateral heterogeneity: {np.max(lateral_std):.1f} m/s")

def create_detailed_plots(data_dict):
    """Create detailed visualization plots."""
    print(f"\nCREATING DETAILED VISUALIZATIONS...")
    
    vel_seismic = data_dict['vel_seismic']
    vel_velocity = data_dict['vel_velocity']
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Seismic gather (shot record)
    ax1 = plt.subplot(3, 4, 1)
    gather = vel_seismic[0, 0, :, :]  # First shot
    im1 = ax1.imshow(gather.T, aspect='auto', cmap='RdBu', 
                     extent=[0, gather.shape[0], gather.shape[1], 0])
    ax1.set_title('Seismic Shot Gather')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Receiver Number')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Velocity model
    ax2 = plt.subplot(3, 4, 2)
    vel_model = vel_velocity[0, 0]
    im2 = ax2.imshow(vel_model, aspect='auto', cmap='viridis')
    ax2.set_title('Velocity Model')
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Depth')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Single trace
    ax3 = plt.subplot(3, 4, 3)
    trace = vel_seismic[0, 0, :, 35]
    ax3.plot(trace)
    ax3.set_title('Sample Seismic Trace')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)
    
    # 4. Frequency spectrum
    ax4 = plt.subplot(3, 4, 4)
    freqs, psd = signal.welch(trace, nperseg=256)
    ax4.semilogy(freqs, psd)
    ax4.set_title('Power Spectral Density')
    ax4.set_xlabel('Normalized Frequency')
    ax4.set_ylabel('Power')
    ax4.grid(True, alpha=0.3)
    
    # 5. Velocity depth profile
    ax5 = plt.subplot(3, 4, 5)
    depth_profile = np.mean(vel_model, axis=1)
    depths = np.arange(len(depth_profile))
    ax5.plot(depth_profile, depths)
    ax5.set_title('Velocity vs Depth')
    ax5.set_xlabel('Velocity (m/s)')
    ax5.set_ylabel('Depth')
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3)
    
    # 6. Receiver response
    ax6 = plt.subplot(3, 4, 6)
    receiver_response = np.mean(vel_seismic[0, 0], axis=0)
    ax6.plot(receiver_response)
    ax6.set_title('Average Receiver Response')
    ax6.set_xlabel('Receiver Number')
    ax6.set_ylabel('Average Amplitude')
    ax6.grid(True, alpha=0.3)
    
    # 7. Amplitude vs offset
    ax7 = plt.subplot(3, 4, 7)
    # Take amplitude at a specific time
    time_slice = 100
    amplitudes = np.abs(vel_seismic[0, 0, time_slice, :])
    ax7.plot(amplitudes)
    ax7.set_title(f'Amplitude vs Offset (t={time_slice})')
    ax7.set_xlabel('Receiver Number (Offset)')
    ax7.set_ylabel('Amplitude')
    ax7.grid(True, alpha=0.3)
    
    # 8. Velocity histogram
    ax8 = plt.subplot(3, 4, 8)
    ax8.hist(vel_model.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax8.set_title('Velocity Distribution')
    ax8.set_xlabel('Velocity (m/s)')
    ax8.set_ylabel('Frequency')
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved comprehensive analysis plot as 'comprehensive_analysis.png'")

def main():
    """Main analysis function."""
    data_dict = analyze_waveform_characteristics()
    analyze_signal_quality(data_dict)
    analyze_velocity_models(data_dict)
    create_detailed_plots(data_dict)
    
    print(f"\n" + "="*80)
    print("ENHANCED ANALYSIS COMPLETE!")
    print("Generated files:")
    print("- comprehensive_analysis.png")
    print("="*80)

if __name__ == "__main__":
    main()
