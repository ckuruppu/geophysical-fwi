#!/usr/bin/env python3
"""
Comprehensive analysis script for geophysical waveform data.
Analyzes seismic data and velocity maps from the OpenFWI dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

class WaveformDataAnalyzer:
    def __init__(self, data_root="waveform-inversion"):
        self.data_root = Path(data_root)
        self.train_samples = self.data_root / "train_samples"
        self.test_dir = self.data_root / "test"
        
        # Dataset families and their structures
        self.vel_style_families = ["FlatVel_A", "FlatVel_B", "CurveVel_A", "CurveVel_B", "Style_A", "Style_B"]
        self.fault_families = ["FlatFault_A", "FlatFault_B", "CurveFault_A", "CurveFault_B"]
        
        self.analysis_results = {}
        
    def load_sample_data(self):
        """Load sample data from each family type for analysis."""
        print("Loading sample data from each family...")
        
        # Load Vel/Style family data (data/model structure)
        vel_data_path = self.train_samples / "FlatVel_A" / "data" / "data1.npy"
        vel_model_path = self.train_samples / "FlatVel_A" / "model" / "model1.npy"
        
        if vel_data_path.exists() and vel_model_path.exists():
            self.vel_seismic_data = np.load(vel_data_path)
            self.vel_velocity_map = np.load(vel_model_path)
            print(f"Loaded Vel family - Seismic: {self.vel_seismic_data.shape}, Velocity: {self.vel_velocity_map.shape}")
        
        # Load Fault family data (seis_/vel_ structure)
        fault_seis_path = self.train_samples / "FlatFault_A" / "seis2_1_0.npy"
        fault_vel_path = self.train_samples / "FlatFault_A" / "vel2_1_0.npy"
        
        if fault_seis_path.exists() and fault_vel_path.exists():
            self.fault_seismic_data = np.load(fault_seis_path)
            self.fault_velocity_map = np.load(fault_vel_path)
            print(f"Loaded Fault family - Seismic: {self.fault_seismic_data.shape}, Velocity: {self.fault_velocity_map.shape}")
        
        # Load test data sample
        test_files = list(self.test_dir.glob("*.npy"))
        if test_files:
            self.test_data = np.load(test_files[0])
            print(f"Loaded test sample: {self.test_data.shape}")
    
    def analyze_data_shapes_and_ranges(self):
        """Analyze data shapes, value ranges, and basic statistics."""
        print("\n" + "="*60)
        print("DATA SHAPE AND RANGE ANALYSIS")
        print("="*60)
        
        datasets = [
            ("Vel Family Seismic", self.vel_seismic_data),
            ("Vel Family Velocity", self.vel_velocity_map),
            ("Fault Family Seismic", self.fault_seismic_data),
            ("Fault Family Velocity", self.fault_velocity_map),
            ("Test Data", self.test_data)
        ]
        
        for name, data in datasets:
            print(f"\n{name}:")
            print(f"  Shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  Min value: {data.min():.6f}")
            print(f"  Max value: {data.max():.6f}")
            print(f"  Mean: {data.mean():.6f}")
            print(f"  Std: {data.std():.6f}")
            print(f"  Memory usage: {data.nbytes / 1024**2:.2f} MB")
            
            # Check for NaN or infinite values
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  WARNING: {nan_count} NaN values, {inf_count} infinite values")
    
    def analyze_time_scales(self):
        """Analyze temporal characteristics of seismic data."""
        print("\n" + "="*60)
        print("TIME SCALE ANALYSIS")
        print("="*60)
        
        # Analyze seismic data temporal dimension
        for name, data in [("Vel Family", self.vel_seismic_data), ("Fault Family", self.fault_seismic_data)]:
            if len(data.shape) == 4:  # (batch, sources, time, receivers)
                time_steps = data.shape[2]
                print(f"\n{name} Seismic Data:")
                print(f"  Time steps: {time_steps}")
                print(f"  Number of sources: {data.shape[1]}")
                print(f"  Number of receivers: {data.shape[3]}")
                
                # Analyze a single trace
                sample_trace = data[0, 0, :, 0]  # First batch, first source, all time, first receiver
                print(f"  Sample trace range: [{sample_trace.min():.6f}, {sample_trace.max():.6f}]")
                
                # Estimate dominant period (assuming some sampling rate)
                # This is a rough estimate - actual sampling rate would be needed for precise analysis
                autocorr = np.correlate(sample_trace, sample_trace, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                # Find peaks in autocorrelation to estimate periodicity
                peaks, _ = signal.find_peaks(autocorr[1:], height=autocorr.max() * 0.1)
                if len(peaks) > 0:
                    dominant_period = peaks[0] + 1
                    print(f"  Estimated dominant period: {dominant_period} time steps")
    
    def analyze_frequency_content(self):
        """Perform frequency analysis of seismic data."""
        print("\n" + "="*60)
        print("FREQUENCY ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Frequency Analysis of Seismic Data', fontsize=16)
        
        datasets = [
            ("Vel Family", self.vel_seismic_data),
            ("Fault Family", self.fault_seismic_data)
        ]
        
        for idx, (name, data) in enumerate(datasets):
            if len(data.shape) == 4:
                # Take a sample trace
                sample_trace = data[0, 0, :, 0]
                
                # Compute FFT
                fft_vals = fft(sample_trace)
                freqs = fftfreq(len(sample_trace))
                
                # Plot time domain
                axes[idx, 0].plot(sample_trace)
                axes[idx, 0].set_title(f'{name} - Time Domain')
                axes[idx, 0].set_xlabel('Time Steps')
                axes[idx, 0].set_ylabel('Amplitude')
                
                # Plot frequency domain (magnitude)
                axes[idx, 1].plot(freqs[:len(freqs)//2], np.abs(fft_vals[:len(fft_vals)//2]))
                axes[idx, 1].set_title(f'{name} - Frequency Domain')
                axes[idx, 1].set_xlabel('Normalized Frequency')
                axes[idx, 1].set_ylabel('Magnitude')
                
                # Print frequency statistics
                magnitude = np.abs(fft_vals[:len(fft_vals)//2])
                peak_freq_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
                peak_freq = freqs[peak_freq_idx]
                
                print(f"\n{name}:")
                print(f"  Peak frequency (normalized): {peak_freq:.4f}")
                print(f"  Frequency resolution: {freqs[1] - freqs[0]:.6f}")
                print(f"  Nyquist frequency: {freqs[len(freqs)//2]:.4f}")
        
        plt.tight_layout()
        plt.savefig('frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_spatial_characteristics(self):
        """Analyze spatial characteristics of velocity maps."""
        print("\n" + "="*60)
        print("SPATIAL CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Velocity Map Analysis', fontsize=16)
        
        datasets = [
            ("Vel Family", self.vel_velocity_map),
            ("Fault Family", self.fault_velocity_map)
        ]
        
        for idx, (name, data) in enumerate(datasets):
            if len(data.shape) == 3:  # (batch, height, width)
                sample_map = data[0]  # First sample
                
                # Original velocity map
                im1 = axes[idx, 0].imshow(sample_map, cmap='viridis', aspect='auto')
                axes[idx, 0].set_title(f'{name} - Velocity Map')
                axes[idx, 0].set_xlabel('Width')
                axes[idx, 0].set_ylabel('Height (Depth)')
                plt.colorbar(im1, ax=axes[idx, 0])
                
                # Horizontal gradient (first derivative)
                grad_x = np.gradient(sample_map, axis=1)
                im2 = axes[idx, 1].imshow(grad_x, cmap='RdBu', aspect='auto')
                axes[idx, 1].set_title(f'{name} - Horizontal Gradient')
                axes[idx, 1].set_xlabel('Width')
                axes[idx, 1].set_ylabel('Height (Depth)')
                plt.colorbar(im2, ax=axes[idx, 1])
                
                # Vertical gradient
                grad_y = np.gradient(sample_map, axis=0)
                im3 = axes[idx, 2].imshow(grad_y, cmap='RdBu', aspect='auto')
                axes[idx, 2].set_title(f'{name} - Vertical Gradient')
                axes[idx, 2].set_xlabel('Width')
                axes[idx, 2].set_ylabel('Height (Depth)')
                plt.colorbar(im3, ax=axes[idx, 2])
                
                # Print spatial statistics
                print(f"\n{name} Velocity Map:")
                print(f"  Dimensions: {sample_map.shape[0]} x {sample_map.shape[1]} (H x W)")
                print(f"  Velocity range: [{sample_map.min():.3f}, {sample_map.max():.3f}]")
                print(f"  Horizontal gradient range: [{grad_x.min():.6f}, {grad_x.max():.6f}]")
                print(f"  Vertical gradient range: [{grad_y.min():.6f}, {grad_y.max():.6f}]")
                print(f"  Horizontal gradient std: {grad_x.std():.6f}")
                print(f"  Vertical gradient std: {grad_y.std():.6f}")
        
        plt.tight_layout()
        plt.savefig('spatial_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_second_derivatives(self):
        """Analyze second derivatives (curvature) of velocity maps."""
        print("\n" + "="*60)
        print("SECOND DERIVATIVE (CURVATURE) ANALYSIS")
        print("="*60)
        
        datasets = [
            ("Vel Family", self.vel_velocity_map),
            ("Fault Family", self.fault_velocity_map)
        ]
        
        for name, data in datasets:
            if len(data.shape) == 3:
                sample_map = data[0]
                
                # Compute second derivatives
                grad_xx = np.gradient(np.gradient(sample_map, axis=1), axis=1)
                grad_yy = np.gradient(np.gradient(sample_map, axis=0), axis=0)
                grad_xy = np.gradient(np.gradient(sample_map, axis=1), axis=0)
                
                # Laplacian (sum of second derivatives)
                laplacian = grad_xx + grad_yy
                
                print(f"\n{name} Second Derivatives:")
                print(f"  d²/dx² range: [{grad_xx.min():.8f}, {grad_xx.max():.8f}]")
                print(f"  d²/dy² range: [{grad_yy.min():.8f}, {grad_yy.max():.8f}]")
                print(f"  d²/dxdy range: [{grad_xy.min():.8f}, {grad_xy.max():.8f}]")
                print(f"  Laplacian range: [{laplacian.min():.8f}, {laplacian.max():.8f}]")
                print(f"  Laplacian std: {laplacian.std():.8f}")

    def analyze_receiver_source_patterns(self):
        """Analyze patterns across receivers and sources."""
        print("\n" + "="*60)
        print("RECEIVER AND SOURCE PATTERN ANALYSIS")
        print("="*60)

        datasets = [
            ("Vel Family", self.vel_seismic_data),
            ("Fault Family", self.fault_seismic_data)
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Receiver and Source Analysis', fontsize=16)

        for idx, (name, data) in enumerate(datasets):
            if len(data.shape) == 4:  # (batch, sources, time, receivers)
                sample = data[0]  # First batch

                # Average over time to see source-receiver patterns
                time_avg = np.mean(sample, axis=1)  # Shape: (sources, receivers)

                # Plot source-receiver heatmap
                im1 = axes[idx, 0].imshow(time_avg, cmap='viridis', aspect='auto')
                axes[idx, 0].set_title(f'{name} - Source-Receiver Pattern')
                axes[idx, 0].set_xlabel('Receivers')
                axes[idx, 0].set_ylabel('Sources')
                plt.colorbar(im1, ax=axes[idx, 0])

                # Plot receiver response (averaged over sources and time)
                receiver_response = np.mean(sample, axis=(0, 1))
                axes[idx, 1].plot(receiver_response)
                axes[idx, 1].set_title(f'{name} - Average Receiver Response')
                axes[idx, 1].set_xlabel('Receiver Index')
                axes[idx, 1].set_ylabel('Average Amplitude')

                # Plot source response (averaged over receivers and time)
                source_response = np.mean(sample, axis=(1, 2))
                axes[idx, 2].plot(source_response)
                axes[idx, 2].set_title(f'{name} - Average Source Response')
                axes[idx, 2].set_xlabel('Source Index')
                axes[idx, 2].set_ylabel('Average Amplitude')

                print(f"\n{name}:")
                print(f"  Source-receiver correlation: {np.corrcoef(time_avg.flatten(), time_avg.T.flatten())[0,1]:.4f}")
                print(f"  Receiver response range: [{receiver_response.min():.6f}, {receiver_response.max():.6f}]")
                print(f"  Source response range: [{source_response.min():.6f}, {source_response.max():.6f}]")

        plt.tight_layout()
        plt.savefig('receiver_source_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_depth_profiles(self):
        """Analyze velocity profiles with depth."""
        print("\n" + "="*60)
        print("DEPTH PROFILE ANALYSIS")
        print("="*60)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Velocity Depth Profiles', fontsize=16)

        datasets = [
            ("Vel Family", self.vel_velocity_map),
            ("Fault Family", self.fault_velocity_map)
        ]

        for idx, (name, data) in enumerate(datasets):
            if len(data.shape) == 3:
                sample_map = data[0]

                # Average velocity profile with depth
                depth_profile = np.mean(sample_map, axis=1)  # Average across width
                depth_std = np.std(sample_map, axis=1)  # Standard deviation across width

                depths = np.arange(len(depth_profile))

                axes[idx].plot(depth_profile, depths, label='Mean velocity', linewidth=2)
                axes[idx].fill_betweenx(depths,
                                      depth_profile - depth_std,
                                      depth_profile + depth_std,
                                      alpha=0.3, label='±1 std')
                axes[idx].set_xlabel('Velocity')
                axes[idx].set_ylabel('Depth (increasing downward)')
                axes[idx].set_title(f'{name} - Depth Profile')
                axes[idx].legend()
                axes[idx].invert_yaxis()  # Depth increases downward
                axes[idx].grid(True, alpha=0.3)

                # Calculate depth gradient
                depth_gradient = np.gradient(depth_profile)

                print(f"\n{name} Depth Analysis:")
                print(f"  Surface velocity: {depth_profile[0]:.4f}")
                print(f"  Bottom velocity: {depth_profile[-1]:.4f}")
                print(f"  Velocity increase with depth: {depth_profile[-1] - depth_profile[0]:.4f}")
                print(f"  Average depth gradient: {np.mean(depth_gradient):.6f}")
                print(f"  Max depth gradient: {np.max(np.abs(depth_gradient)):.6f}")

        plt.tight_layout()
        plt.savefig('depth_profiles.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics."""
        print("\n" + "="*60)
        print("COMPREHENSIVE SUMMARY STATISTICS")
        print("="*60)

        # Create summary dataframe
        summary_data = []

        datasets = [
            ("Vel_Seismic", self.vel_seismic_data),
            ("Vel_Velocity", self.vel_velocity_map),
            ("Fault_Seismic", self.fault_seismic_data),
            ("Fault_Velocity", self.fault_velocity_map),
            ("Test_Data", self.test_data)
        ]

        for name, data in datasets:
            stats = {
                'Dataset': name,
                'Shape': str(data.shape),
                'Dimensions': len(data.shape),
                'Total_Elements': data.size,
                'Memory_MB': data.nbytes / 1024**2,
                'Min': data.min(),
                'Max': data.max(),
                'Mean': data.mean(),
                'Std': data.std(),
                'Median': np.median(data),
                'Q25': np.percentile(data, 25),
                'Q75': np.percentile(data, 75),
                'Skewness': self._calculate_skewness(data),
                'Kurtosis': self._calculate_kurtosis(data)
            }
            summary_data.append(stats)

        # Create and display summary table
        df = pd.DataFrame(summary_data)
        print("\nSummary Statistics Table:")
        print(df.to_string(index=False, float_format='%.6f'))

        # Save to CSV
        df.to_csv('data_summary_statistics.csv', index=False)
        print(f"\nSummary statistics saved to 'data_summary_statistics.csv'")

    def _calculate_skewness(self, data):
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive geophysical waveform data analysis...")
        print("="*80)

        # Load data
        self.load_sample_data()

        # Run all analyses
        self.analyze_data_shapes_and_ranges()
        self.analyze_time_scales()
        self.analyze_frequency_content()
        self.analyze_spatial_characteristics()
        self.analyze_second_derivatives()
        self.analyze_receiver_source_patterns()
        self.analyze_depth_profiles()
        self.generate_summary_statistics()

        print("\n" + "="*80)
        print("Analysis complete! Generated files:")
        print("- frequency_analysis.png")
        print("- spatial_analysis.png")
        print("- receiver_source_analysis.png")
        print("- depth_profiles.png")
        print("- data_summary_statistics.csv")
        print("="*80)

def main():
    """Main function to run the analysis."""
    analyzer = WaveformDataAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
