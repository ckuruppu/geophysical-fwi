#!/usr/bin/env python3
"""
Comprehensive analysis of the full waveform inversion dataset.
Optimized for 14GB+ data with batch processing, memory optimization, and parallelism.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import gc
import psutil
import time
from functools import partial
import json

class MemoryMonitor:
    """Monitor memory usage during processing."""

    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, memory_mb)
        return memory_mb

    def print_memory_status(self, context=""):
        """Print current memory status."""
        current_mb = self.get_memory_usage()
        available_mb = psutil.virtual_memory().available / 1024 / 1024
        print(f"  💾 Memory {context}: {current_mb:.1f}MB used, {available_mb:.1f}MB available")

def analyze_file_batch(file_paths: List[Path], batch_id: int) -> Dict:
    """Analyze a batch of files in parallel."""
    print(f"    🔄 Processing batch {batch_id} ({len(file_paths)} files)...")

    batch_info = {
        'batch_id': batch_id,
        'files_processed': 0,
        'total_samples': 0,
        'shapes': [],
        'file_sizes_mb': [],
        'processing_time': 0
    }

    start_time = time.time()

    for file_path in file_paths:
        try:
            # Get file size without loading
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            batch_info['file_sizes_mb'].append(file_size_mb)

            # Load only metadata (shape) efficiently
            with open(file_path, 'rb') as f:
                # Read numpy header to get shape without loading data
                version = np.lib.format.read_magic(f)
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)

                batch_info['shapes'].append(shape)

                # Estimate samples based on shape
                if len(shape) == 4:  # Batched data (batch, sources, time, receivers)
                    batch_info['total_samples'] += shape[0]
                elif len(shape) == 3:  # Single sample (sources, time, receivers)
                    batch_info['total_samples'] += 1
                else:
                    batch_info['total_samples'] += 1

                batch_info['files_processed'] += 1

        except Exception as e:
            print(f"      ⚠️  Error processing {file_path}: {e}")
            continue

    batch_info['processing_time'] = time.time() - start_time
    print(f"    ✅ Batch {batch_id} completed in {batch_info['processing_time']:.2f}s")

    return batch_info

def analyze_scenario_parallel(scenario_path: Path, max_workers: int = 4) -> Dict:
    """Analyze a single scenario using parallel processing."""
    print(f"\n🏔️  Analyzing scenario: {scenario_path.name}")

    scenario_info = {
        'scenario': scenario_path.name,
        'structure': 'unknown',
        'total_files': 0,
        'total_samples': 0,
        'total_size_gb': 0,
        'data_shapes': [],
        'model_shapes': [],
        'processing_time': 0
    }

    start_time = time.time()
    memory_monitor = MemoryMonitor()

    # Check structure
    data_dir = scenario_path / "data"
    model_dir = scenario_path / "model"

    if data_dir.exists() and model_dir.exists():
        scenario_info['structure'] = 'data/model_dirs'
        data_files = list(data_dir.glob("*.npy"))
        model_files = list(model_dir.glob("*.npy"))
        all_files = data_files + model_files

        print(f"  📁 Structure: data/ and model/ subdirectories")
        print(f"  📊 Data files: {len(data_files)}, Model files: {len(model_files)}")

    else:
        scenario_info['structure'] = 'direct_files'
        seis_files = list(scenario_path.glob("seis*.npy"))
        vel_files = list(scenario_path.glob("vel*.npy"))
        all_files = seis_files + vel_files

        print(f"  📁 Structure: direct .npy files")
        print(f"  📊 Seismic files: {len(seis_files)}, Velocity files: {len(vel_files)}")

    scenario_info['total_files'] = len(all_files)

    if not all_files:
        print(f"  ⚠️  No data files found in {scenario_path}")
        return scenario_info

    # Process files in parallel batches
    batch_size = max(1, len(all_files) // max_workers)
    file_batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]

    print(f"  🔄 Processing {len(file_batches)} batches with {max_workers} workers...")
    memory_monitor.print_memory_status("before processing")

    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(analyze_file_batch, batch, i): i
            for i, batch in enumerate(file_batches)
        }

        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                batch_result = future.result()
                scenario_info['total_samples'] += batch_result['total_samples']
                scenario_info['data_shapes'].extend(batch_result['shapes'])
                scenario_info['total_size_gb'] += sum(batch_result['file_sizes_mb']) / 1024

                # Force garbage collection after each batch
                gc.collect()

            except Exception as e:
                print(f"    ❌ Batch {batch_id} failed: {e}")

    scenario_info['processing_time'] = time.time() - start_time
    memory_monitor.print_memory_status("after processing")

    print(f"  ✅ Scenario completed: {scenario_info['total_samples']} samples, "
          f"{scenario_info['total_size_gb']:.2f}GB in {scenario_info['processing_time']:.2f}s")

    return scenario_info

def analyze_dataset_structure():
    """Analyze the complete dataset structure with memory optimization."""
    print("🔍 COMPREHENSIVE DATASET ANALYSIS (OPTIMIZED)")
    print("="*60)

    data_root = Path("waveform-inversion/train_samples")

    # Find all geological scenarios
    scenarios = [d for d in data_root.iterdir() if d.is_dir()]
    scenarios.sort()

    print(f"📊 Found {len(scenarios)} geological scenarios")

    # System info
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
    print(f"💻 System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")

    # Determine optimal worker count
    max_workers = min(cpu_count, 8)  # Cap at 8 to avoid overwhelming system
    print(f"🔧 Using {max_workers} parallel workers")

    total_samples = 0
    total_size_gb = 0
    scenario_info = []

    overall_start_time = time.time()
    memory_monitor = MemoryMonitor()

    # Process scenarios in parallel (but limit to avoid memory issues)
    scenario_batch_size = min(4, len(scenarios))  # Process 4 scenarios at a time
    scenario_batches = [scenarios[i:i + scenario_batch_size]
                       for i in range(0, len(scenarios), scenario_batch_size)]

    for batch_idx, scenario_batch in enumerate(scenario_batches):
        print(f"\n🔄 Processing scenario batch {batch_idx + 1}/{len(scenario_batches)}")
        memory_monitor.print_memory_status(f"batch {batch_idx + 1} start")

        # Process scenarios in this batch sequentially to control memory
        for scenario in scenario_batch:
            try:
                scenario_result = analyze_scenario_parallel(scenario, max_workers)
                scenario_info.append(scenario_result)
                total_samples += scenario_result['total_samples']
                total_size_gb += scenario_result['total_size_gb']

                # Force garbage collection after each scenario
                gc.collect()

            except Exception as e:
                print(f"  ❌ Failed to analyze {scenario.name}: {e}")
                continue

        memory_monitor.print_memory_status(f"batch {batch_idx + 1} end")
        print(f"  📊 Batch {batch_idx + 1} completed")

    overall_time = time.time() - overall_start_time

    print(f"\n📊 DATASET SUMMARY:")
    print(f"  Total scenarios: {len(scenarios)}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total size: {total_size_gb:.2f} GB")
    print(f"  Processing time: {overall_time:.2f}s")
    print(f"  Peak memory usage: {memory_monitor.peak_memory:.1f} MB")

    # Create summary DataFrame
    if scenario_info:
        df = pd.DataFrame(scenario_info)
        print(f"\n📋 Detailed breakdown:")
        print(df[['scenario', 'structure', 'total_files', 'total_samples', 'total_size_gb']].to_string(index=False))

        # Save detailed results to JSON for later use
        with open('dataset_analysis_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_scenarios': len(scenarios),
                    'total_samples': total_samples,
                    'total_size_gb': total_size_gb,
                    'processing_time': overall_time,
                    'peak_memory_mb': memory_monitor.peak_memory
                },
                'scenarios': scenario_info
            }, f, indent=2)
        print(f"📄 Detailed results saved to 'dataset_analysis_results.json'")

    return scenario_info, total_samples

def categorize_geological_complexity():
    """Categorize scenarios by geological complexity."""
    print(f"\n🏔️  GEOLOGICAL COMPLEXITY CATEGORIZATION:")
    
    complexity_mapping = {
        'simple': ['FlatVel_A', 'FlatVel_B'],  # Simple layered models
        'intermediate': ['Style_A', 'Style_B', 'CurveVel_A', 'CurveVel_B'],  # Style variations and curved velocities
        'complex': ['FlatFault_A', 'FlatFault_B', 'CurveFault_A', 'CurveFault_B']  # Fault structures
    }
    
    for complexity, scenarios in complexity_mapping.items():
        print(f"  {complexity.upper()}:")
        for scenario in scenarios:
            print(f"    - {scenario}")
    
    return complexity_mapping

def estimate_training_requirements(total_samples: int, total_size_gb: float):
    """Estimate training requirements for the full dataset with memory considerations."""
    print(f"\n⏱️  TRAINING REQUIREMENTS ESTIMATION:")

    # Current performance metrics (from previous tests)
    efficient_inference_time = 0.026  # seconds per sample
    enhanced_inference_time = 0.365   # seconds per sample

    # Memory requirements (estimated)
    sample_memory_mb = (total_size_gb * 1024) / total_samples if total_samples > 0 else 0

    print(f"  📊 Total samples: {total_samples:,}")
    print(f"  📊 Total size: {total_size_gb:.2f} GB")
    print(f"  📊 Avg sample size: {sample_memory_mb:.2f} MB")

    # Batch size recommendations based on memory
    available_memory_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
    safe_memory_gb = available_memory_gb * 0.7  # Use 70% of available memory

    # Estimate optimal batch sizes
    efficient_batch_size = max(1, int((safe_memory_gb * 1024) / (sample_memory_mb * 4)))  # 4x overhead
    enhanced_batch_size = max(1, int((safe_memory_gb * 1024) / (sample_memory_mb * 8)))   # 8x overhead

    print(f"  💾 Available memory: {available_memory_gb:.1f} GB")
    print(f"  💾 Safe memory limit: {safe_memory_gb:.1f} GB")

    # Training time estimates with batching
    efficient_batches_per_epoch = max(1, total_samples // efficient_batch_size)
    enhanced_batches_per_epoch = max(1, total_samples // enhanced_batch_size)

    efficient_epoch_time = efficient_batches_per_epoch * efficient_batch_size * efficient_inference_time * 3
    enhanced_epoch_time = enhanced_batches_per_epoch * enhanced_batch_size * enhanced_inference_time * 3

    print(f"\n  🚀 EFFICIENT MODEL:")
    print(f"    Recommended batch size: {efficient_batch_size}")
    print(f"    Batches per epoch: {efficient_batches_per_epoch}")
    print(f"    Time per epoch: {efficient_epoch_time:.1f}s ({efficient_epoch_time/60:.1f} min)")
    print(f"    Time for 20 epochs: {efficient_epoch_time * 20 / 60:.1f} min")

    print(f"\n  🎯 ENHANCED MODEL:")
    print(f"    Recommended batch size: {enhanced_batch_size}")
    print(f"    Batches per epoch: {enhanced_batches_per_epoch}")
    print(f"    Time per epoch: {enhanced_epoch_time:.1f}s ({enhanced_epoch_time/60:.1f} min)")
    print(f"    Time for 20 epochs: {enhanced_epoch_time * 20 / 60:.1f} min")

    # Parallelism recommendations
    cpu_count = mp.cpu_count()
    optimal_workers = min(cpu_count, 8)

    print(f"\n🔧 PARALLELISM RECOMMENDATIONS:")
    print(f"  CPU cores: {cpu_count}")
    print(f"  Recommended workers: {optimal_workers}")
    print(f"  DataLoader workers: {min(4, cpu_count // 2)}")

    # Memory and performance recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")

    if total_size_gb > 10:
        print(f"  📊 Large dataset ({total_size_gb:.1f}GB):")
        print(f"    - Use data streaming/lazy loading")
        print(f"    - Implement gradient accumulation")
        print(f"    - Consider mixed precision training")
        print(f"    - Use memory mapping for large files")

    if efficient_epoch_time > 1800:  # > 30 minutes
        print(f"  ⚠️  Long training times detected:")
        print(f"    - Consider GPU acceleration")
        print(f"    - Implement distributed training")
        print(f"    - Use progressive training strategy")

    if sample_memory_mb > 100:  # Large samples
        print(f"  💾 Large samples ({sample_memory_mb:.1f}MB each):")
        print(f"    - Reduce batch size")
        print(f"    - Use gradient checkpointing")
        print(f"    - Implement data compression")

    return {
        'efficient_batch_size': efficient_batch_size,
        'enhanced_batch_size': enhanced_batch_size,
        'efficient_epoch_time': efficient_epoch_time,
        'enhanced_epoch_time': enhanced_epoch_time,
        'optimal_workers': optimal_workers
    }

def load_sample_efficiently(file_path: Path, sample_idx: int = 0) -> np.ndarray:
    """Load a single sample from a file without loading the entire file."""
    try:
        # For large files, use memory mapping
        data = np.load(file_path, mmap_mode='r')

        if len(data.shape) == 4:  # Batched data
            if sample_idx < data.shape[0]:
                return np.array(data[sample_idx])  # Copy to memory
            else:
                return np.array(data[0])
        else:
            return np.array(data)
    except Exception as e:
        print(f"    ⚠️  Error loading {file_path}: {e}")
        return None

def visualize_sample_data():
    """Visualize samples from different geological scenarios with memory optimization."""
    print(f"\n🎨 CREATING SAMPLE VISUALIZATIONS (MEMORY OPTIMIZED)...")

    data_root = Path("waveform-inversion/train_samples")
    scenarios = ['FlatVel_A', 'Style_A', 'FlatFault_A']  # Representative samples

    memory_monitor = MemoryMonitor()
    memory_monitor.print_memory_status("before visualization")

    _, axes = plt.subplots(3, 3, figsize=(15, 12))

    for i, scenario_name in enumerate(scenarios):
        print(f"  🎨 Processing {scenario_name}...")
        scenario_path = data_root / scenario_name

        seismic_sample = None
        velocity_sample = None

        try:
            # Load data based on structure with memory optimization
            data_dir = scenario_path / "data"
            model_dir = scenario_path / "model"

            if data_dir.exists():
                data_files = list(data_dir.glob("*.npy"))
                model_files = list(model_dir.glob("*.npy"))

                if data_files and model_files:
                    # Use memory-efficient loading
                    seismic_data = load_sample_efficiently(data_files[0], 0)
                    velocity_data = load_sample_efficiently(model_files[0], 0)

                    if seismic_data is not None and velocity_data is not None:
                        # Handle different data structures
                        if len(seismic_data.shape) == 3:  # (sources, time, receivers)
                            seismic_sample = seismic_data[0, :, :]  # First source
                        else:
                            seismic_sample = seismic_data

                        if len(velocity_data.shape) == 3:  # (channels, height, width)
                            velocity_sample = velocity_data[0, :, :]  # First channel
                        else:
                            velocity_sample = velocity_data
            else:
                # Direct files
                seis_files = list(scenario_path.glob("seis*.npy"))
                vel_files = list(scenario_path.glob("vel*.npy"))

                if seis_files and vel_files:
                    seismic_data = load_sample_efficiently(seis_files[0], 0)
                    velocity_data = load_sample_efficiently(vel_files[0], 0)

                    if seismic_data is not None and velocity_data is not None:
                        seismic_sample = seismic_data[0, :, :] if len(seismic_data.shape) == 3 else seismic_data
                        velocity_sample = velocity_data

            # Plot only if data was successfully loaded
            if seismic_sample is not None and velocity_sample is not None:
                # Plot seismic data
                axes[i, 0].imshow(seismic_sample, aspect='auto', cmap='seismic')
                axes[i, 0].set_title(f'{scenario_name}\nSeismic Data')
                axes[i, 0].set_xlabel('Receiver')
                axes[i, 0].set_ylabel('Time')

                # Plot velocity model
                im = axes[i, 1].imshow(velocity_sample, cmap='jet', vmin=1500, vmax=4500)
                axes[i, 1].set_title(f'{scenario_name}\nVelocity Model')
                axes[i, 1].set_xlabel('X')
                axes[i, 1].set_ylabel('Depth')
                plt.colorbar(im, ax=axes[i, 1], label='Velocity (m/s)')

                # Plot depth profile
                center_x = velocity_sample.shape[1] // 2
                depth_profile = velocity_sample[:, center_x]
                depth = np.arange(len(depth_profile))

                axes[i, 2].plot(depth_profile, depth, 'b-', linewidth=2)
                axes[i, 2].set_xlabel('Velocity (m/s)')
                axes[i, 2].set_ylabel('Depth')
                axes[i, 2].set_title(f'{scenario_name}\nDepth Profile')
                axes[i, 2].invert_yaxis()
                axes[i, 2].grid(True, alpha=0.3)

                print(f"    ✅ {scenario_name} visualization completed")
            else:
                print(f"    ❌ Failed to load data for {scenario_name}")
                # Create empty plots
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, f'No data\n{scenario_name}',
                                   ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].set_title(f'{scenario_name}\nNo Data')

            # Force garbage collection after each scenario
            gc.collect()

        except Exception as e:
            print(f"    ❌ Error processing {scenario_name}: {e}")
            # Create error plots
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f'Error\n{scenario_name}',
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'{scenario_name}\nError')

    plt.tight_layout()
    plt.savefig('full_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    memory_monitor.print_memory_status("after visualization")
    print(f"📊 Sample visualization saved to 'full_dataset_analysis.png'")

def main():
    """Main analysis function with memory optimization and parallelism."""
    print("🚀 STARTING OPTIMIZED DATASET ANALYSIS")
    print("="*60)

    start_time = time.time()
    memory_monitor = MemoryMonitor()

    try:
        # System information
        print(f"💻 System Info:")
        print(f"  CPUs: {mp.cpu_count()}")
        print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"  Available: {psutil.virtual_memory().available / 1024**3:.1f} GB")

        # Analyze dataset structure with parallelism
        print(f"\n📊 Phase 1: Dataset Structure Analysis")
        scenario_info, total_samples = analyze_dataset_structure()

        if not scenario_info:
            print("❌ No data found. Exiting.")
            return None, None, 0

        # Calculate total size
        total_size_gb = sum(info.get('total_size_gb', 0) for info in scenario_info)

        # Categorize by complexity
        print(f"\n📊 Phase 2: Geological Complexity Categorization")
        complexity_mapping = categorize_geological_complexity()

        # Estimate training requirements with memory considerations
        print(f"\n📊 Phase 3: Training Requirements Estimation")
        training_reqs = estimate_training_requirements(total_samples, total_size_gb)

        # Create visualizations (memory optimized)
        print(f"\n📊 Phase 4: Sample Visualizations")
        try:
            visualize_sample_data()
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
            print("Continuing without visualizations...")

        # Final summary
        total_time = time.time() - start_time
        memory_monitor.print_memory_status("final")

        print(f"\n🎉 OPTIMIZED DATASET ANALYSIS COMPLETED!")
        print(f"⏱️  Total analysis time: {total_time:.2f}s")
        print(f"📊 Dataset summary:")
        print(f"  - Scenarios: {len(scenario_info)}")
        print(f"  - Total samples: {total_samples:,}")
        print(f"  - Total size: {total_size_gb:.2f} GB")
        print(f"  - Peak memory: {memory_monitor.peak_memory:.1f} MB")

        print(f"\n💡 Next steps:")
        print(f"  1. Use batch size {training_reqs['efficient_batch_size']} for efficient model")
        print(f"  2. Use batch size {training_reqs['enhanced_batch_size']} for enhanced model")
        print(f"  3. Use {training_reqs['optimal_workers']} workers for parallel processing")
        print(f"  4. Implement data streaming for memory efficiency")

        # Save comprehensive results
        results = {
            'analysis_summary': {
                'total_scenarios': len(scenario_info),
                'total_samples': total_samples,
                'total_size_gb': total_size_gb,
                'analysis_time': total_time,
                'peak_memory_mb': memory_monitor.peak_memory
            },
            'scenarios': scenario_info,
            'complexity_mapping': complexity_mapping,
            'training_requirements': training_reqs
        }

        with open('comprehensive_dataset_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"📄 Comprehensive results saved to 'comprehensive_dataset_analysis.json'")

        return scenario_info, complexity_mapping, total_samples

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0

if __name__ == "__main__":
    main()
