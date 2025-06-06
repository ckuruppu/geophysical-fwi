# Geophysical Waveform Full Waveform Inversion (FWI) Dataset Analysis

A comprehensive analysis toolkit for the OpenFWI geophysical dataset, designed for Full Waveform Inversion research and development.

## Overview

This repository contains analysis tools and documentation for understanding and working with the OpenFWI dataset, which consists of seismic waveform recordings and corresponding velocity maps for machine learning applications in geophysics.

## Dataset Description

The dataset is derived from OpenFWI, a large-scale benchmark dataset for seismic imaging tasks. It contains three main families of data:

### Dataset Families

1. **Vel Family** (`FlatVel_A/B`, `CurveVel_A/B`)
   - Simple layered velocity models
   - Ideal for initial algorithm development
   - Structure: `data/` (seismic) and `model/` (velocity) subdirectories

2. **Fault Family** (`FlatFault_A/B`, `CurveFault_A/B`)
   - Complex models with lateral heterogeneity and fault structures
   - Advanced testing scenarios
   - Structure: `seis_*.npy` (seismic) and `vel_*.npy` (velocity) files

3. **Style Family** (`Style_A/B`)
   - Intermediate complexity models
   - Various geological styles
   - Structure: `data/` (seismic) and `model/` (velocity) subdirectories

### Data Dimensions

| Dataset Type | Shape | Description |
|--------------|-------|-------------|
| Training Seismic | `(500, 5, 1000, 70)` | batch × sources × time_steps × receivers |
| Training Velocity | `(500, 1, 70, 70)` | batch × layers × height × width |
| Test Seismic | `(5, 1000, 70)` | sources × time_steps × receivers |

## Key Characteristics

### Temporal Properties
- **Time Steps**: 1,000 per recording
- **Recording Duration**: 2-4 seconds (estimated)
- **Dominant Frequencies**: 3.9-7.8 Hz
- **Sampling**: High temporal resolution suitable for FWI

### Spatial Properties
- **Model Grid**: 70 × 70 points
- **Physical Coverage**: ~0.7-1.8 km × 0.7-1.8 km
- **Receivers**: 70 per shot (1.7-3.5 km array length)
- **Sources**: 5 per model

### Velocity Characteristics
- **Range**: 1,501-4,500 m/s (meters per second)
- **Physical Interpretation**: Sedimentary to crystalline rock velocities
- **Vel Family**: Strong depth gradients, minimal lateral variation
- **Fault Family**: Moderate depth gradients, significant lateral heterogeneity

## Repository Structure

```
├── README.md                     # This file
├── data_analysis.py             # Comprehensive data analysis script
├── enhanced_data_analysis.py    # Detailed analysis with geological insights
├── data_loader_example.py       # Example data loading utilities
├── data_analysis_report.md      # Complete analysis report
├── data_summary_statistics.csv  # Statistical summary
└── waveform-inversion/          # Dataset directory
    ├── train_samples/           # Training data
    │   ├── FlatVel_A/          # Vel family data
    │   ├── FlatFault_A/        # Fault family data
    │   └── Style_A/            # Style family data
    ├── test/                   # Test data (2000+ files)
    └── sample_submission.csv   # Submission format example
```

## Installation and Setup

### Prerequisites
```bash
pip install numpy matplotlib pandas scipy
```

### Dataset Setup
1. Download the OpenFWI dataset
2. Extract to `waveform-inversion/` directory
3. Ensure the following structure exists:
   ```
   waveform-inversion/
   ├── train_samples/
   ├── test/
   └── sample_submission.csv
   ```

## Usage

### Quick Start Analysis
Run the comprehensive data analysis:
```bash
python3 data_analysis.py
```

This generates:
- Statistical summaries
- Frequency analysis plots
- Spatial characteristic visualizations
- Data quality assessments

### Enhanced Analysis
For detailed geological insights:
```bash
python3 enhanced_data_analysis.py
```

Provides:
- Signal quality analysis
- Velocity model interpretation
- Geological context
- Comprehensive visualizations

### Data Loading Examples
```python
from data_loader_example import load_vel_style_data, load_fault_data

# Load Vel family data
seismic_data, velocity_models = load_vel_style_data("FlatVel_A")

# Load Fault family data
fault_seismic, fault_velocity = load_fault_data("FlatFault_A")

# Basic usage
print(f"Seismic shape: {seismic_data.shape}")
print(f"Velocity range: {velocity_models.min():.0f}-{velocity_models.max():.0f} m/s")
```

### Working with Different Families

#### Vel/Style Families (data/model structure)
```python
import numpy as np
from pathlib import Path

# Load data
data_path = Path("waveform-inversion/train_samples/FlatVel_A")
seismic = np.load(data_path / "data/data1.npy")
velocity = np.load(data_path / "model/model1.npy")

# Data shapes
# seismic: (500, 5, 1000, 70) - batch, sources, time, receivers
# velocity: (500, 1, 70, 70) - batch, layers, height, width
```

#### Fault Families (seis_/vel_ structure)
```python
# Load fault data
data_path = Path("waveform-inversion/train_samples/FlatFault_A")
seismic = np.load(data_path / "seis2_1_0.npy")
velocity = np.load(data_path / "vel2_1_0.npy")

# Same shapes as above
```

#### Test Data
```python
# Load test data
test_path = Path("waveform-inversion/test")
test_files = list(test_path.glob("*.npy"))
test_sample = np.load(test_files[0])

# test_sample shape: (5, 1000, 70) - sources, time, receivers
```

## Analysis Results Summary

### Data Quality Metrics
- **Memory Usage**: ~667 MB per seismic dataset, ~9 MB per velocity model
- **Value Ranges**:
  - Seismic: -25.8 to +50.2 (normalized amplitudes)
  - Velocity: 1,501 to 4,500 m/s (realistic geological range)
- **Signal Quality**: High SNR with clear first arrivals
- **Completeness**: No missing values or data corruption

### Geological Interpretation

#### Vel Family Models
- **Characteristics**: Simple layered Earth models
- **Velocity Gradient**: 36.6 m/s per grid point (strong increase with depth)
- **Lateral Variation**: Minimal (0.0 m/s average heterogeneity)
- **Use Case**: Algorithm development and initial testing

#### Fault Family Models
- **Characteristics**: Complex heterogeneous models with fault structures
- **Velocity Gradient**: 14.3 m/s per grid point (moderate increase)
- **Lateral Variation**: Significant (24.6 m/s average, up to 285 m/s)
- **Use Case**: Robustness testing and advanced algorithm validation

### Frequency Content
- **Dominant Frequencies**: 3.9-7.8 Hz (depending on sampling rate)
- **Bandwidth**: Suitable for FWI applications
- **Temporal Resolution**: 1000 time steps providing detailed waveform sampling

## Applications

### Full Waveform Inversion (FWI)
- **Multi-scale Approach**: Use frequency content for hierarchical inversion
- **Regularization**: Leverage gradient information for geological constraints
- **Validation**: Test across different geological complexity levels

### Machine Learning
- **Training Data**: 500 samples per family provide good diversity
- **Batch Processing**: Efficient structure for deep learning applications
- **Transfer Learning**: Progress from simple to complex models

### Seismic Processing
- **Algorithm Development**: Test processing workflows
- **Quality Control**: Validate processing parameters
- **Benchmarking**: Compare different methodologies

## Best Practices

### Data Preprocessing
1. **Normalization**: Consider amplitude normalization for consistent training
2. **Frequency Filtering**: Apply appropriate bandpass filters
3. **Quality Control**: Check for outliers and data integrity

### Model Development
1. **Start Simple**: Begin with Vel family for algorithm development
2. **Increase Complexity**: Progress to Fault family for robustness testing
3. **Validate Thoroughly**: Use Style family for generalization testing

### Performance Optimization
1. **Memory Management**: Load data in batches for large-scale processing
2. **Parallel Processing**: Leverage multiple sources for concurrent processing
3. **GPU Acceleration**: Utilize GPU computing for intensive calculations

## Generated Files

Running the analysis scripts creates:
- `frequency_analysis.png` - Spectral analysis plots
- `spatial_analysis.png` - Velocity model visualizations
- `receiver_source_analysis.png` - Acquisition geometry analysis
- `depth_profiles.png` - Velocity-depth relationships
- `comprehensive_analysis.png` - Complete overview plots
- `data_summary_statistics.csv` - Numerical summary table
- `data_analysis_report.md` - Detailed analysis report

## Contributing

Contributions are welcome! Please feel free to submit pull requests for:
- Additional analysis tools
- Improved visualization methods
- Performance optimizations
- Documentation enhancements

## References

- OpenFWI: Large-scale benchmark dataset for seismic imaging
- Full Waveform Inversion methodology and applications
- Seismic data processing and interpretation techniques

## License

This analysis toolkit is provided for research and educational purposes. Please refer to the original OpenFWI dataset license for data usage terms.
