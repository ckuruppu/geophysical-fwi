# Hybrid CNN-RNN Architecture for Seismic Waveform Inversion

A comprehensive deep learning implementation for Full Waveform Inversion (FWI) using a hybrid CNN-RNN architecture specifically designed for the OpenFWI dataset.

## 🏗️ Architecture Overview

### Why Hybrid CNN-RNN?

**Seismic data has both spatial and temporal structure:**
- **Temporal**: Wave propagation through time (1000 time steps)
- **Spatial**: Receiver arrays and geological structures (70×70 velocity models)

**Our hybrid approach:**
1. **RNN Component**: Processes temporal sequences to extract wave propagation patterns
2. **CNN Component**: Processes spatial receiver arrays to extract geological structures
3. **Fusion Module**: Combines multi-source information using attention mechanisms

### Architecture Components

```
Input: Seismic Data (batch, 5_sources, 1000_time, 70_receivers)
    ↓
[Temporal Encoder - Bidirectional LSTM + Attention]
    ↓ (batch, sources, receivers, temporal_features)
[Spatial Encoder - Multi-scale CNN + Spatial Attention]
    ↓ (batch, sources, spatial_features)
[Source Fusion - Multi-head Attention]
    ↓ (batch, fused_features)
[Velocity Decoder - Transpose Convolutions]
    ↓
Output: Velocity Model (batch, 1, 70, 70)
```

## 📁 File Structure

```
├── seismic_inversion_model.py      # Main hybrid architecture
├── physics_informed_losses.py     # Physics-based loss functions
├── seismic_data_loader.py         # Data loading and preprocessing
├── train_seismic_inversion.py     # Training pipeline
├── evaluate_model.py              # Model evaluation and visualization
├── demo_hybrid_architecture.py    # Demo and testing script
└── HYBRID_CNN_RNN_README.md      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib pandas scipy seaborn
pip install tensorboard
```

### 2. Test the Architecture

```bash
python demo_hybrid_architecture.py
```

This will:
- Test the model architecture with synthetic data
- Verify data loading (if OpenFWI dataset available)
- Demonstrate physics-informed loss functions
- Create visualization of results

### 3. Train the Model

```bash
python train_seismic_inversion.py
```

Features:
- **Progressive Training**: Starts with simple models, progresses to complex
- **Physics-Informed Losses**: Incorporates seismic wave physics
- **Comprehensive Logging**: TensorBoard integration
- **Automatic Checkpointing**: Saves best models

### 4. Evaluate Results

```bash
python evaluate_model.py
```

Provides:
- Quantitative metrics (RMSE, MAE, structural similarity)
- Prediction visualizations
- Error analysis
- Velocity profile comparisons

## 🧠 Model Architecture Details

### Temporal Encoder (RNN Component)

```python
class TemporalEncoder(nn.Module):
    - Bidirectional LSTM (2 layers, 64 hidden units)
    - Multi-head attention (8 heads)
    - Processes each seismic trace through time
    - Extracts: first breaks, wave arrivals, amplitude patterns
```

**Why LSTM?**
- Captures temporal dependencies in wave propagation
- Bidirectional processing sees both early and late arrivals
- Attention focuses on important time steps (first breaks, reflections)

### Spatial Encoder (CNN Component)

```python
class SpatialEncoder(nn.Module):
    - Multi-scale 1D convolutions (kernels: 3, 5, 7)
    - Spatial attention mechanism
    - Processes receiver arrays for each source
    - Extracts: coherent arrivals, velocity gradients, geological boundaries
```

**Why CNN?**
- Detects spatial patterns across receiver arrays
- Multi-scale kernels capture different geological features
- Translation invariant (same feature detected anywhere)

### Source Fusion Module

```python
class SourceFusionModule(nn.Module):
    - Multi-head attention across 5 sources
    - Learns optimal combination of source information
    - Handles varying source-receiver geometries
```

### Velocity Decoder

```python
class VelocityDecoder(nn.Module):
    - Transpose convolutions for upsampling
    - Skip connections preserve fine details
    - Generates 70×70 velocity models
```

## 🔬 Physics-Informed Loss Functions

### 1. Basic MSE Loss
```python
mse_loss = MSE(predicted_velocity, target_velocity)
```

### 2. Smoothness Regularization
```python
# Penalizes unrealistic velocity variations
smoothness_loss = ||∇velocity||²
```

### 3. Gradient Preservation
```python
# Ensures structural features are preserved
gradient_loss = MSE(∇predicted, ∇target)
```

### 4. Geological Realism
```python
# Enforces realistic velocity ranges and depth trends
geological_loss = depth_penalty + range_penalty
```

### 5. Travel Time Consistency
```python
# Ensures predictions are consistent with seismic arrivals
travel_time_loss = consistency_check(velocity, seismic_data)
```

## 📊 Training Strategy

### Progressive Training Schedule

1. **Simple Phase** (20 epochs): FlatVel families only
   - Learn basic layered velocity structures
   - Establish fundamental wave propagation patterns

2. **Intermediate Phase** (15 epochs): Add Style families
   - Introduce geological complexity
   - Learn structural variations

3. **Complex Phase** (15 epochs): Add Fault families
   - Handle lateral heterogeneity
   - Learn fault detection and characterization

4. **Full Phase** (10 epochs): All families
   - Final refinement on complete dataset
   - Robust generalization

### Data Augmentation

- **Noise Addition**: Realistic SNR levels
- **Time Shifting**: Simulate timing uncertainties
- **Amplitude Scaling**: Account for acquisition variations
- **Receiver Dropout**: Handle missing data

## 📈 Expected Performance

Based on the data analysis, the model should achieve:

- **RMSE**: < 200 m/s (typical seismic inversion accuracy)
- **Relative Error**: < 10% for most velocity ranges
- **Structural Similarity**: > 0.8 correlation with true models
- **Inference Time**: ~50ms per sample (GPU)

## 🔧 Customization Options

### Architecture Modifications

```python
# Adjust model capacity
model = SeismicInversionNet(
    temporal_hidden_size=128,    # Increase for more temporal detail
    spatial_output_channels=256, # Increase for more spatial detail
    num_sources=5,               # Match your data
    output_size=(70, 70)         # Match your velocity grid
)
```

### Loss Function Weights

```python
# Adjust physics constraints
criterion = PhysicsInformedLoss(
    mse_weight=1.0,           # Basic accuracy
    smoothness_weight=0.2,    # Increase for smoother models
    gradient_weight=0.1,      # Increase for better structure
    travel_time_weight=0.3,   # Increase for physics consistency
    geological_weight=0.15    # Increase for realism
)
```

## 🎯 Key Features

### 1. **Multi-Scale Processing**
- Temporal: LSTM captures wave propagation sequences
- Spatial: Multi-scale CNNs detect geological features
- Integration: Attention mechanisms combine information

### 2. **Physics Integration**
- Travel time consistency checks
- Geological realism constraints
- Gradient preservation for structural accuracy

### 3. **Robust Training**
- Progressive complexity increase
- Comprehensive data augmentation
- Adaptive loss weighting

### 4. **Comprehensive Evaluation**
- Multiple accuracy metrics
- Error analysis and visualization
- Geological interpretation tools

## 🔍 Understanding the Results

### Velocity Model Interpretation

- **High Velocities (>4000 m/s)**: Crystalline basement rocks
- **Medium Velocities (2000-4000 m/s)**: Sedimentary layers
- **Low Velocities (<2000 m/s)**: Shallow sediments, water

### Common Challenges

1. **Low SNR Data**: Use stronger smoothness regularization
2. **Complex Geology**: Increase model capacity and training time
3. **Limited Data**: Apply more aggressive data augmentation
4. **Computational Limits**: Reduce batch size or model size

## 📚 References

This implementation is based on:
- OpenFWI dataset structure and characteristics
- Modern deep learning best practices for geophysics
- Physics-informed neural network principles
- Hybrid architectures for spatiotemporal data

## 🤝 Contributing

To extend this implementation:
1. Add new geological families to the training data
2. Implement additional physics constraints
3. Experiment with different attention mechanisms
4. Add uncertainty quantification
5. Implement real-time inference capabilities

---

**Note**: This implementation is designed specifically for the OpenFWI dataset structure. Adaptation may be needed for other seismic datasets.
