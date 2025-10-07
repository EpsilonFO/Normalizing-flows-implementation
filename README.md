# Normalizing Flows: Interactive Implementation

An interactive Streamlit application for exploring and experimenting with Normalizing Flows models. This educational project provides hands-on demonstrations of three major normalizing flow architectures: NICE, RealNVP, and Glow.

## 📖 About Normalizing Flows

Normalizing Flows are a class of generative models that learn to transform a simple distribution (typically Gaussian) into a complex target distribution through a sequence of invertible transformations. Unlike VAEs and GANs, normalizing flows provide:

- **Exact likelihood computation**: Calculate the exact probability of data points
- **Invertible transformations**: Both generation and inference are possible
- **Tractable training**: Direct optimization via maximum likelihood

The core principle relies on the change of variables theorem. Given a random variable `z` with known density `p(z)` and a bijective function `f`, the density of the transformed variable `x = f(z)` is:

```
p(x) = p(z) |det(∂f/∂z)⁻¹|
```

## 🎯 Implemented Models

### NICE (Non-linear Independent Components Estimation)
- **Year**: 2014
- **Innovation**: Additive coupling layers with unit Jacobian determinant
- **Transformation**: `y₂ = x₂ + m(x₁)`
- **Advantages**: Simple, efficient, volume-preserving
- **Limitations**: Limited expressivity due to additive-only transformations

### RealNVP (Real-valued Non-Volume Preserving)
- **Year**: 2017
- **Innovation**: Affine coupling layers with scaling and translation
- **Transformation**: `y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁)`
- **Advantages**: More expressive, multi-scale architecture, checkerboard masking
- **Limitations**: Fixed permutations, requires many layers for complex distributions

### Glow
- **Year**: 2018
- **Innovation**: Invertible 1×1 convolutions, ActNorm layers
- **Architecture**: Multi-level flow with squeeze operations
- **Advantages**: State-of-the-art image generation, semantic manipulation
- **Limitations**: Computationally expensive, high memory requirements

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/EpsilonFO/Normalizing-flows-implementation.git
cd Normalizing-flows-implementation
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## 💡 How to Use

The application provides an intuitive interface with four main sections:

### 1. Presentation
- Overview of normalizing flows concepts
- Mathematical foundations
- Comparison of different architectures

### 2. NICE Model
- Interactive training on 2D distributions (Two Moons)
- Adjustable hyperparameters:
  - Number of coupling layers (2-12)
  - Hidden layer dimensions (32-256)
  - Training iterations (1000-5000)
- Real-time visualization of learned distributions

### 3. RealNVP Model
- Training on Two Moons distribution
- Configurable parameters:
  - Number of coupling blocks (2-12)
  - Training iterations (1000-5000)
- Step-by-step visualization during training
- Side-by-side comparison with target distribution

### 4. Glow Model
- Training on MNIST dataset (28×28 grayscale images)
- Advanced configuration:
  - Multi-scale levels L (1-4)
  - Flow steps per level K (4-32)
  - Batch size (32-256)
  - Training iterations (1000-20000)
- Sample generation across all digit classes
- **Note**: Training can take several hours without GPU acceleration

## 📁 Project Structure

```
Normalizing-flows-implementation/
├── app.py                      # Main Streamlit application entry point
├── presentation.py             # Presentation page with theory
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── nice/                       # NICE model implementation
│   ├── nice.py                 # Streamlit interface for NICE
│   ├── train_nice.py           # Model architecture and training logic
│   └── images/                 # Visualization assets
│
├── realnvp/                    # RealNVP model implementation
│   ├── real_nvp.py             # Streamlit interface for RealNVP
│   ├── train_realnvp.py        # Model architecture and training logic
│   └── images/                 # Visualization assets
│
├── glow/                       # Glow model implementation
│   ├── glow.py                 # Streamlit interface for Glow
│   ├── train_glow.py           # Model architecture and training logic
│   └── images/                 # Visualization assets
│
└── docs/                       # Sphinx documentation
    ├── conf.py                 # Sphinx configuration
    ├── index.rst               # Documentation index
    └── *.rst                   # Module documentation files
```

## 🔧 Technical Details

### Model Architectures

All models implement:
- Forward transformation (data → latent space)
- Inverse transformation (latent space → data)
- Log-determinant Jacobian computation
- Efficient likelihood estimation

### Training Process

- **Optimizer**: Adam with weight decay
- **Loss function**: Negative log-likelihood (KL divergence minimization)
- **Gradient handling**: NaN/Inf detection and skipping
- **Visualization**: Real-time distribution plots during training

### Dependencies

Key libraries:
- `torch` & `torchvision`: Deep learning framework
- `normflows`: Normalizing flows utilities
- `streamlit`: Web application framework
- `matplotlib` & `seaborn`: Visualization
- `numpy` & `scipy`: Numerical computing
- `pillow`: Image processing

## 📚 Documentation

Generate comprehensive documentation using Sphinx:

```bash
cd docs
make html
```

View the documentation by opening `docs/_build/html/index.html` in your browser.

## 🛠️ Development

### Code Quality

The project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run checks manually
pre-commit run --all-files
```

Configuration includes:
- **Black**: Code formatting (max line length: 100)
- **Flake8**: Linting (ignoring E203, E261, W503)
- **YAML validation**: Configuration file checking

### Adding New Models

To add a new normalizing flow model:

1. Create a new directory (e.g., `newmodel/`)
2. Implement `newmodel.py` with Streamlit interface
3. Implement `train_newmodel.py` with model architecture
4. Update `app.py` to include the new model
5. Add documentation in `docs/`

## 📖 References

### Papers

- **NICE**: [Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516) (Dinh et al., 2014)
- **RealNVP**: [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803) (Dinh et al., 2017)
- **Glow**: [Generative Flow with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039) (Kingma & Dhariwal, 2018)

### Additional Resources

- [Normalizing Flows Tutorial](https://arxiv.org/abs/1912.02762)
- [VincentStimper/normalizing-flows](https://github.com/VincentStimper/normalizing-flows) - PyTorch implementation
- [OpenAI Glow Repository](https://github.com/openai/glow) - Official Glow implementation

## 👥 Authors

- [@MohammedLbkl](https://github.com/MohammedLbkl)
- [@EpsilonFO](https://github.com/EpsilonFO)

## 📝 License

This project is an educational implementation for learning purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Bug fixes
- New model implementations
- Documentation improvements
- Performance optimizations

## ⚠️ Notes

- **GPU Recommended**: Training Glow on MNIST without GPU can take several hours
- **Memory Usage**: Large batch sizes may require significant RAM
- **Datasets**: MNIST will be automatically downloaded to `datasets/` on first run
- **Browser Compatibility**: Best viewed in Chrome, Firefox, or Safari
