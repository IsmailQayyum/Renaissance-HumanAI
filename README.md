# Renaissance Text GAN

![Generated Sample](https://github.com/user-attachments/assets/e5166fde-d083-4f33-a335-c4040a35b12c)

## Overview

This project implements a Generative Adversarial Network (GAN) to create synthetic Renaissance-style printed text with authentic historical printing imperfections. The system is designed to generate text samples that mimic 17th-century Spanish texts, featuring:

- **Ink bleeding** at character edges
- **Smudging effects** that distort text clarity
- **Faded sections** that simulate uneven ink application
- **Paper texture and grain** that replicate aged parchment

## Generated Results

<p align="center">
  <img src="https://github.com/user-attachments/assets/9e32e089-2c78-4d1b-8204-69ee05a19302" alt="Generated Example 1" width="600">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/763665ab-26ec-48ea-a5dd-2e228bbf0ef3" alt="Generated Example 2" width="600">
</p>

<p align="center"><i>Example with more pronounced degradation effects</i></p>

## Architecture

This project is implemented using PyTorch with a dual-component architecture:

### Generator Network
- Residual block architecture for high-quality generation
- Upsampling layers to create high-resolution output
- Specialized for generating text-like structures

### Degradation System
- Custom degradation layer with learnable parameters
- Convolutional operations for ink bleeding effects
- Spatial transformations for smudging
- Noise injection for paper texture simulation

## Evaluation Metrics

To assess the quality of generated Renaissance text, the following metrics are used:

1. **FID (Fréchet Inception Distance)** - Measures the similarity between generated and real images.
2. **LPIPS (Learned Perceptual Image Patch Similarity)** - Quantifies perceptual similarity.
3. **Historical Artifact Realism Score** - A custom metric evaluating:
   - Ink bleeding authenticity
   - Text irregularity levels
   - Paper texture realism

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/renaissance-text-gan.git
cd renaissance-text-gan

# Install dependencies
pip install -e .
```

### Quick Demo

```bash
# Run the demo script
python run_demo.py --output_dir results
```

### Generate Samples

```bash
# Generate Renaissance text samples with varying degradation levels
python generate.py --render_text --output_dir samples --num_samples 5 --save_comparison --degradation_intensity 0.7
```

### Train Your Own Model

```bash
# Create a dataset
python data/create_dataset.py --output_dir dataset --num_samples 1000

# Train the GAN model
python train.py --data_dir dataset/degraded --output_dir model --num_epochs 100
```

## Project Structure

```
renaissance_text_gan/
├── data/                  # Dataset creation and handling
├── models/                # GAN architecture implementation
│   ├── gan_models.py      # Generator and discriminator models
│   └── combined_model.py  # Full generation pipeline
├── utils/                 # Utility functions
│   ├── data_utils.py      # Data loading and transformations
│   └── eval_metrics.py    # Evaluation metrics
├── train.py               # Training script
├── generate.py            # Generation script
└── run_demo.py            # Demo script
```

## Implementation Details

### Data Preprocessing

Text samples are taken from classical Spanish literature (primarily *Don Quixote*) and rendered with:
- Appropriate Renaissance-style fonts
- Page layouts typical of the period
- Various font sizes and densities

### Degradation Effects

The degradation process applies several transformations:
- **Gaussian blur** for ink bleeding simulation
- **Median filters** for smudging effects
- **Contrast adjustments** for faded text
- **Noise injection** for paper grain
- **Random masks** for localized degradation

### Training Process

The GAN is trained adversarially:
1. The generator creates clean text images.
2. The degradation layer applies Renaissance-style imperfections.
3. The discriminator learns to distinguish real vs. generated degraded samples.
4. Both networks improve iteratively to create more realistic outputs.

## Future Work

- Integrate more diverse text samples from various languages and historical periods.
- Expand the range of degradation effects to include tears, water damage, and stains.
- Develop a conditional GAN to control specific degradation parameters.
- Create a web interface for interactive text degradation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The PyTorch team for their excellent deep learning framework.
- Historical libraries for providing reference materials.
- The academic community researching document degradation models.
