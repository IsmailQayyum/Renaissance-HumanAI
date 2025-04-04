# Renaissance Text GAN

![image](https://github.com/user-attachments/assets/03ce9173-b815-4af7-978b-25769a7c72c5)


<hr>

## Overview

This project implements a Generative Adversarial Network (GAN) for creating synthetic Renaissance-style printed text with authentic historical printing imperfections. I've designed this system to generate text samples that mimic the characteristics of 17th-century Spanish texts, featuring:

- **Ink bleeding** at character edges
- **Smudging effects** that distort text clarity
- **Faded sections** that simulate uneven ink application
- **Paper texture and grain** that replicate aged parchment

## Results

![image](https://github.com/user-attachments/assets/73413db3-f5ac-482f-844b-10e751da7899)

![image](https://github.com/user-attachments/assets/922631b4-98ae-404e-856a-79b29656856e)




<div align="center">
  <p><em>Example with more pronounced degradation effects</em></p>
</div>

![image](https://github.com/user-attachments/assets/060d9ed1-ca73-4270-ac99-2aa28038624d)

<hr>

## Architecture

I've implemented this project using PyTorch with a dual-component architecture:

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

I evaluate the quality of generated Renaissance text using:

1. **FID (Fréchet Inception Distance)** - Measures the similarity between generated and real images
2. **LPIPS (Learned Perceptual Image Patch Similarity)** - Quantifies perceptual similarity
3. **Historical Artifact Realism Score** - A custom metric that quantifies:
   - Ink bleeding authenticity
   - Text irregularity levels
   - Paper texture realism

<hr>

### Generated Images

![image](https://github.com/user-attachments/assets/836d5284-c91c-4bf2-bb15-4177edcd253f)

![image](https://github.com/user-attachments/assets/d5b98dd4-3c06-47ac-9d96-d430e79a2267)



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

<hr>

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

<hr>

## Implementation Details

### Data Preprocessing

I use text samples from classical Spanish literature (primarily Don Quixote) and render them with:
- Appropriate Renaissance-style fonts
- Page layout typical of the period
- Various font sizes and densities

### Degradation Effects

The degradation process applies several transformations:
- **Gaussian blur** for ink bleeding simulation
- **Median filters** for smudging effects
- **Contrast adjustments** for faded text
- **Noise injection** for paper grain
- **Random masks** for localized degradation


<hr>
