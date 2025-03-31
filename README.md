# 📜 Renaissance Text GAN

<div align="center">
  <img src="final_results/text_comparison_1.png" alt="Renaissance Text Sample" width="700">
  <p><i>Generated Renaissance-style text with realistic printing imperfections</i></p>
</div>

## 🖋️ Overview

This project implements a Generative Adversarial Network (GAN) for creating synthetic Renaissance-style printed text with authentic historical printing imperfections. I've designed this system to generate text samples that mimic the characteristics of 17th-century Spanish texts, featuring:

- **Ink bleeding** at character edges
- **Smudging effects** that distort text clarity
- **Faded sections** that simulate uneven ink application
- **Paper texture and grain** that replicate aged parchment

## 📊 Results

<div align="center">
  <table>
    <tr>
      <td><img src="![image](https://github.com/user-attachments/assets/2422e53e-4a2e-4a36-acf9-8f64328f4422)" width="300"></td>
      <td><img src="final_results/degraded_text_2.png" width="300"></td>
    </tr>
    <tr>
      <td align="center"><b>Clean Text</b></td>
      <td align="center"><b>Renaissance Degraded Text</b></td>
    </tr>
  </table>
</div>

<div align="center">
  <img src="final_results_heavy/text_comparison_3.png" alt="Heavy Degradation Example" width="700">
  <p><i>Example with more pronounced degradation effects</i></p>
</div>

## 🏗️ Architecture

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

## 📊 Evaluation Metrics

I evaluate the quality of generated Renaissance text using:

1. **FID (Fréchet Inception Distance)** - Measures the similarity between generated and real images
2. **LPIPS (Learned Perceptual Image Patch Similarity)** - Quantifies perceptual similarity
3. **Historical Artifact Realism Score** - A custom metric that quantifies:
   - Ink bleeding authenticity
   - Text irregularity levels
   - Paper texture realism

## 🚀 Getting Started

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

## 🏁 Project Structure

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

## 📝 Implementation Details

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

### Training Process

The GAN is trained adversarially:
1. The generator creates clean text images
2. The degradation layer applies Renaissance-style imperfections
3. The discriminator learns to distinguish real vs. generated degraded samples
4. Both networks improve iteratively to create more realistic outputs

## 🔍 Future Work

- Integrate more diverse text samples from various languages and historical periods
- Expand the range of degradation effects to include tears, water damage, and stains
- Develop a conditional GAN to control specific degradation parameters
- Create a web interface for interactive text degradation

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The PyTorch team for their excellent deep learning framework
- Historical libraries for providing reference materials
- The academic community researching document degradation models 
