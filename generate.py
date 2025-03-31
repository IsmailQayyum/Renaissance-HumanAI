import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.gan_models import Generator, DegradationLayer
from utils.data_utils import add_renaissance_degradations, save_samples

def generate_samples(args):
    """Generate synthetic Renaissance text samples using the trained models."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    generator = Generator(
        latent_dim=args.latent_dim,
        channels=args.g_channels,
        num_residual_blocks=args.num_residual_blocks
    ).to(device)
    
    degradation_layer = DegradationLayer().to(device)
    
    # Load model weights
    generator.load_state_dict(checkpoint["generator_state_dict"])
    degradation_layer.load_state_dict(checkpoint["degradation_state_dict"])
    
    # Set models to evaluation mode
    generator.eval()
    degradation_layer.eval()
    
    print(f"Generating {args.num_samples} samples...")
    
    # Generate samples
    with torch.no_grad():
        for i in tqdm(range(args.num_samples)):
            # Generate random noise
            z = torch.randn(1, args.latent_dim, 1, 1, device=device)
            
            # Generate clean image
            fake_img_clean = generator(z)
            
            # Apply degradation effects
            if args.use_degradation_layer:
                # Use the learned degradation layer
                fake_img_degraded = degradation_layer(fake_img_clean)
            else:
                # Use the manual degradation function with custom intensity
                fake_img_degraded = add_renaissance_degradations(
                    fake_img_clean, intensity=args.degradation_intensity
                )
                if isinstance(fake_img_degraded, Image.Image):
                    # Convert PIL image back to tensor
                    fake_img_degraded = torch.from_numpy(np.array(fake_img_degraded)).float() / 255.0
                    fake_img_degraded = fake_img_degraded.unsqueeze(0).to(device)
                    # Normalize to [-1, 1]
                    fake_img_degraded = fake_img_degraded * 2 - 1
            
            # Save both clean and degraded versions if requested
            if args.save_clean:
                save_samples([fake_img_clean[0]], args.output_dir, prefix=f"clean_{i+1}")
            
            save_samples([fake_img_degraded[0]], args.output_dir, prefix=f"degraded_{i+1}")
            
            # Also save side-by-side comparison if requested
            if args.save_comparison and args.save_clean:
                # Denormalize images from [-1, 1] to [0, 255]
                clean_img = (fake_img_clean[0] * 0.5 + 0.5).cpu().numpy().transpose(1, 2, 0) * 255
                degraded_img = (fake_img_degraded[0] * 0.5 + 0.5).cpu().numpy().transpose(1, 2, 0) * 255
                
                # Convert to uint8
                clean_img = clean_img.astype(np.uint8)
                degraded_img = degraded_img.astype(np.uint8)
                
                if clean_img.shape[2] == 1:
                    clean_img = clean_img.squeeze(2)
                if degraded_img.shape[2] == 1:
                    degraded_img = degraded_img.squeeze(2)
                
                # Create a side-by-side comparison
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(clean_img, cmap='gray')
                axes[0].set_title("Clean Text")
                axes[0].axis('off')
                
                axes[1].imshow(degraded_img, cmap='gray')
                axes[1].set_title("Degraded Text")
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"comparison_{i+1}.png"))
                plt.close()
    
    print(f"Generated samples saved to {args.output_dir}")


def render_text_image(text, font_path, width=512, height=512, font_size=24):
    """
    Render text to an image using specified font.
    
    Args:
        text (str): Text to render
        font_path (str): Path to TTF font file
        width (int): Image width
        height (int): Image height
        font_size (int): Font size
        
    Returns:
        PIL.Image: Rendered text image
    """
    # Create a white image
    image = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(image)
    
    # Always use default font for simplicity and better compatibility
    font = ImageFont.load_default()
    print("Using default font for rendering")
    
    # Split text into lines
    lines = text.split('\n')
    
    # Calculate text position
    y_offset = (height - len(lines) * font_size) // 2
    
    # Draw text lines
    for i, line in enumerate(lines):
        # Calculate approximate line width
        try:
            # PIL >= 8.0.0
            line_width = draw.textlength(line, font=font)
        except (AttributeError, TypeError):
            # Older PIL versions
            try:
                line_width = font.getsize(line)[0]
            except (AttributeError, TypeError):
                # Fallback to a safe estimate
                line_width = len(line) * (font_size // 2)
        
        x_offset = (width - line_width) // 2
        draw.text((x_offset, y_offset + i * font_size), line, font=font, fill=0)
    
    return image


def render_spanish_sample(args):
    """Render a sample of 17th century Spanish text."""
    
    # Sample Spanish text from the 17th century (Don Quixote excerpt)
    spanish_text = """En un lugar de la Mancha, de cuyo nombre no quiero acordarme,
no ha mucho tiempo que vivía un hidalgo de los de lanza
en astillero, adarga antigua, rocín flaco y galgo corredor.
Una olla de algo más vaca que carnero, salpicón las más
noches, duelos y quebrantos los sábados, lantejas los
viernes, algún palomino de añadidura los domingos,
consumían las tres partes de su hacienda."""
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Always use default font - no need to download or load TTF
    args.font_path = None
    
    # Render text images
    print("Rendering Spanish text samples...")
    for i in range(args.num_samples):
        # Render clean text image
        text_image = render_text_image(
            spanish_text, 
            args.font_path,
            width=args.img_size,
            height=args.img_size,
            font_size=args.font_size
        )
        
        # Save clean image
        if args.save_clean:
            text_image.save(os.path.join(args.output_dir, f"clean_text_{i+1}.png"))
        
        # Apply Renaissance degradation effects
        degraded_image = add_renaissance_degradations(text_image, intensity=args.degradation_intensity)
        degraded_image.save(os.path.join(args.output_dir, f"degraded_text_{i+1}.png"))
        
        # Create comparison if requested
        if args.save_comparison and args.save_clean:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(text_image, cmap='gray')
            axes[0].set_title("Clean Text")
            axes[0].axis('off')
            
            axes[1].imshow(degraded_image, cmap='gray')
            axes[1].set_title("Degraded Text")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"text_comparison_{i+1}.png"))
            plt.close()
    
    print(f"Text samples saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Renaissance Text Samples")
    
    # Generation settings
    parser.add_argument("--output_dir", type=str, default="generated_samples", 
                        help="Directory to save generated samples")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("--no_cuda", action="store_true", 
                        help="Disable CUDA")
    
    # Model settings
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="Dimension of latent space")
    parser.add_argument("--g_channels", type=int, default=64,
                        help="Number of generator channels")
    parser.add_argument("--num_residual_blocks", type=int, default=6,
                        help="Number of residual blocks in generator")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Size of generated images")
    
    # Degradation settings
    parser.add_argument("--use_degradation_layer", action="store_true",
                        help="Use the learned degradation layer")
    parser.add_argument("--degradation_intensity", type=float, default=0.7,
                        help="Intensity of manual degradation effects (0.0-1.0)")
    
    # Output settings
    parser.add_argument("--save_clean", action="store_true",
                        help="Save clean (non-degraded) images")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save side-by-side comparison of clean and degraded")
    
    # Text rendering settings
    parser.add_argument("--render_text", action="store_true",
                        help="Render Spanish text samples instead of using GAN")
    parser.add_argument("--font_path", type=str, default=None,
                        help="Path to TTF font file for text rendering")
    parser.add_argument("--font_size", type=int, default=24,
                        help="Font size for text rendering")
    
    args = parser.parse_args()
    
    if args.render_text:
        render_spanish_sample(args)
    else:
        if args.checkpoint is None:
            print("Error: Model checkpoint must be provided unless using --render_text")
            exit(1)
        generate_samples(args) 