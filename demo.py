import os
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import subprocess

from models.combined_model import RenaissanceTextGenerator
from utils.data_utils import add_renaissance_degradations, save_samples
from utils.eval_metrics import calculate_historical_artifact_score

def run_demo(args):
    """Run a complete demo of the Renaissance text generation pipeline."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Renaissance Text Generation Demo ===")
    
    # Step 1: Create a small dataset
    if not args.skip_dataset_creation:
        print("\n--- Step 1: Creating small dataset ---")
        dataset_cmd = [
            "python", "data/create_dataset.py",
            "--output_dir", os.path.join(args.output_dir, "dataset"),
            "--num_samples", str(args.num_dataset_samples),
            "--img_size", str(args.img_size),
            "--degradation_intensity", str(args.degradation_intensity)
        ]
        subprocess.run(dataset_cmd)
    
    # Step 2: Check if model already trained or pre-downloaded
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"\n--- Using existing model checkpoint: {args.checkpoint} ---")
        model_path = args.checkpoint
    else:
        print("\n--- No checkpoint provided, will render using degradation function ---")
        model_path = None
    
    # Step 3: Generate samples
    print("\n--- Generating Renaissance text samples ---")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    if model_path is not None:
        # Generate using trained model
        model = RenaissanceTextGenerator(
            latent_dim=args.latent_dim,
            channels=args.g_channels,
            num_residual_blocks=args.num_residual_blocks,
            pretrained=model_path
        ).to(device)
        
        model.eval()
        
        if args.bleed_scale is not None or args.smudge_scale is not None or \
           args.fade_scale is not None or args.noise_scale is not None:
            print("Adjusting degradation parameters:")
            model.adjust_degradation(
                bleed_scale=args.bleed_scale,
                smudge_scale=args.smudge_scale,
                fade_scale=args.fade_scale,
                noise_scale=args.noise_scale
            )
        
        # Generate samples
        print(f"Generating {args.num_samples} samples...")
        
        with torch.no_grad():
            clean_samples, degraded_samples = model.generate_samples(
                num_samples=args.num_samples,
                device=device
            )
        
        # Save samples
        os.makedirs(os.path.join(args.output_dir, "generated"), exist_ok=True)
        
        for i in range(args.num_samples):
            save_samples([clean_samples[i]], os.path.join(args.output_dir, "generated"), 
                          prefix=f"clean_{i+1}")
            save_samples([degraded_samples[i]], os.path.join(args.output_dir, "generated"), 
                          prefix=f"degraded_{i+1}")
            
            # Create side-by-side comparison
            clean_img = (clean_samples[i] * 0.5 + 0.5).cpu().numpy().transpose(1, 2, 0) * 255
            degraded_img = (degraded_samples[i] * 0.5 + 0.5).cpu().numpy().transpose(1, 2, 0) * 255
            
            if clean_img.shape[2] == 1:
                clean_img = clean_img.squeeze(2)
            if degraded_img.shape[2] == 1:
                degraded_img = degraded_img.squeeze(2)
            
            clean_img = clean_img.astype(np.uint8)
            degraded_img = degraded_img.astype(np.uint8)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(clean_img, cmap='gray')
            axes[0].set_title("Clean Generated Text")
            axes[0].axis('off')
            
            axes[1].imshow(degraded_img, cmap='gray')
            axes[1].set_title("Degraded Renaissance Text")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "generated", f"comparison_{i+1}.png"))
            plt.close()
            
        # Step 4: Evaluate the generated samples
        print("\n--- Step 4: Evaluating samples ---")
        
        artifact_scores = calculate_historical_artifact_score(degraded_samples, device)
        
        print("\nHistorical Artifact Scores:")
        print(f"  Ink Bleeding: {artifact_scores['ink_bleeding']:.4f}")
        print(f"  Text Irregularity: {artifact_scores['text_irregularity']:.4f}")
        print(f"  Paper Texture: {artifact_scores['paper_texture']:.4f}")
        print(f"  Overall Score: {artifact_scores['overall']:.4f}")
    
    else:
        # If no model checkpoint is available, render Spanish text directly
        # with degradation effects
        print("No model checkpoint available. Generating text with manual degradation...")
        render_cmd = [
            "python", "generate.py",
            "--output_dir", os.path.join(args.output_dir, "generated"),
            "--num_samples", str(args.num_samples),
            "--render_text",
            "--degradation_intensity", str(args.degradation_intensity),
            "--img_size", str(args.img_size),
            "--save_clean",
            "--save_comparison"
        ]
        subprocess.run(render_cmd)
    
    print(f"\nDemo completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renaissance Text Generation Demo")
    
    # General settings
    parser.add_argument("--output_dir", type=str, default="demo_results",
                        help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Size of generated images")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA")
    
    # Dataset creation settings
    parser.add_argument("--skip_dataset_creation", action="store_true",
                        help="Skip dataset creation step")
    parser.add_argument("--num_dataset_samples", type=int, default=100,
                        help="Number of samples to create for the dataset")
    parser.add_argument("--degradation_intensity", type=float, default=0.7,
                        help="Intensity of degradation effects (0.0-1.0)")
    
    # Model settings
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="Dimension of latent space")
    parser.add_argument("--g_channels", type=int, default=64,
                        help="Number of generator channels")
    parser.add_argument("--num_residual_blocks", type=int, default=6,
                        help="Number of residual blocks in generator")
    
    # Degradation settings
    parser.add_argument("--bleed_scale", type=float, default=None,
                        help="Scale for ink bleeding effect")
    parser.add_argument("--smudge_scale", type=float, default=None,
                        help="Scale for smudging effect")
    parser.add_argument("--fade_scale", type=float, default=None,
                        help="Scale for fading effect")
    parser.add_argument("--noise_scale", type=float, default=None,
                        help="Scale for paper texture noise")
    
    args = parser.parse_args()
    run_demo(args) 