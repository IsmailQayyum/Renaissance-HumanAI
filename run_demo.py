#!/usr/bin/env python
"""
Quick script to run the Renaissance Text GAN demo with sensible defaults.
This will:
1. Create a small dataset of Renaissance text
2. Generate samples using direct text rendering with degradation effects

To run with CUDA (if available):
    python run_demo.py

To specify output directory:
    python run_demo.py --output_dir my_results
"""

import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run Renaissance Text GAN Demo")
    parser.add_argument("--output_dir", type=str, default="demo_results",
                        help="Directory to save results")
    parser.add_argument("--num_samples", type=str, default="5",
                        help="Number of samples to generate")
    parser.add_argument("--render_only", action="store_true",
                        help="Skip dataset creation and only render samples")
    
    args = parser.parse_args()
    
    print("=== Renaissance Text GAN Demo ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create fonts directory
    fonts_dir = os.path.join("fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    
    # Step 1: Create a small dataset of Renaissance text
    if not args.render_only:
        print("\n--- Step 1: Creating small dataset ---")
        dataset_cmd = [
            sys.executable,
            os.path.join("data", "create_dataset.py"), 
            "--output_dir", os.path.join(args.output_dir, "dataset"),
            "--num_samples", "100",
            "--img_size", "256",
            "--degradation_intensity", "0.7"
        ]
        subprocess.run(dataset_cmd)
    
    # Step 2: Generate samples
    print("\n--- Step 2: Generating Renaissance text samples ---")
    
    # Use the default font - no need to download specific font
    render_cmd = [
        sys.executable,
        "generate.py",
        "--output_dir", os.path.join(args.output_dir, "generated"),
        "--num_samples", args.num_samples,
        "--render_text",
        "--degradation_intensity", "0.7",
        "--img_size", "256",
        "--save_clean",
        "--save_comparison"
    ]
    subprocess.run(render_cmd)
    
    print(f"\nDemo completed! Results saved to {args.output_dir}")
    print("- Dataset:", os.path.join(args.output_dir, "dataset"))
    print("- Generated samples:", os.path.join(args.output_dir, "generated"))
    
    print("\nTo see the results, open the comparison images in the generated directory.")
    print("To train a GAN model on this dataset, use:")
    print(f"  python train.py --data_dir {os.path.join(args.output_dir, 'dataset/degraded')} --output_dir {os.path.join(args.output_dir, 'model')}")

if __name__ == "__main__":
    main() 