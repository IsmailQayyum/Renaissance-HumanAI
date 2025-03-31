import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.gan_models import Generator, Discriminator, DegradationLayer
from utils.data_utils import create_dataloaders, save_samples
from utils.eval_metrics import evaluate_samples

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_gan(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Load data
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Initialize models
    generator = Generator(
        latent_dim=args.latent_dim,
        channels=args.g_channels,
        num_residual_blocks=args.num_residual_blocks
    ).to(device)
    
    discriminator = Discriminator(
        channels=args.d_channels
    ).to(device)
    
    degradation_layer = DegradationLayer().to(device)
    
    # Initialize optimizers
    g_optimizer = optim.Adam(
        list(generator.parameters()) + list(degradation_layer.parameters()),
        lr=args.g_lr,
        betas=(args.beta1, 0.999)
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.d_lr,
        betas=(args.beta1, 0.999)
    )
    
    # Learning rate schedulers
    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=args.lr_decay_step, gamma=0.5)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=args.lr_decay_step, gamma=0.5)
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    
    # Fixed noise for sampling
    fixed_noise = torch.randn(16, args.latent_dim, 1, 1, device=device)
    
    # Training loop
    global_step = 0
    best_fid = float('inf')
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        generator.train()
        discriminator.train()
        degradation_layer.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, real_imgs in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # Labels for real and fake images
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            #########################
            # Train Discriminator
            #########################
            d_optimizer.zero_grad()
            
            # Real images
            d_real_output = discriminator(real_imgs)
            d_real_loss = adversarial_loss(d_real_output, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_imgs_clean = generator(z)
            fake_imgs = degradation_layer(fake_imgs_clean)
            
            d_fake_output = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(d_fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            #########################
            # Train Generator
            #########################
            g_optimizer.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_imgs_clean = generator(z)
            fake_imgs = degradation_layer(fake_imgs_clean)
            
            # Try to fool the discriminator
            g_output = discriminator(fake_imgs)
            g_loss = adversarial_loss(g_output, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                "D Loss": f"{d_loss.item():.4f}",
                "G Loss": f"{g_loss.item():.4f}",
                "Epoch": f"{epoch+1}/{args.num_epochs}",
                "Step": global_step
            })
            
            # Log to TensorBoard
            if global_step % args.log_interval == 0:
                writer.add_scalar("Loss/discriminator", d_loss.item(), global_step)
                writer.add_scalar("Loss/generator", g_loss.item(), global_step)
                writer.add_scalar("Loss/d_real", d_real_loss.item(), global_step)
                writer.add_scalar("Loss/d_fake", d_fake_loss.item(), global_step)
            
            global_step += 1
        
        # Update learning rates
        g_scheduler.step()
        d_scheduler.step()
        
        # Generate and save samples
        if (epoch + 1) % args.sample_interval == 0:
            generator.eval()
            degradation_layer.eval()
            
            with torch.no_grad():
                fake_samples_clean = generator(fixed_noise)
                fake_samples = degradation_layer(fake_samples_clean)
                
                # Save samples
                save_samples(fake_samples, os.path.join(args.output_dir, "samples"), 
                             prefix=f"epoch_{epoch+1}")
                
                # Add to TensorBoard
                grid_clean = torch.cat([img for img in fake_samples_clean[:4]], dim=2)
                grid_degraded = torch.cat([img for img in fake_samples[:4]], dim=2)
                writer.add_image("Samples/clean", grid_clean, epoch + 1)
                writer.add_image("Samples/degraded", grid_degraded, epoch + 1)
        
        # Evaluate model
        if (epoch + 1) % args.eval_interval == 0:
            generator.eval()
            degradation_layer.eval()
            
            # Get real and fake samples for evaluation
            real_samples = []
            fake_samples = []
            
            with torch.no_grad():
                for real_imgs in val_loader:
                    real_imgs = real_imgs.to(device)
                    real_samples.append(real_imgs)
                    
                    # Generate fake samples
                    z = torch.randn(real_imgs.size(0), args.latent_dim, 1, 1, device=device)
                    fake_imgs_clean = generator(z)
                    fake_imgs = degradation_layer(fake_imgs_clean)
                    fake_samples.append(fake_imgs)
                    
                    if len(real_samples) >= 10:  # Limit number of batches for evaluation
                        break
            
            real_samples = torch.cat(real_samples, dim=0)
            fake_samples = torch.cat(fake_samples, dim=0)
            
            # Evaluate
            metrics = evaluate_samples(real_samples, fake_samples, device)
            
            # Log metrics
            writer.add_scalar("Metrics/FID", metrics["fid"], epoch + 1)
            writer.add_scalar("Metrics/LPIPS", metrics["lpips"], epoch + 1)
            writer.add_scalar("Metrics/Historical_Overall", 
                               metrics["historical_artifacts"]["overall"], epoch + 1)
            writer.add_scalar("Metrics/Ink_Bleeding", 
                               metrics["historical_artifacts"]["ink_bleeding"], epoch + 1)
            writer.add_scalar("Metrics/Text_Irregularity", 
                               metrics["historical_artifacts"]["text_irregularity"], epoch + 1)
            writer.add_scalar("Metrics/Paper_Texture", 
                               metrics["historical_artifacts"]["paper_texture"], epoch + 1)
            
            print(f"Epoch {epoch+1} Evaluation:")
            print(f"  FID: {metrics['fid']:.4f}")
            print(f"  LPIPS: {metrics['lpips']:.4f}")
            print(f"  Historical Score: {metrics['historical_artifacts']['overall']:.4f}")
            
            # Save best model
            if metrics["fid"] < best_fid:
                best_fid = metrics["fid"]
                torch.save({
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "degradation_state_dict": degradation_layer.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                    "metrics": metrics,
                }, os.path.join(args.output_dir, "checkpoints", "best_model.pth"))
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "degradation_state_dict": degradation_layer.state_dict(),
                "g_optimizer_state_dict": g_optimizer.state_dict(),
                "d_optimizer_state_dict": d_optimizer.state_dict(),
            }, os.path.join(args.output_dir, "checkpoints", f"checkpoint_epoch_{epoch+1}.pth"))
    
    # Save final model
    torch.save({
        "epoch": args.num_epochs,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "degradation_state_dict": degradation_layer.state_dict(),
        "g_optimizer_state_dict": g_optimizer.state_dict(),
        "d_optimizer_state_dict": d_optimizer.state_dict(),
    }, os.path.join(args.output_dir, "checkpoints", "final_model.pth"))
    
    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renaissance Text GAN")
    
    # Training settings
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Model settings
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of latent space")
    parser.add_argument("--g_channels", type=int, default=64, help="Number of generator channels")
    parser.add_argument("--d_channels", type=int, default=64, help="Number of discriminator channels")
    parser.add_argument("--num_residual_blocks", type=int, default=6, help="Number of residual blocks in generator")
    parser.add_argument("--img_size", type=int, default=256, help="Size of training images")
    
    # Optimization settings
    parser.add_argument("--g_lr", type=float, default=0.0002, help="Generator learning rate")
    parser.add_argument("--d_lr", type=float, default=0.0002, help="Discriminator learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--lr_decay_step", type=int, default=20, help="LR decay step size")
    
    # Logging and saving settings
    parser.add_argument("--log_interval", type=int, default=100, help="How many steps between logging")
    parser.add_argument("--sample_interval", type=int, default=5, help="How many epochs between sampling")
    parser.add_argument("--eval_interval", type=int, default=5, help="How many epochs between evaluation")
    parser.add_argument("--save_interval", type=int, default=10, help="How many epochs between saving")
    
    args = parser.parse_args()
    train_gan(args) 