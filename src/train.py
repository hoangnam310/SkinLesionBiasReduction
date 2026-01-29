"""
Training script for the conditional GAN on Fitzpatrick17k dataset.

This script handles the complete training loop including:
- Data loading
- Model initialization
- Training with adversarial loss
- Checkpoint saving
- Sample generation for monitoring
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_process import get_dataloader, NUM_SKIN_TONES, NUM_LESION_TYPES
from cgan import get_models
from utils import (
    weights_init, 
    save_checkpoint, 
    load_checkpoint,
    generate_samples,
    visualize_samples,
    setup_logging,
    log_message
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train cGAN on Fitzpatrick17k dataset")
    
    # Data arguments
    parser.add_argument("--csv_path", type=str, 
                        default="dataset/fitzpatrick17k_cleaned.csv",
                        help="Path to CSV file with image metadata")
    parser.add_argument("--image_dir", type=str, 
                        default="dataset/images",
                        help="Directory containing images")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--beta1", type=float, default=0.5,
                        help="Beta1 parameter for Adam")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Beta2 parameter for Adam")
    
    # Model arguments
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="Dimension of noise vector")
    parser.add_argument("--embedding_dim", type=int, default=50,
                        help="Dimension of condition embeddings")
    parser.add_argument("--ngf", type=int, default=64,
                        help="Base number of generator filters")
    parser.add_argument("--ndf", type=int, default=64,
                        help="Base number of discriminator filters")
    
    # Training techniques
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor for real labels")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Standard deviation of noise added to discriminator inputs")
    parser.add_argument("--noise_decay", type=float, default=0.995,
                        help="Decay factor for discriminator noise")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory for outputs (checkpoints, samples, logs)")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--sample_interval", type=int, default=5,
                        help="Generate samples every N epochs")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of samples to generate for visualization")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    # Hardware
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu/mps). Auto-detect if not specified")
    
    return parser.parse_args()


def get_device(device_str: str = None) -> torch.device:
    """Get the appropriate device for training."""
    if device_str is not None:
        return torch.device(device_str)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    latent_dim: int,
    label_smoothing: float = 0.1,
    noise_std: float = 0.0
) -> dict:
    """
    Train for one epoch.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        dataloader: Training data loader
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        criterion: Loss function (BCE)
        device: Training device
        latent_dim: Dimension of noise vector
        label_smoothing: Smoothing factor for real labels
        noise_std: Standard deviation of noise for discriminator inputs
        
    Returns:
        Dictionary with average losses for the epoch
    """
    generator.train()
    discriminator.train()
    
    total_loss_d = 0.0
    total_loss_g = 0.0
    total_d_real = 0.0
    total_d_fake = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for real_images, skin_tones, lesion_types in pbar:
        batch_size = real_images.size(0)
        
        # Move to device
        real_images = real_images.to(device)
        skin_tones = skin_tones.to(device)
        lesion_types = lesion_types.to(device)
        
        # Labels with smoothing
        real_label = 1.0 - label_smoothing
        fake_label = 0.0
        
        # Add noise to real images (helps stabilize training)
        if noise_std > 0:
            real_images = real_images + torch.randn_like(real_images) * noise_std
        
        # =====================
        # Train Discriminator
        # =====================
        optimizer_d.zero_grad()
        
        # Real images
        output_real = discriminator(real_images, skin_tones, lesion_types)
        labels_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        loss_d_real = criterion(output_real.view(-1), labels_real)
        
        # Fake images
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(noise, skin_tones, lesion_types)
        
        if noise_std > 0:
            fake_images_noisy = fake_images.detach() + torch.randn_like(fake_images) * noise_std
        else:
            fake_images_noisy = fake_images.detach()
        
        output_fake = discriminator(fake_images_noisy, skin_tones, lesion_types)
        labels_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
        loss_d_fake = criterion(output_fake.view(-1), labels_fake)
        
        # Total discriminator loss
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()
        
        # =====================
        # Train Generator
        # =====================
        optimizer_g.zero_grad()
        
        # Generate fake images again (no noise for generator training)
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(noise, skin_tones, lesion_types)
        output = discriminator(fake_images, skin_tones, lesion_types)
        
        # Generator wants discriminator to think fakes are real
        labels_real = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
        loss_g = criterion(output.view(-1), labels_real)
        
        loss_g.backward()
        optimizer_g.step()
        
        # Track statistics
        total_loss_d += loss_d.item()
        total_loss_g += loss_g.item()
        total_d_real += output_real.mean().item()
        total_d_fake += output_fake.mean().item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "D_loss": f"{loss_d.item():.4f}",
            "G_loss": f"{loss_g.item():.4f}",
            "D(x)": f"{output_real.mean().item():.3f}",
            "D(G(z))": f"{output_fake.mean().item():.3f}"
        })
    
    return {
        "loss_d": total_loss_d / num_batches,
        "loss_g": total_loss_g / num_batches,
        "d_real": total_d_real / num_batches,
        "d_fake": total_d_fake / num_batches
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    log_dir = output_dir / "logs"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(log_dir)
    log_message(log_file, f"Starting training with args: {args}")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir / "tensorboard")
    
    # Get device
    device = get_device(args.device)
    log_message(log_file, f"Using device: {device}")
    
    # Load data
    log_message(log_file, "Loading dataset...")
    dataloader, dataset = get_dataloader(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create models
    log_message(log_file, "Creating models...")
    generator, discriminator = get_models(
        latent_dim=args.latent_dim,
        embedding_dim=args.embedding_dim,
        ngf=args.ngf,
        ndf=args.ndf,
        device=device
    )
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Create optimizers
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        log_message(log_file, f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(
            args.resume, generator, discriminator, 
            optimizer_g, optimizer_d, device
        )
        start_epoch = checkpoint["epoch"]
        log_message(log_file, f"Resumed from epoch {start_epoch}")
    
    # Fixed noise and conditions for visualization
    fixed_noise = torch.randn(args.num_samples, args.latent_dim, device=device)
    # Create varied conditions for visualization
    fixed_skin_tones = torch.tensor(
        [i % NUM_SKIN_TONES for i in range(args.num_samples)], 
        device=device
    )
    fixed_lesion_types = torch.tensor(
        [i % NUM_LESION_TYPES for i in range(args.num_samples)], 
        device=device
    )
    
    # Training loop
    log_message(log_file, f"Starting training for {args.epochs} epochs...")
    noise_std = args.noise_std
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        metrics = train_epoch(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            criterion=criterion,
            device=device,
            latent_dim=args.latent_dim,
            label_smoothing=args.label_smoothing,
            noise_std=noise_std
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Decay noise
        noise_std *= args.noise_decay
        
        # Log metrics
        log_msg = (
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"D_loss: {metrics['loss_d']:.4f}, G_loss: {metrics['loss_g']:.4f}, "
            f"D(x): {metrics['d_real']:.3f}, D(G(z)): {metrics['d_fake']:.3f}, "
            f"Time: {epoch_time:.2f}s"
        )
        log_message(log_file, log_msg)
        
        # Tensorboard logging
        writer.add_scalar("Loss/Discriminator", metrics["loss_d"], epoch)
        writer.add_scalar("Loss/Generator", metrics["loss_g"], epoch)
        writer.add_scalar("Score/D(x)", metrics["d_real"], epoch)
        writer.add_scalar("Score/D(G(z))", metrics["d_fake"], epoch)
        
        # Generate and save samples
        if (epoch + 1) % args.sample_interval == 0:
            generator.eval()
            with torch.inference_mode():
                fake_samples = generator(fixed_noise, fixed_skin_tones, fixed_lesion_types)
            
            # Save grid
            save_path = sample_dir / f"samples_epoch_{epoch+1:04d}.png"
            visualize_samples(
                fake_samples, 
                fixed_skin_tones, 
                fixed_lesion_types,
                save_path=save_path
            )
            
            # Add to tensorboard
            from torchvision.utils import make_grid
            grid = make_grid(fake_samples, normalize=True, value_range=(-1, 1))
            writer.add_image("Generated Samples", grid, epoch)
            
            generator.train()
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pt"
            save_checkpoint(
                generator, discriminator,
                optimizer_g, optimizer_d,
                epoch + 1, metrics,
                checkpoint_path
            )
            log_message(log_file, f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    save_checkpoint(
        generator, discriminator,
        optimizer_g, optimizer_d,
        args.epochs, metrics,
        final_path
    )
    log_message(log_file, f"Training complete. Final model saved to {final_path}")
    
    # Close tensorboard writer
    writer.close()
    
    # Generate final samples for all skin tones (focus on underrepresented)
    log_message(log_file, "Generating samples for all skin tones...")
    for skin_tone in range(NUM_SKIN_TONES):
        for lesion_type in range(NUM_LESION_TYPES):
            samples = generate_samples(
                generator, device, args.latent_dim,
                skin_tone=skin_tone,
                lesion_type=lesion_type,
                num_samples=8
            )
            save_path = sample_dir / f"final_st{skin_tone+1}_lt{lesion_type}.png"
            visualize_samples(samples, 
                            torch.full((8,), skin_tone, device=device),
                            torch.full((8,), lesion_type, device=device),
                            save_path=save_path,
                            nrow=4)
    
    log_message(log_file, "All done!")


if __name__ == "__main__":
    main()
