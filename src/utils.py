"""
Utility functions for cGAN training and inference.

This module provides helper functions for:
- Weight initialization
- Checkpoint save/load
- Sample generation
- Visualization
- Logging
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image


def weights_init(m: nn.Module) -> None:
    """
    Initialize network weights following DCGAN paper.
    
    Conv and ConvTranspose layers: Normal distribution with mean=0, std=0.02
    BatchNorm layers: Normal distribution with mean=1, std=0.02
    
    Args:
        m: Module to initialize
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def save_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Union[str, Path],
    **kwargs
) -> None:
    """
    Save training checkpoint.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        epoch: Current epoch number
        metrics: Dictionary of training metrics
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: Optional[torch.optim.Optimizer] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint file
        generator: Generator model to load weights into
        discriminator: Discriminator model to load weights into
        optimizer_g: Optional generator optimizer to load state into
        optimizer_d: Optional discriminator optimizer to load state into
        device: Device to map checkpoint to
        
    Returns:
        Dictionary containing checkpoint data
    """
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    
    if optimizer_g is not None and "optimizer_g_state_dict" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    
    if optimizer_d is not None and "optimizer_d_state_dict" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    
    return checkpoint


def load_generator_only(
    path: Union[str, Path],
    generator: nn.Module,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Load only the generator from a checkpoint.
    
    Useful for inference when only generation is needed.
    
    Args:
        path: Path to checkpoint file
        generator: Generator model to load weights into
        device: Device to map checkpoint to
        
    Returns:
        Dictionary containing checkpoint data
    """
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    
    return checkpoint


def generate_samples(
    generator: nn.Module,
    device: torch.device,
    latent_dim: int,
    skin_tone: int,
    lesion_type: int,
    num_samples: int = 16,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate synthetic images for specific conditions.
    
    Args:
        generator: Trained generator model
        device: Device to generate on
        latent_dim: Dimension of noise vector
        skin_tone: Skin tone class (0-5, corresponding to Fitzpatrick 1-6)
        lesion_type: Lesion type class (0=benign, 1=malignant, 2=non-neoplastic)
        num_samples: Number of samples to generate
        seed: Optional random seed for reproducibility
        
    Returns:
        Generated images tensor of shape (num_samples, 3, 64, 64)
    """
    generator.eval()
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create noise and condition tensors
    noise = torch.randn(num_samples, latent_dim, device=device)
    skin_tones = torch.full((num_samples,), skin_tone, dtype=torch.long, device=device)
    lesion_types = torch.full((num_samples,), lesion_type, dtype=torch.long, device=device)
    
    with torch.inference_mode()():
        samples = generator(noise, skin_tones, lesion_types)
    
    return samples


def generate_balanced_samples(
    generator: nn.Module,
    device: torch.device,
    latent_dim: int,
    samples_per_class: int = 100,
    target_skin_tones: Optional[list] = None,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate balanced samples across specified skin tones and all lesion types.
    
    Useful for augmenting underrepresented classes.
    
    Args:
        generator: Trained generator model
        device: Device to generate on
        latent_dim: Dimension of noise vector
        samples_per_class: Number of samples per (skin_tone, lesion_type) pair
        target_skin_tones: List of skin tone classes to generate (default: [4, 5] for FST 5 & 6)
        seed: Optional random seed
        
    Returns:
        Tuple of (images, skin_tone_labels, lesion_type_labels)
    """
    if target_skin_tones is None:
        # Default to underrepresented skin tones (Fitzpatrick 5 & 6)
        target_skin_tones = [4, 5]
    
    generator.eval()
    
    if seed is not None:
        torch.manual_seed(seed)
    
    all_images = []
    all_skin_tones = []
    all_lesion_types = []
    
    num_lesion_types = 3
    
    for skin_tone in target_skin_tones:
        for lesion_type in range(num_lesion_types):
            samples = generate_samples(
                generator, device, latent_dim,
                skin_tone=skin_tone,
                lesion_type=lesion_type,
                num_samples=samples_per_class
            )
            all_images.append(samples)
            all_skin_tones.extend([skin_tone] * samples_per_class)
            all_lesion_types.extend([lesion_type] * samples_per_class)
    
    images = torch.cat(all_images, dim=0)
    skin_tones = torch.tensor(all_skin_tones, device=device)
    lesion_types = torch.tensor(all_lesion_types, device=device)
    
    return images, skin_tones, lesion_types


def visualize_samples(
    images: torch.Tensor,
    skin_tones: torch.Tensor,
    lesion_types: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    nrow: int = 4,
    figsize: Tuple[int, int] = (12, 12),
    show: bool = False
) -> None:
    """
    Create and optionally save a grid visualization of generated samples.
    
    Args:
        images: Image tensor of shape (N, 3, H, W) in range [-1, 1]
        skin_tones: Skin tone labels
        lesion_types: Lesion type labels
        save_path: Optional path to save the figure
        nrow: Number of images per row
        figsize: Figure size
        show: Whether to display the figure
    """
    # Denormalize images from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    
    # Create grid
    grid = make_grid(images.cpu(), nrow=nrow, padding=2, normalize=False)
    
    # Convert to numpy
    grid_np = grid.permute(1, 2, 0).numpy()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(grid_np)
    ax.axis("off")
    
    # Add title with condition information
    skin_tones_np = skin_tones.cpu().numpy() if torch.is_tensor(skin_tones) else skin_tones
    lesion_types_np = lesion_types.cpu().numpy() if torch.is_tensor(lesion_types) else lesion_types
    
    lesion_names = ["Benign", "Malignant", "Non-neoplastic"]
    title_lines = []
    for i in range(min(len(skin_tones_np), nrow)):
        st = skin_tones_np[i] + 1  # Convert back to Fitzpatrick scale
        lt = lesion_names[lesion_types_np[i]]
        title_lines.append(f"ST{st}-{lt}")
    
    ax.set_title(" | ".join(title_lines), fontsize=10)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_generated_images(
    images: torch.Tensor,
    output_dir: Union[str, Path],
    prefix: str = "generated",
    start_idx: int = 0
) -> list:
    """
    Save individual generated images to disk.
    
    Args:
        images: Image tensor of shape (N, 3, H, W) in range [-1, 1]
        output_dir: Directory to save images
        prefix: Filename prefix
        start_idx: Starting index for filenames
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    
    saved_paths = []
    for i, img in enumerate(images):
        path = output_dir / f"{prefix}_{start_idx + i:06d}.png"
        save_image(img, path)
        saved_paths.append(path)
    
    return saved_paths


def setup_logging(log_dir: Union[str, Path]) -> Path:
    """
    Setup logging to file.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Path to log file
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "training.log"
    
    return log_file


def log_message(log_file: Path, message: str) -> None:
    """
    Log a message to both console and file.
    
    Args:
        log_file: Path to log file
        message: Message to log
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    
    print(formatted)
    
    with open(log_file, "a") as f:
        f.write(formatted + "\n")


def compute_class_weights(distribution: Dict[str, Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute class weights for balanced sampling.
    
    Args:
        distribution: Class distribution from dataset.get_class_distribution()
        
    Returns:
        Tuple of (skin_tone_weights, lesion_type_weights)
    """
    skin_dist = distribution["skin_tone"]
    lesion_dist = distribution["lesion_type"]
    
    # Compute inverse frequency weights for skin tones
    total_skin = sum(skin_dist.values())
    skin_weights = []
    for i in range(1, 7):  # Fitzpatrick 1-6
        count = skin_dist.get(i, 1)
        skin_weights.append(total_skin / (6 * count))
    
    # Compute inverse frequency weights for lesion types
    total_lesion = sum(lesion_dist.values())
    lesion_weights = []
    for lesion in ["benign", "malignant", "non-neoplastic"]:
        count = lesion_dist.get(lesion, 1)
        lesion_weights.append(total_lesion / (3 * count))
    
    return torch.tensor(skin_weights), torch.tensor(lesion_weights)


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image from [-1, 1] to [0, 1].
    
    Args:
        image: Image tensor in range [-1, 1]
        
    Returns:
        Image tensor in range [0, 1]
    """
    return (image + 1) / 2


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize image from [0, 1] to [-1, 1].
    
    Args:
        image: Image tensor in range [0, 1]
        
    Returns:
        Image tensor in range [-1, 1]
    """
    return image * 2 - 1


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test weight initialization
    conv = nn.Conv2d(3, 64, 4, 2, 1)
    weights_init(conv)
    print(f"Conv weight mean: {conv.weight.mean():.4f}, std: {conv.weight.std():.4f}")
    
    bn = nn.BatchNorm2d(64)
    weights_init(bn)
    print(f"BatchNorm weight mean: {bn.weight.mean():.4f}, std: {bn.weight.std():.4f}")
    
    # Test denormalize
    test_tensor = torch.randn(4, 3, 64, 64) * 0.5  # Range roughly [-1.5, 1.5]
    denorm = denormalize_image(test_tensor.clamp(-1, 1))
    print(f"Denormalized range: [{denorm.min():.2f}, {denorm.max():.2f}]")
    
    print("\nAll utility tests passed!")
