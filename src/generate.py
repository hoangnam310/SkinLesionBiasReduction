"""
Script to generate synthetic skin lesion images using a trained cGAN.

This script is used after training to generate images for underrepresented
skin tones (Fitzpatrick 5 & 6) to balance the dataset for fair classification.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgan import Generator, GeneratorUpsample
from utils import (
    load_generator_only,
    generate_samples,
    save_generated_images,
    visualize_samples
)
from data_process import (
    NUM_SKIN_TONES,
    NUM_LESION_TYPES,
    FITZPATRICK_DECODING,
    LESION_TYPE_DECODING
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic skin lesion images using trained cGAN"
    )
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="Dimension of noise vector (must match training)")
    parser.add_argument("--embedding_dim", type=int, default=50,
                        help="Dimension of condition embeddings (must match training)")
    parser.add_argument("--ngf", type=int, default=64,
                        help="Base number of generator filters (must match training)")
    parser.add_argument("--gen_arch", type=str, default="auto",
                        choices=["auto", "upsample", "deconv"],
                        help="Generator architecture. 'auto' picks from the checkpoint's "
                             "state_dict keys ('up.*' -> upsample, 'deconv.*' -> deconv).")
    parser.add_argument("--use_attention", type=str, default="auto",
                        choices=["auto", "true", "false"],
                        help="Whether the checkpoint was trained with SelfAttention. "
                             "'auto' detects by looking for '.query.weight' keys.")
    
    # Generation arguments
    parser.add_argument("--output_dir", type=str, default="generated_images",
                        help="Directory to save generated images")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to generate per class")
    parser.add_argument("--target_skin_tones", type=int, nargs="+", default=[5, 6],
                        help="Skin tones to generate (Fitzpatrick scale 1-6)")
    parser.add_argument("--target_lesion_types", type=str, nargs="+", 
                        default=["benign", "malignant", "non-neoplastic"],
                        help="Lesion types to generate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation")
    
    # Output arguments
    parser.add_argument("--save_grid", action="store_true",
                        help="Save sample grid visualization")
    parser.add_argument("--create_csv", action="store_true",
                        help="Create CSV file with generated image metadata")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Hardware
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu/mps)")
    
    return parser.parse_args()


def get_device(device_str: str = None) -> torch.device:
    """Get the appropriate device for generation."""
    if device_str is not None:
        return torch.device(device_str)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    """Main generation function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for images
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Load checkpoint and pick the matching generator architecture.
    print(f"Loading generator from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    gen_state = checkpoint["generator_state_dict"]

    if args.gen_arch == "auto":
        has_up = any(k.startswith("up.") for k in gen_state)
        has_deconv = any(k.startswith("deconv.") for k in gen_state)
        if has_up and not has_deconv:
            gen_arch = "upsample"
        elif has_deconv and not has_up:
            gen_arch = "deconv"
        else:
            raise RuntimeError(
                f"Cannot auto-detect generator architecture from checkpoint keys "
                f"(has_up={has_up}, has_deconv={has_deconv}). "
                f"Pass --gen_arch upsample|deconv explicitly."
            )
        print(f"Auto-detected gen_arch={gen_arch!r}")
    else:
        gen_arch = args.gen_arch

    if args.use_attention == "auto":
        use_attention = any(".query.weight" in k for k in gen_state)
        print(f"Auto-detected use_attention={use_attention}")
    else:
        use_attention = args.use_attention == "true"

    if gen_arch == "upsample":
        generator = GeneratorUpsample(
            latent_dim=args.latent_dim,
            embedding_dim=args.embedding_dim,
            ngf=args.ngf,
            use_attention=use_attention,
        )
    else:
        if use_attention:
            raise ValueError("use_attention=True is not supported with gen_arch='deconv'.")
        generator = Generator(
            latent_dim=args.latent_dim,
            embedding_dim=args.embedding_dim,
            ngf=args.ngf,
        )

    generator = generator.to(device)
    generator.load_state_dict(gen_state)
    epoch = checkpoint.get("epoch", "unknown")
    print(f"Loaded checkpoint from epoch {epoch}")
    
    generator.eval()
    
    # Convert skin tones to indices (Fitzpatrick 1-6 -> 0-5)
    skin_tone_indices = [st - 1 for st in args.target_skin_tones]
    
    # Convert lesion types to indices
    lesion_type_to_idx = {"benign": 0, "malignant": 1, "non-neoplastic": 2}
    lesion_type_indices = [lesion_type_to_idx[lt] for lt in args.target_lesion_types]
    
    # Generate images
    metadata = []
    total_generated = 0
    
    print(f"\nGenerating {args.num_samples} samples for each class combination...")
    print(f"Target skin tones: {args.target_skin_tones}")
    print(f"Target lesion types: {args.target_lesion_types}")
    
    for skin_tone_idx in skin_tone_indices:
        skin_tone = skin_tone_idx + 1  # Convert back to Fitzpatrick scale
        
        for lesion_type_idx in lesion_type_indices:
            lesion_type = args.target_lesion_types[lesion_type_indices.index(lesion_type_idx)]
            
            print(f"\nGenerating: Fitzpatrick {skin_tone}, {lesion_type}")
            
            # Create class-specific subdirectory
            class_dir = images_dir / f"st{skin_tone}_{lesion_type}"
            class_dir.mkdir(exist_ok=True)
            
            # Generate in batches
            num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
            generated_count = 0
            
            pbar = tqdm(range(num_batches), desc=f"ST{skin_tone}-{lesion_type}")
            
            for batch_idx in pbar:
                # Calculate batch size for this iteration
                remaining = args.num_samples - generated_count
                current_batch_size = min(args.batch_size, remaining)
                
                if current_batch_size <= 0:
                    break
                
                # Generate samples
                with torch.inference_mode():
                    noise = torch.randn(current_batch_size, args.latent_dim, device=device)
                    skin_tones_tensor = torch.full(
                        (current_batch_size,), skin_tone_idx, 
                        dtype=torch.long, device=device
                    )
                    lesion_types_tensor = torch.full(
                        (current_batch_size,), lesion_type_idx,
                        dtype=torch.long, device=device
                    )
                    
                    images = generator(noise, skin_tones_tensor, lesion_types_tensor)
                
                # Save images
                for i, img in enumerate(images):
                    img_idx = generated_count + i
                    filename = f"generated_st{skin_tone}_{lesion_type}_{img_idx:06d}.png"
                    filepath = class_dir / filename
                    
                    # Denormalize and save
                    img_save = (img + 1) / 2
                    img_save = img_save.clamp(0, 1)
                    
                    from torchvision.utils import save_image
                    save_image(img_save, filepath)
                    
                    # Track metadata
                    metadata.append({
                        "filename": filename,
                        "filepath": str(filepath),
                        "fitzpatrick_scale": skin_tone,
                        "three_partition_label": lesion_type,
                        "is_synthetic": True
                    })
                
                generated_count += current_batch_size
                total_generated += current_batch_size
            
            # Save sample grid for this class
            if args.save_grid:
                grid_samples = generate_samples(
                    generator, device, args.latent_dim,
                    skin_tone=skin_tone_idx,
                    lesion_type=lesion_type_idx,
                    num_samples=16
                )
                grid_path = output_dir / f"grid_st{skin_tone}_{lesion_type}.png"
                visualize_samples(
                    grid_samples,
                    torch.full((16,), skin_tone_idx),
                    torch.full((16,), lesion_type_idx),
                    save_path=grid_path,
                    nrow=4
                )
    
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Total images generated: {total_generated}")
    print(f"Images saved to: {images_dir}")
    
    # Save metadata CSV
    if args.create_csv:
        csv_path = output_dir / "generated_metadata.csv"
        df = pd.DataFrame(metadata)
        df.to_csv(csv_path, index=False)
        print(f"Metadata saved to: {csv_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("Summary by class:")
    print(f"{'='*50}")
    
    for skin_tone in args.target_skin_tones:
        for lesion_type in args.target_lesion_types:
            class_count = sum(
                1 for m in metadata 
                if m["fitzpatrick_scale"] == skin_tone 
                and m["three_partition_label"] == lesion_type
            )
            print(f"  Fitzpatrick {skin_tone}, {lesion_type}: {class_count} images")


if __name__ == "__main__":
    main()
