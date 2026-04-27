"""
Skin Lesion Bias Reduction - cGAN Pipeline

This package provides tools for generating synthetic skin lesion images
using conditional GANs to address dataset bias in skin tone representation.
"""

from .data_process import (
    SkinLesionDataset,
    get_dataloader,
    get_augmented_transform,
    FITZPATRICK_ENCODING,
    LESION_TYPE_ENCODING,
    NUM_SKIN_TONES,
    NUM_LESION_TYPES
)

from .cgan import (
    Generator,
    Discriminator,
    cGAN,
    get_models
)

from .utils import (
    weights_init,
    save_checkpoint,
    load_checkpoint,
    load_generator_only,
    generate_samples,
    generate_balanced_samples,
    visualize_samples,
    save_generated_images,
    denormalize_image,
    normalize_image,
    compute_fid
)

__version__ = "0.1.0"
__all__ = [
    # Data
    "SkinLesionDataset",
    "get_dataloader",
    "get_augmented_transform",
    "FITZPATRICK_ENCODING",
    "LESION_TYPE_ENCODING",
    "NUM_SKIN_TONES",
    "NUM_LESION_TYPES",
    # Models
    "Generator",
    "Discriminator",
    "cGAN",
    "get_models",
    # Utils
    "weights_init",
    "save_checkpoint",
    "load_checkpoint",
    "load_generator_only",
    "generate_samples",
    "generate_balanced_samples",
    "visualize_samples",
    "save_generated_images",
    "denormalize_image",
    "normalize_image",
    "compute_fid",
]
