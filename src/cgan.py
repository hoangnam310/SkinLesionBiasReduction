import math

import torch
import torch.nn as nn
from typing import Tuple

from data_process import NUM_SKIN_TONES, NUM_LESION_TYPES


def _validate_image_size(image_size: int) -> int:
    """Image size must be a power of 2 in {32, 64, 128, 256, 512}.

    The networks build a 4x4 feature map and up/down-sample by stride 2 each
    stage, so the resolution must satisfy `image_size = 4 * 2^k` for k>=3.
    """
    if image_size < 32 or (image_size & (image_size - 1)) != 0:
        raise ValueError(
            f"image_size must be a power of 2 >= 32 (got {image_size})."
        )
    return image_size


def _generator_stage_channels(image_size: int, ngf: int) -> list[Tuple[int, int]]:
    """Return [(in_ch, out_ch), ...] for each upsample stage.

    Stage 0 takes the (ngf*16, 4, 4) feature map. Each stage doubles spatial
    and halves channels (floored at ngf*1). The final stage outputs 3 (RGB).
    For image_size=64 this reproduces the original ngf*16 -> 8 -> 4 -> 2 -> 3
    ladder exactly so existing checkpoints stay loadable.
    """
    num_stages = int(math.log2(image_size // 4))
    stages = []
    in_ch = ngf * 16
    for stage_idx in range(num_stages):
        if stage_idx == num_stages - 1:
            out_ch = 3
        else:
            out_mult = max(16 >> (stage_idx + 1), 1)
            out_ch = ngf * out_mult
        stages.append((in_ch, out_ch))
        in_ch = out_ch
    return stages


def _critic_stage_channels(ndf: int, num_stages: int) -> list[int]:
    """Per-stage output channel count for Critic/Discriminator stride-2 convs.

    Doubles channels each stage capped at ndf*8 (matches the original 64x64
    ladder ndf -> 2 -> 4 -> 8 and extends safely for deeper nets).
    """
    return [ndf * min(1 << i, 8) for i in range(num_stages)]


class SelfAttention(nn.Module):
    """SAGAN-style self-attention block.

    Computes softmax attention over all spatial positions in a feature map:
        out = gamma * (Value @ softmax(Query^T Key)) + x
    `gamma` is initialized to 0 so the layer starts as the identity and only
    becomes useful as the gradient signal teaches it to be.

    Args:
        in_channels: number of channels in the input feature map.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        reduced = max(in_channels // 8, 1)
        self.query = nn.Conv2d(in_channels, reduced, kernel_size=1)
        self.key = nn.Conv2d(in_channels, reduced, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        q = self.query(x).reshape(B, -1, N).permute(0, 2, 1)  # (B, N, C/8)
        k = self.key(x).reshape(B, -1, N)                      # (B, C/8, N)
        v = self.value(x).reshape(B, -1, N)                    # (B, C, N)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)          # (B, N, N)
        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(B, C, H, W)
        return self.gamma * out + x


class Generator(nn.Module):
    """    
    Takes a noise vector concatenated with condition embeddings and
    generates a 64x64 RGB image.
    
    Args:
        latent_dim: Dimension of the noise vector (default: 100)
        embedding_dim: Dimension of condition embeddings representing the classes (default: 50)
        num_skin_tones: Number of skin tone classes (default: 6)
        num_lesion_types: Number of lesion type classes (default: 3)
        ngf: Base number of generator filters (default: 64)
    """
    
    def __init__(
        self,
        latent_dim = 100,
        embedding_dim = 50,
        num_skin_tones  = NUM_SKIN_TONES,
        num_lesion_types  = NUM_LESION_TYPES,
        ngf = 64 # number generator filter
    ):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        
        # Condition embeddings
        self.skin_tone_embedding = nn.Embedding(num_skin_tones, embedding_dim)
        self.lesion_type_embedding = nn.Embedding(num_lesion_types, embedding_dim)
        
        # Total input dimension: noise + skin_tone_embed + lesion_type_embed -> cGAN
        self.input_dim = latent_dim + 2 * embedding_dim
        
        # Project and reshape: input_dim -> 1024 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, ngf * 16 * 4 * 4),
            nn.BatchNorm1d(ngf * 16 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Transposed convolution layers
        # Input: 1024 x 4 x 4
        self.deconv = nn.Sequential(
            # Layer 1: 1024 x 4 x 4 -> 512 x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # Layer 2: 512 x 8 x 8 -> 256 x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # Layer 3: 256 x 16 x 16 -> 128 x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # Layer 4: 128 x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(ngf * 2, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(
        self, 
        noise: torch.Tensor, 
        skin_tone: torch.Tensor, 
        lesion_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate images conditioned on skin tone and lesion type.
        
        Args:
            noise: Random noise tensor of shape (batch_size, latent_dim)
            skin_tone: Skin tone labels of shape (batch_size,)
            lesion_type: Lesion type labels of shape (batch_size,)
            
        Returns:
            Generated images of shape (batch_size, 3, 64, 64)
        """
        # Get condition embeddings
        skin_embed = self.skin_tone_embedding(skin_tone)  # (batch, embedding_dim)
        lesion_embed = self.lesion_type_embedding(lesion_type)  # (batch, embedding_dim)
        
        # Concatenate noise and conditions
        x = torch.cat([noise, skin_embed, lesion_embed], dim=1)  # (batch, input_dim)
        
        # Project and reshape
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)  # (batch, 1024, 4, 4)
        
        # Generate image
        x = self.deconv(x)
        
        return x


class GeneratorUpsample(nn.Module):
    """
    Same conditioning as Generator (concat noise + skin/lesion embeddings, FC to
    1024x4x4) but each upsampling stage is `Upsample(nearest) + Conv2d(3x3)`
    instead of stride-2 ConvTranspose2d. This removes the uneven-overlap
    checkerboard artifacts that show up in DCGAN-style outputs.

    Args:
        latent_dim: Dimension of the noise vector (default: 100)
        embedding_dim: Dimension of condition embeddings (default: 50)
        num_skin_tones: Number of skin tone classes (default: 6)
        num_lesion_types: Number of lesion type classes (default: 3)
        ngf: Base number of generator filters (default: 64)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        embedding_dim: int = 50,
        num_skin_tones: int = NUM_SKIN_TONES,
        num_lesion_types: int = NUM_LESION_TYPES,
        ngf: int = 64,
        use_attention: bool = False,
        image_size: int = 64,
    ):
        super(GeneratorUpsample, self).__init__()

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        self.image_size = _validate_image_size(image_size)

        self.skin_tone_embedding = nn.Embedding(num_skin_tones, embedding_dim)
        self.lesion_type_embedding = nn.Embedding(num_lesion_types, embedding_dim)

        self.input_dim = latent_dim + 2 * embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, ngf * 16 * 4 * 4),
            nn.BatchNorm1d(ngf * 16 * 4 * 4),
            nn.ReLU(True),
        )

        def up_block(in_ch, out_ch, final=False):
            layers = [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=final),
            ]
            if not final:
                layers += [nn.BatchNorm2d(out_ch), nn.ReLU(True)]
            return layers

        stage_channels = _generator_stage_channels(self.image_size, ngf)
        layers: list[nn.Module] = []
        spatial = 4
        for stage_idx, (in_ch, out_ch) in enumerate(stage_channels):
            if use_attention and spatial == 32:
                layers.append(SelfAttention(in_ch))  # attention at 32x32
            final = stage_idx == len(stage_channels) - 1
            layers += up_block(in_ch, out_ch, final=final)
            spatial *= 2
        layers.append(nn.Tanh())
        self.up = nn.Sequential(*layers)

    def forward(
        self,
        noise: torch.Tensor,
        skin_tone: torch.Tensor,
        lesion_type: torch.Tensor,
    ) -> torch.Tensor:
        skin_embed = self.skin_tone_embedding(skin_tone)
        lesion_embed = self.lesion_type_embedding(lesion_type)
        x = torch.cat([noise, skin_embed, lesion_embed], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.up(x)
        return x


class Discriminator(nn.Module):
    """
    DCGAN-style Discriminator conditioned on skin tone and lesion type.
    
    Takes an image and condition embeddings and outputs a probability
    of the image being real.
    
    Args:
        embedding_dim: Dimension of condition embeddings (default: 50)
        num_skin_tones: Number of skin tone classes (default: 6)
        num_lesion_types: Number of lesion type classes (default: 3)
        ndf: Base number of discriminator filters (default: 64)
    """
    
    def __init__(
        self,
        embedding_dim = 50,
        num_skin_tones = NUM_SKIN_TONES,
        num_lesion_types = NUM_LESION_TYPES,
        ndf = 64
    ):
        super(Discriminator, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Condition embeddings
        self.skin_tone_embedding = nn.Embedding(num_skin_tones, embedding_dim)
        self.lesion_type_embedding = nn.Embedding(num_lesion_types, embedding_dim)
        
        # Project condition embeddings to spatial representation
        # Each embedding is projected to a single channel of 64x64
        self.skin_embed_proj = nn.Linear(embedding_dim, 64 * 64)
        self.lesion_embed_proj = nn.Linear(embedding_dim, 64 * 64)
        
        # Input channels: 3 (image) + 2 (condition channels)
        self.conv = nn.Sequential(
            # Layer 1: 5 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(3 + 2, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 128 x 16 x 16 -> 256 x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 256 x 8 x 8 -> 512 x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: 512 x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        image: torch.Tensor, 
        skin_tone: torch.Tensor, 
        lesion_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Discriminate images conditioned on skin tone and lesion type.
        
        Args:
            image: Input images of shape (batch_size, 3, 64, 64)
            skin_tone: Skin tone labels of shape (batch_size,)
            lesion_type: Lesion type labels of shape (batch_size,)
            
        Returns:
            Probability of image being real, shape (batch_size, 1)
        """
        batch_size = image.size(0)
        
        # Get condition embeddings
        skin_embed = self.skin_tone_embedding(skin_tone)  # (batch, embedding_dim)
        lesion_embed = self.lesion_type_embedding(lesion_type)  # (batch, embedding_dim)
        
        # Project embeddings to spatial representation
        skin_channel = self.skin_embed_proj(skin_embed)  # (batch, 64*64)
        skin_channel = skin_channel.view(batch_size, 1, 64, 64)
        
        lesion_channel = self.lesion_embed_proj(lesion_embed)  # (batch, 64*64)
        lesion_channel = lesion_channel.view(batch_size, 1, 64, 64)
        
        # Concatenate image with condition channels
        x = torch.cat([image, skin_channel, lesion_channel], dim=1)  # (batch, 5, 64, 64)
        
        # Discriminate
        x = self.conv(x)
        x = x.view(batch_size, -1)
        
        return x


class Critic(nn.Module):
    """
    WGAN-GP Critic conditioned on skin tone and lesion type.
    
    For WGAN-GP, we remove BatchNorm and Sigmoid. The critic outputs unbounded values
    representing how "real" an image looks. Uses LayerNorm instead of BatchNorm
    for stability with gradient penalty.
    
    Args:
        embedding_dim: Dimension of condition embeddings (default: 50)
        num_skin_tones: Number of skin tone classes (default: 6)
        num_lesion_types: Number of lesion type classes (default: 3)
        ndf: Base number of critic filters (default: 64)
    """
    
    def __init__(
        self,
        embedding_dim: int = 50,
        num_skin_tones: int = NUM_SKIN_TONES,
        num_lesion_types: int = NUM_LESION_TYPES,
        ndf: int = 64,
        norm: str = "instance",
        use_attention: bool = False,
        image_size: int = 64,
    ):
        super(Critic, self).__init__()

        self.embedding_dim = embedding_dim
        self.norm = norm
        self.use_attention = use_attention
        self.image_size = _validate_image_size(image_size)

        # Condition embeddings
        self.skin_tone_embedding = nn.Embedding(num_skin_tones, embedding_dim)
        self.lesion_type_embedding = nn.Embedding(num_lesion_types, embedding_dim)

        # Project condition embeddings to one spatial channel matching the image.
        self.skin_embed_proj = nn.Linear(embedding_dim, self.image_size * self.image_size)
        self.lesion_embed_proj = nn.Linear(embedding_dim, self.image_size * self.image_size)

        # No BatchNorm for WGAN-GP (interferes with gradient penalty).
        # Default is InstanceNorm; LayerNorm is the original WGAN-GP paper's
        # recommendation and is opt-in via norm="layer".
        def norm_layer(channels: int, spatial: int) -> nn.Module:
            if norm == "instance":
                return nn.InstanceNorm2d(channels, affine=True)
            if norm == "layer":
                return nn.LayerNorm([channels, spatial, spatial])
            raise ValueError(f"Unknown norm={norm!r}; expected 'instance' or 'layer'.")

        num_stages = int(math.log2(self.image_size // 4))
        out_channels_per_stage = _critic_stage_channels(ndf, num_stages)

        layers: list[nn.Module] = []
        in_ch = 3 + 2  # RGB + 2 condition channels
        spatial = self.image_size
        for stage_idx, out_ch in enumerate(out_channels_per_stage):
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=True))
            spatial //= 2
            if stage_idx > 0:
                # Skip norm on the first conv (matches the original 64x64 design).
                layers.append(norm_layer(out_ch, spatial))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if use_attention and spatial == 32:
                layers.append(SelfAttention(out_ch))  # attention at 32x32
            in_ch = out_ch

        # Final 4x4 -> 1x1 conv. No Sigmoid — Wasserstein output is unbounded.
        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 0, bias=True))
        self.conv = nn.Sequential(*layers)
    
    def forward(
        self, 
        image: torch.Tensor, 
        skin_tone: torch.Tensor, 
        lesion_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Critique images conditioned on skin tone and lesion type.
        
        Args:
            image: Input images of shape (batch_size, 3, 64, 64)
            skin_tone: Skin tone labels of shape (batch_size,)
            lesion_type: Lesion type labels of shape (batch_size,)
            
        Returns:
            Critic scores (unbounded), shape (batch_size, 1)
        """
        batch_size = image.size(0)
        
        # Get condition embeddings
        skin_embed = self.skin_tone_embedding(skin_tone)
        lesion_embed = self.lesion_type_embedding(lesion_type)
        
        # Project embeddings to spatial representation
        skin_channel = self.skin_embed_proj(skin_embed).view(
            batch_size, 1, self.image_size, self.image_size
        )
        lesion_channel = self.lesion_embed_proj(lesion_embed).view(
            batch_size, 1, self.image_size, self.image_size
        )
        
        # Concatenate image with condition channels
        x = torch.cat([image, skin_channel, lesion_channel], dim=1)
        
        # Critique
        x = self.conv(x)
        x = x.view(batch_size, -1)
        
        return x


class cGAN(nn.Module):
    """
    Conditional GAN combining Generator and Discriminator.
    
    This wrapper class provides convenient methods for training and inference.
    
    Args:
        latent_dim: Dimension of the noise vector (default: 100)
        embedding_dim: Dimension of condition embeddings (default: 50)
        ngf: Base number of generator filters (default: 64)
        ndf: Base number of discriminator filters (default: 64)
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        embedding_dim: int = 50,
        ngf: int = 64,
        ndf: int = 64
    ):
        super(cGAN, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.generator = Generator(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            ngf=ngf
        )
        
        self.discriminator = Discriminator(
            embedding_dim=embedding_dim,
            ndf=ndf
        )
    
    def generate(
        self,
        num_samples: int,
        skin_tone: torch.Tensor,
        lesion_type: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate images with specified conditions.
        
        Args:
            num_samples: Number of images to generate
            skin_tone: Skin tone labels of shape (num_samples,)
            lesion_type: Lesion type labels of shape (num_samples,)
            device: Device to generate on
            
        Returns:
            Generated images of shape (num_samples, 3, 64, 64)
        """
        noise = torch.randn(num_samples, self.latent_dim, device=device)
        
        with torch.inference_mode():
            images = self.generator(noise, skin_tone, lesion_type)
        
        return images
    
    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random noise for the generator."""
        return torch.randn(batch_size, self.latent_dim, device=device)


def get_models(
    latent_dim: int = 100,
    embedding_dim: int = 50,
    ngf: int = 64,
    ndf: int = 64,
    device: torch.device = None
) -> Tuple[Generator, Discriminator]:
    """
    Create Generator and Discriminator models.
    
    Args:
        latent_dim: Dimension of the noise vector
        embedding_dim: Dimension of condition embeddings
        ngf: Base number of generator filters
        ndf: Base number of discriminator filters
        device: Device to move models to
        
    Returns:
        Tuple of (Generator, Discriminator)
    """
    generator = Generator(
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        ngf=ngf
    )
    
    discriminator = Discriminator(
        embedding_dim=embedding_dim,
        ndf=ndf
    )
    
    if device is not None:
        generator = generator.to(device)
        discriminator = discriminator.to(device)
    
    return generator, discriminator


def get_wgan_models(
    latent_dim: int = 100,
    embedding_dim: int = 50,
    ngf: int = 64,
    ndf: int = 64,
    device: torch.device = None,
    gen_arch: str = "upsample",
    use_attention: bool = False,
    critic_norm: str = "instance",
    image_size: int = 64,
) -> Tuple[nn.Module, Critic]:
    """
    Create Generator and Critic models for WGAN-GP.

    Args:
        latent_dim: Dimension of the noise vector
        embedding_dim: Dimension of condition embeddings
        ngf: Base number of generator filters
        ndf: Base number of critic filters
        device: Device to move models to
        gen_arch: "upsample" (Upsample+Conv, default) or "deconv" (ConvTranspose2d).
        use_attention: If True, insert a SelfAttention block at the 32x32 stage of
            both networks. Only honored for gen_arch="upsample"; the legacy
            ConvTranspose2d Generator does not support attention.
        critic_norm: "instance" (default) or "layer". LayerNorm matches the
            original WGAN-GP paper recommendation.

    Returns:
        Tuple of (Generator, Critic)
    """
    if gen_arch == "upsample":
        generator = GeneratorUpsample(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            ngf=ngf,
            use_attention=use_attention,
            image_size=image_size,
        )
    elif gen_arch == "deconv":
        if use_attention:
            raise ValueError("use_attention=True is not supported with gen_arch='deconv'.")
        if image_size != 64:
            raise ValueError(
                f"gen_arch='deconv' is hardcoded to 64x64; got image_size={image_size}. "
                "Use gen_arch='upsample' for other resolutions."
            )
        generator = Generator(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            ngf=ngf,
        )
    else:
        raise ValueError(f"Unknown gen_arch={gen_arch!r}; expected 'upsample' or 'deconv'.")

    critic = Critic(
        embedding_dim=embedding_dim,
        ndf=ndf,
        norm=critic_norm,
        use_attention=use_attention,
        image_size=image_size,
    )

    if device is not None:
        generator = generator.to(device)
        critic = critic.to(device)

    return generator, critic


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models
    generator, discriminator = get_models(device=device)
    
    # Print model summaries
    print("\n" + "=" * 50)
    print("Generator Architecture:")
    print("=" * 50)
    print(generator)
    
    print("\n" + "=" * 50)
    print("Discriminator Architecture:")
    print("=" * 50)
    print(discriminator)
    
    # Test forward passes
    batch_size = 4
    latent_dim = 100
    
    # Generate test inputs
    noise = torch.randn(batch_size, latent_dim, device=device)
    skin_tones = torch.randint(0, NUM_SKIN_TONES, (batch_size,), device=device)
    lesion_types = torch.randint(0, NUM_LESION_TYPES, (batch_size,), device=device)
    
    print("\n" + "=" * 50)
    print("Testing Forward Passes:")
    print("=" * 50)
    
    # Test generator
    with torch.inference_mode():
        fake_images = generator(noise, skin_tones, lesion_types)
    print(f"Generator input: noise {noise.shape}, skin_tones {skin_tones.shape}, lesion_types {lesion_types.shape}")
    print(f"Generator output: {fake_images.shape}")
    print(f"Output range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
    
    # Test discriminator
    with torch.inference_mode():
        real_prob = discriminator(fake_images, skin_tones, lesion_types)
    print(f"\nDiscriminator input: images {fake_images.shape}")
    print(f"Discriminator output: {real_prob.shape}")
    print(f"Output range: [{real_prob.min():.3f}, {real_prob.max():.3f}]")
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nModel Parameters:")
    print(f"  Generator: {gen_params:,}")
    print(f"  Discriminator: {disc_params:,}")
    print(f"  Total: {gen_params + disc_params:,}")
