import torch
import torch.nn as nn
from typing import Tuple

from data_process import NUM_SKIN_TONES, NUM_LESION_TYPES


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
    ):
        super(GeneratorUpsample, self).__init__()

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

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

        self.up = nn.Sequential(
            *up_block(ngf * 16, ngf * 8),   # 4x4 -> 8x8
            *up_block(ngf * 8, ngf * 4),    # 8x8 -> 16x16
            *up_block(ngf * 4, ngf * 2),    # 16x16 -> 32x32
            *up_block(ngf * 2, 3, final=True),  # 32x32 -> 64x64
            nn.Tanh(),
        )

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
        ndf: int = 64
    ):
        super(Critic, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Condition embeddings
        self.skin_tone_embedding = nn.Embedding(num_skin_tones, embedding_dim)
        self.lesion_type_embedding = nn.Embedding(num_lesion_types, embedding_dim)
        
        # Project condition embeddings to spatial representation
        self.skin_embed_proj = nn.Linear(embedding_dim, 64 * 64)
        self.lesion_embed_proj = nn.Linear(embedding_dim, 64 * 64)
        
        # Input channels: 3 (image) + 2 (condition channels)
        # No BatchNorm for WGAN-GP (interferes with gradient penalty)
        # Using InstanceNorm instead which works per-sample
        self.conv = nn.Sequential(
            # Layer 1: 5 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(3 + 2, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 128 x 16 x 16 -> 256 x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 256 x 8 x 8 -> 512 x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: 512 x 4 x 4 -> 1 x 1 x 1
            # No Sigmoid! Output is unbounded for Wasserstein distance
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True)
        )
    
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
        skin_channel = self.skin_embed_proj(skin_embed).view(batch_size, 1, 64, 64)
        lesion_channel = self.lesion_embed_proj(lesion_embed).view(batch_size, 1, 64, 64)
        
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

    Returns:
        Tuple of (Generator, Critic)
    """
    if gen_arch == "upsample":
        generator = GeneratorUpsample(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            ngf=ngf,
        )
    elif gen_arch == "deconv":
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
