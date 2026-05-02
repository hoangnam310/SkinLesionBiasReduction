"""
WGAN-GP training script for the conditional GAN on Fitzpatrick17k.

Differences from src/train.py:
- Uses `Critic` (no Sigmoid, InstanceNorm) instead of `Discriminator`.
- Wasserstein loss with gradient penalty (Gulrajani et al. 2017).
- `n_critic` critic updates per generator update.
- DiffAugment (Zhao et al. 2020) applied to real / fake / interpolated to
  prevent the critic from overfitting on a small medical dataset.
- Adam(beta1=0, beta2=0.9) defaults — standard for WGAN-GP.
- Defaults to the `GeneratorUpsample` architecture (Upsample + Conv2d) to
  avoid the checkerboard artifacts that ConvTranspose2d produces.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_process import get_dataloader, NUM_SKIN_TONES, NUM_LESION_TYPES
from cgan import get_wgan_models
from diffaugment import diff_augment
from utils import (
    weights_init,
    generate_samples,
    visualize_samples,
    setup_logging,
    log_message,
    compute_fid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train WGAN-GP cGAN on Fitzpatrick17k")

    parser.add_argument("--csv_path", type=str, default="dataset/fitzpatrick17k_cleaned.csv")
    parser.add_argument("--image_dir", type=str, default="dataset/images")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_c", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Critic updates per generator update.")
    parser.add_argument("--lambda_gp", type=float, default=10.0,
                        help="Gradient penalty coefficient.")
    parser.add_argument("--diffaug_policy", type=str, default="color,translation,cutout",
                        help="DiffAugment policy. Empty string disables.")

    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--gen_arch", type=str, default="upsample",
                        choices=["upsample", "deconv"])

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--sample_interval", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--fid_interval", type=int, default=20)
    parser.add_argument("--fid_num_samples", type=int, default=1000)

    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def get_device(device_str=None):
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    skin_tones: torch.Tensor,
    lesion_types: torch.Tensor,
    device: torch.device,
    diffaug_policy: str,
) -> torch.Tensor:
    """Standard WGAN-GP gradient penalty on alpha-interpolated samples.

    DiffAugment is applied inside the chain so the critic sees augmented inputs
    consistently; gradients are taken w.r.t. the raw interpolated tensor since
    that is what carries `requires_grad`.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device, dtype=real.dtype)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)

    interp_in = diff_augment(interpolated, diffaug_policy) if diffaug_policy else interpolated
    score = critic(interp_in, skin_tones, lesion_types)

    grads = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.reshape(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def train_epoch(
    generator,
    critic,
    dataloader,
    optimizer_g,
    optimizer_c,
    device,
    latent_dim,
    n_critic,
    lambda_gp,
    diffaug_policy,
):
    generator.train()
    critic.train()

    sum_loss_c = 0.0
    sum_loss_g = 0.0
    sum_w_dist = 0.0
    sum_gp = 0.0
    n_critic_steps = 0
    n_gen_steps = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    step_in_epoch = 0

    for real_images, skin_tones, lesion_types in pbar:
        real_images = real_images.to(device)
        skin_tones = skin_tones.to(device)
        lesion_types = lesion_types.to(device)
        batch_size = real_images.size(0)

        # ---- Critic step ----
        optimizer_c.zero_grad()

        noise = torch.randn(batch_size, latent_dim, device=device)
        with torch.no_grad():
            fake_images = generator(noise, skin_tones, lesion_types)

        real_in = diff_augment(real_images, diffaug_policy) if diffaug_policy else real_images
        fake_in = diff_augment(fake_images, diffaug_policy) if diffaug_policy else fake_images

        score_real = critic(real_in, skin_tones, lesion_types)
        score_fake = critic(fake_in, skin_tones, lesion_types)

        gp = gradient_penalty(
            critic, real_images, fake_images,
            skin_tones, lesion_types, device, diffaug_policy,
        )

        wasserstein_dist = score_real.mean() - score_fake.mean()
        loss_c = -wasserstein_dist + lambda_gp * gp
        loss_c.backward()
        optimizer_c.step()

        sum_loss_c += loss_c.item()
        sum_w_dist += wasserstein_dist.item()
        sum_gp += gp.item()
        n_critic_steps += 1

        # ---- Generator step (every n_critic critic steps) ----
        if (step_in_epoch + 1) % n_critic == 0:
            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise, skin_tones, lesion_types)
            fake_in = diff_augment(fake_images, diffaug_policy) if diffaug_policy else fake_images
            loss_g = -critic(fake_in, skin_tones, lesion_types).mean()
            loss_g.backward()
            optimizer_g.step()

            sum_loss_g += loss_g.item()
            n_gen_steps += 1

            pbar.set_postfix({
                "C_loss": f"{loss_c.item():.3f}",
                "G_loss": f"{loss_g.item():.3f}",
                "W_dist": f"{wasserstein_dist.item():.3f}",
                "GP": f"{gp.item():.3f}",
            })
        else:
            pbar.set_postfix({
                "C_loss": f"{loss_c.item():.3f}",
                "W_dist": f"{wasserstein_dist.item():.3f}",
                "GP": f"{gp.item():.3f}",
            })

        step_in_epoch += 1

    return {
        "loss_c": sum_loss_c / max(n_critic_steps, 1),
        "loss_g": sum_loss_g / max(n_gen_steps, 1),
        "w_dist": sum_w_dist / max(n_critic_steps, 1),
        "gp": sum_gp / max(n_critic_steps, 1),
    }


def save_wgan_checkpoint(generator, critic, optimizer_g, optimizer_c, epoch, metrics, path):
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_c_state_dict": optimizer_c.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        },
        path,
    )


def load_wgan_checkpoint(path, generator, critic, optimizer_g, optimizer_c, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator_state_dict"])
    critic.load_state_dict(ckpt["critic_state_dict"])
    if optimizer_g is not None and "optimizer_g_state_dict" in ckpt:
        optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
    if optimizer_c is not None and "optimizer_c_state_dict" in ckpt:
        optimizer_c.load_state_dict(ckpt["optimizer_c_state_dict"])
    return ckpt


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"wgan_{timestamp}"
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = setup_logging(log_dir)
    log_message(log_file, f"Starting WGAN-GP training with args: {args}")

    writer = SummaryWriter(log_dir / "tensorboard")

    device = get_device(args.device)
    log_message(log_file, f"Using device: {device}")

    log_message(log_file, "Loading dataset...")
    dataloader, _ = get_dataloader(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    log_message(log_file, f"Creating models (gen_arch={args.gen_arch})...")
    generator, critic = get_wgan_models(
        latent_dim=args.latent_dim,
        embedding_dim=args.embedding_dim,
        ngf=args.ngf,
        ndf=args.ndf,
        device=device,
        gen_arch=args.gen_arch,
    )
    generator.apply(weights_init)
    critic.apply(weights_init)

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_c = optim.Adam(critic.parameters(), lr=args.lr_c, betas=(args.beta1, args.beta2))

    start_epoch = 0
    if args.resume:
        log_message(log_file, f"Resuming from checkpoint: {args.resume}")
        ckpt = load_wgan_checkpoint(args.resume, generator, critic, optimizer_g, optimizer_c, device)
        start_epoch = ckpt.get("epoch", 0)
        log_message(log_file, f"Resumed from epoch {start_epoch}")

    fixed_noise = torch.randn(args.num_samples, args.latent_dim, device=device)
    fixed_skin_tones = torch.tensor(
        [i % NUM_SKIN_TONES for i in range(args.num_samples)], device=device
    )
    fixed_lesion_types = torch.tensor(
        [i % NUM_LESION_TYPES for i in range(args.num_samples)], device=device
    )

    log_message(log_file, f"Starting training for {args.epochs} epochs...")
    diffaug_policy = args.diffaug_policy or ""

    metrics = {}
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        metrics = train_epoch(
            generator=generator,
            critic=critic,
            dataloader=dataloader,
            optimizer_g=optimizer_g,
            optimizer_c=optimizer_c,
            device=device,
            latent_dim=args.latent_dim,
            n_critic=args.n_critic,
            lambda_gp=args.lambda_gp,
            diffaug_policy=diffaug_policy,
        )

        epoch_time = time.time() - epoch_start

        log_message(
            log_file,
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"C_loss: {metrics['loss_c']:.4f}, G_loss: {metrics['loss_g']:.4f}, "
            f"W_dist: {metrics['w_dist']:.4f}, GP: {metrics['gp']:.4f}, "
            f"Time: {epoch_time:.2f}s",
        )

        writer.add_scalar("Loss/Critic", metrics["loss_c"], epoch)
        writer.add_scalar("Loss/Generator", metrics["loss_g"], epoch)
        writer.add_scalar("Metric/Wasserstein", metrics["w_dist"], epoch)
        writer.add_scalar("Metric/GradientPenalty", metrics["gp"], epoch)

        if (epoch + 1) % args.sample_interval == 0:
            generator.eval()
            with torch.inference_mode():
                fake_samples = generator(fixed_noise, fixed_skin_tones, fixed_lesion_types)
            visualize_samples(
                fake_samples, fixed_skin_tones, fixed_lesion_types,
                save_path=sample_dir / f"samples_epoch_{epoch+1:04d}.png",
            )
            from torchvision.utils import make_grid
            grid = make_grid(fake_samples, normalize=True, value_range=(-1, 1))
            writer.add_image("Generated Samples", grid, epoch)
            generator.train()

        if (epoch + 1) % args.checkpoint_interval == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pt"
            save_wgan_checkpoint(generator, critic, optimizer_g, optimizer_c, epoch + 1, metrics, ckpt_path)
            log_message(log_file, f"Saved checkpoint: {ckpt_path}")

        if args.fid_interval > 0 and (epoch + 1) % args.fid_interval == 0:
            log_message(log_file, "Computing FID...")
            generator.eval()

            real_images_buf = []
            for images, _, _ in dataloader:
                real_images_buf.append(images)
                if sum(b.size(0) for b in real_images_buf) >= args.fid_num_samples:
                    break
            real_images_buf = torch.cat(real_images_buf, dim=0)[: args.fid_num_samples]

            with torch.inference_mode():
                fake_buf = []
                for i in range(0, args.fid_num_samples, args.batch_size):
                    n = min(args.batch_size, args.fid_num_samples - i)
                    noise = torch.randn(n, args.latent_dim, device=device)
                    st = torch.randint(0, NUM_SKIN_TONES, (n,), device=device)
                    lt = torch.randint(0, NUM_LESION_TYPES, (n,), device=device)
                    fake_buf.append(generator(noise, st, lt).cpu())
                fake_buf = torch.cat(fake_buf, dim=0)

            fid = compute_fid(real_images_buf, fake_buf, device)
            log_message(log_file, f"FID: {fid:.2f}")
            writer.add_scalar("FID", fid, epoch)
            generator.train()

    final_path = checkpoint_dir / "final_model.pt"
    save_wgan_checkpoint(generator, critic, optimizer_g, optimizer_c, args.epochs, metrics, final_path)
    log_message(log_file, f"Training complete. Final model saved to {final_path}")
    writer.close()

    log_message(log_file, "Generating samples for all skin tones...")
    for skin_tone in range(NUM_SKIN_TONES):
        for lesion_type in range(NUM_LESION_TYPES):
            samples = generate_samples(
                generator, device, args.latent_dim,
                skin_tone=skin_tone, lesion_type=lesion_type, num_samples=8,
            )
            visualize_samples(
                samples,
                torch.full((8,), skin_tone, device=device),
                torch.full((8,), lesion_type, device=device),
                save_path=sample_dir / f"final_st{skin_tone+1}_lt{lesion_type}.png",
                nrow=4,
            )

    log_message(log_file, "All done!")


if __name__ == "__main__":
    main()
