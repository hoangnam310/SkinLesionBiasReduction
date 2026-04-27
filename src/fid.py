from utils import compute_fid
from data_process import get_dataloader
from cgan import get_models
import torch

NUM_SAMPLES = 500  # Reduce for faster/less memory, increase for more accurate FID
BATCH_SIZE = 32    # For generation

if __name__ == '__main__':
    # Setup
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # 1. Load real images incrementally (not all at once)
    print("Loading real images...")
    dataloader, dataset = get_dataloader(
        csv_path="../dataset/fitzpatrick17k_cleaned.csv",
        image_dir="../dataset/images",
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    real_images = []
    count = 0
    for imgs, _, _ in dataloader:
        real_images.append(imgs)
        count += imgs.size(0)
        if count >= NUM_SAMPLES:
            break
    real_images = torch.cat(real_images, dim=0)[:NUM_SAMPLES]
    print(f"Loaded {len(real_images)} real images")

    # 2. Generate fake images in batches
    print("Generating fake images...")
    generator, _ = get_models(latent_dim=100, device=device)
    checkpoint = torch.load("../outputs/20260131_211809/checkpoints/checkpoint_epoch_0070.pt", map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    fake_images = []
    with torch.no_grad():
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            n = min(BATCH_SIZE, NUM_SAMPLES - i)
            noise = torch.randn(n, 100, device=device)
            skin_tones = torch.randint(0, 6, (n,), device=device)
            lesion_types = torch.randint(0, 3, (n,), device=device)
            fake_images.append(generator(noise, skin_tones, lesion_types).cpu())
    fake_images = torch.cat(fake_images, dim=0)
    print(f"Generated {len(fake_images)} fake images")

    # Free generator memory
    del generator
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    # 3. Compute FID
    print("Computing FID...")
    fid = compute_fid(real_images, fake_images, device)
    print(f"FID: {fid:.2f}")  # Lower is better
