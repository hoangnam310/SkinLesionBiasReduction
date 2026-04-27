import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Callable, Tuple, Optional
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Label encoding mappings
FITZPATRICK_ENCODING = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
LESION_TYPE_ENCODING = {"benign": 0, "malignant": 1, "non-neoplastic": 2}

# Reverse mappings for decoding
FITZPATRICK_DECODING = {v: k for k, v in FITZPATRICK_ENCODING.items()}
LESION_TYPE_DECODING = {v: k for k, v in LESION_TYPE_ENCODING.items()}

# Number of classes
NUM_SKIN_TONES = 6
NUM_LESION_TYPES = 3


class SkinLesionDataset(Dataset):
    """
    Args:
        csv_path: Path to the CSV file containing image metadata
        image_dir: Directory containing the image files
        transform: Optional torchvision transforms to apply
        filter_skin_tones: Optional list of skin tones to include (1-6)
        filter_lesion_types: Optional list of lesion types to include
        preprocess_fn: Optional callable applied to the raw image (uint8 RGB
            numpy array) before `transform`. Use `segmentation.CV2Preprocess`
            to run the cv2-only segmentation-aware crop at load time. For
            SAM-based preprocessing, prefer `src/preprocess_segmentation.py`
            and point `image_dir` at the resulting directory instead.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        filter_skin_tones: Optional[list] = None,
        filter_lesion_types: Optional[list] = None,
        preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.image_dir = image_dir
        self.preprocess_fn = preprocess_fn

        # Load and clean the CSV data
        self.df = pd.read_csv(csv_path)
        
        # Remove invalid fitzpatrick_scale values (-1)
        self.df = self.df[self.df["fitzpatrick_scale"] != -1].reset_index(drop=True)
        
        # Apply optional filters
        if filter_skin_tones is not None:
            self.df = self.df[
                self.df["fitzpatrick_scale"].isin(filter_skin_tones)
            ].reset_index(drop=True)
            
        if filter_lesion_types is not None:
            self.df = self.df[
                self.df["three_partition_label"].isin(filter_lesion_types)
            ].reset_index(drop=True)
        
        # Verify images exist
        self._verify_images()
        
        # Default transform: resize to 64x64, normalize to [-1, 1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
    def download_image(row):
        """Download a single image and save it using its md5hash as filename."""
        url = row['url']
        md5hash = row['md5hash']
        filepath = f'../dataset/images/{md5hash}.jpg'
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            return md5hash, 'skipped'
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return md5hash, 'success'
        
        except Exception as e:
            return md5hash, f'failed: {str(e)}'
    def download_all_images(df, max_workers=10):
        """Download images in parallel using ThreadPoolExecutor."""
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        failed_urls = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_image, row): row['md5hash'] 
                    for _, row in df.iterrows()}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                md5hash, status = future.result()
                if status == 'success':
                    results['success'] += 1
                elif status == 'skipped':
                    results['skipped'] += 1
                else:
                    results['failed'] += 1
                    failed_urls.append((md5hash, status))
        
        print(f"Downloaded: {results['success']}")
        print(f"Skipped (already exists): {results['skipped']}")
        print(f"Failed: {results['failed']}")
        
        return failed_urls
    def _verify_images(self) -> None:
        """Verify that image files exist and filter out missing ones."""
        valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.image_dir, f"{row['md5hash']}.jpg")
            if os.path.exists(img_path):
                valid_indices.append(idx)
        
        if len(valid_indices) < len(self.df):
            print(f"Warning: {len(self.df) - len(valid_indices)} images not found. "
                  f"Using {len(valid_indices)} available images.")
            self.df = self.df.loc[valid_indices].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, skin_tone_label, lesion_type_label)
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, f"{row['md5hash']}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Optional segmentation-aware preprocessing (numpy in / numpy out)
        if self.preprocess_fn is not None:
            image = Image.fromarray(self.preprocess_fn(np.array(image)))

        # Apply transforms
        image = self.transform(image)
        
        # Encode labels
        skin_tone = torch.tensor(FITZPATRICK_ENCODING[row["fitzpatrick_scale"]], 
                                 dtype=torch.long)
        lesion_type = torch.tensor(LESION_TYPE_ENCODING[row["three_partition_label"]], 
                                   dtype=torch.long)
        
        return image, skin_tone, lesion_type
    
def get_dataloader(
    csv_path: str,
    image_dir: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
    filter_skin_tones: Optional[list] = None,
    filter_lesion_types: Optional[list] = None,
    pin_memory: bool = True
) -> Tuple[DataLoader, SkinLesionDataset]:
    """
    Create a DataLoader for the skin lesion dataset.
    
    Args:
        csv_path: Path to the CSV file
        image_dir: Directory containing images
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        transform: Optional custom transforms
        filter_skin_tones: Optional list of skin tones to include
        filter_lesion_types: Optional list of lesion types to include
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    dataset = SkinLesionDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=transform,
        filter_skin_tones=filter_skin_tones,
        filter_lesion_types=filter_lesion_types
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    return dataloader, dataset


def get_augmented_transform(image_size: int = 64) -> transforms.Compose:
    """
    Get augmented transforms for training.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms with augmentation
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data loading")
    parser.add_argument("--csv_path", type=str, 
                        default="dataset/fitzpatrick17k_cleaned.csv")
    parser.add_argument("--image_dir", type=str, 
                        default="dataset/images")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--download_images", type=bool, default=False)
    args = parser.parse_args()
    if args.download_images == "True":
        df = pd.read_csv('../dataset/fitzpatrick17k.csv')
        os.makedirs('../dataset/images', exist_ok=True) 
        failed = SkinLesionDataset.download_all_images(df, max_workers=300)
        print(failed)
    print("Loading dataset...")
    dataloader, dataset = get_dataloader(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size
    )
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test loading a batch
    print("\nTesting batch loading...")
    images, skin_tones, lesion_types = next(iter(dataloader))
    print(f"Image batch shape: {images.shape}")
    print(f"Skin tone labels shape: {skin_tones.shape}")
    print(f"Lesion type labels shape: {lesion_types.shape}")
    print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")