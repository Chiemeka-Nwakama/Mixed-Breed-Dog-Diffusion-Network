from pyspark.sql import SparkSession
from PIL import Image
from io import BytesIO
import pandas as pd

class DeltaImageDataset:
    """
    Dataset for preprocessed images stored in Delta.
    Images are already cropped (bbox) and resized to 64x64.
    Compatible with PyTorch DataLoader when torch is available.
    """
    def __init__(self, delta_path, transform=None, mode="RGB"):
        self.transform = transform
        self.mode = mode
        
        print(f"\nLoading dataset from: {delta_path}")
        
        # Load delta into pandas
        spark = SparkSession.builder.getOrCreate()
        self.data = spark.read.format("delta").load(delta_path).toPandas()
        
        # Extract classes
        self.classes = sorted(self.data['cls'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        print(f"✓ Loaded {len(self.data)} samples")
        print(f"✓ Classes: {len(self.classes)}")
        print(f"✓ Avg per class: {len(self.data) / len(self.classes):.1f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            # Decode preprocessed image (already 64x64 PNG)
            img_bytes = row['img_bytes_prepared']
            image = Image.open(BytesIO(img_bytes)).convert(self.mode)
            
            if self.transform:
                image = self.transform(image)
            
            return image, row['label']
            
        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))


def create_delta_dataloader(delta_path, batch_size, **kwargs):
    """
    Creates DataLoader from delta dataset.
    Compatible with config.yml settings.
    Requires torch and torchvision to be installed.
    """
    # Import torch here so it's only required when actually creating the dataloader
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for training. Install with: pip install torch torchvision"
        ) from e
    
    mode = kwargs.get("mode", "RGB")
    norm = (0.5,) if mode == "L" else (0.5, 0.5, 0.5)
    
    image_size = kwargs.get("image_size", 128)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    
    print(f"\n{'='*50}")
    print(f"Delta Dataset Configuration")
    print(f"{'='*50}")
    print(f"Path: {delta_path}")
    print(f"Target size: {image_size}")
    print(f"Batch size: {batch_size}")
    
    # Transforms: resize from 64x64 to target, augment, normalize
    trans = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm, norm)
    ])
    
    dataset = DeltaImageDataset(
        delta_path=delta_path,
        transform=trans,
        mode=mode
    )
    
    print(f"{'='*50}\n")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 2)
    )
