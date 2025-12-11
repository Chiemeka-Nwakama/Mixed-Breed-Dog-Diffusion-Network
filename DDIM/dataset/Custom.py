import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import random

class ImageDataset(Dataset):
    def __init__(self, root, ann_root, suffix=("jpg", "png"), transform=None, mode="RGB"):
        self.root = Path(root)
        self.ann_root = Path(ann_root)
        self.transform = transform
        self.suffix = suffix
        self.mode = mode
        
        # Find classes (subfolders)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        print(f"\nScanning dataset in: {self.root}")
        print(f"Found {len(self.classes)} classes: {self.classes[:3]}..." if len(self.classes) > 3 else f"Classes: {self.classes}")
        
        # Collect image paths with labels
        self.samples = []
        corrupted_count = 0
        
        for cls in self.classes:
            folder = self.root / cls
            class_samples = []
            
            for ext in suffix:
                for img_path in folder.rglob(f"*.{ext}"):
                    # Validate image can be opened
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # Verify it's actually an image
                        # Re-open because verify() closes the file
                        with Image.open(img_path) as img:
                            img.convert(self.mode)  # Test conversion
                        class_samples.append((img_path, self.class_to_idx[cls]))
                    except Exception as e:
                        print(f"  Skipping corrupted: {img_path.name}")
                        corrupted_count += 1
            
            self.samples.extend(class_samples)
            print(f"  {cls}: {len(class_samples)} valid images")
        
        if corrupted_count > 0:
            print(f"\n  Skipped {corrupted_count} corrupted images")
        
        if len(self.samples) == 0:
            raise ValueError(f" No valid images found in {root}!\nCheck that:")
            print("  1. Path points to 'Images' folder")
            print("  2. Images are in breed subfolders")
            print("  3. Images are .jpg or .png")
        
        print(f"\n✓ Total valid samples: {len(self.samples)}")
        print(f"✓ Number of classes: {len(self.classes)}")
        print(f"✓ Average per class: {len(self.samples) / len(self.classes):.1f}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def load_bbox(self, img_path):
        """Load bounding box from matching XML annotation."""
        cls = img_path.parent.name
        xml_name = img_path.stem + ".xml"  # FIXED: Added .xml extension
        xml_path = self.ann_root / cls / xml_name
        
        if not xml_path.exists():
            return None
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bbox = root.find("object").find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            # Validate bbox
            if xmax <= xmin or ymax <= ymin:
                return None
                
            return (xmin, ymin, xmax, ymax)
        except Exception:
            return None
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert(self.mode)
            
            # Load and apply bounding box
            bbox = self.load_bbox(img_path)
            if bbox is not None:
                image = image.crop(bbox)
            
            # Safety check for very small images
            if image.size[0] < 32 or image.size[1] < 32:
                image = image.resize((64, 64))
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f" Error loading {img_path.name}: {e}")
            # Return a random valid sample instead
            return self.__getitem__((idx + 1) % len(self.samples))
        
        



def create_custom_dataset(data_path, batch_size, **kwargs):
    """
    Creates a DataLoader for the custom dataset.
    
    OPTIMIZED FOR SMALL DATASETS (100+ images per class)
    
    Args:
        data_path (str): Root folder of dataset (should point to 'Images' folder)
        batch_size (int): Batch size for DataLoader
        kwargs:
            mode (str): "RGB" or "L"
            image_size (tuple or int): Resize image size
            suffix (tuple): Allowed image extensions
            shuffle (bool)
            drop_last (bool)
            pin_memory (bool)
            num_workers (int)
    """
    mode = kwargs.get("mode", "RGB")
    norm = (0.5,) if mode == "L" else (0.5, 0.5, 0.5)
    
    # Handle both tuple and int for image_size
    image_size = kwargs.get("image_size", (64, 64))
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    
    print(f"\n{'='*50}")
    print(f"Dataset Configuration")
    print(f"{'='*50}")
    print(f"Data path: {data_path}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Mode: {mode}")
    
    # AGGRESSIVE AUGMENTATION for small datasets
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(norm, norm)
    ])
    
    # Check if annotation path exists
    ann_root = Path(data_path).parent / "Annotation"
    if not ann_root.exists():
        print(f"\n  Warning: Annotation folder not found at {ann_root}")
        print("   Images will not be cropped to bounding boxes")
    
    dataset = ImageDataset(
        root=data_path,
        ann_root=ann_root,
        suffix=kwargs.get("suffix", ("jpg", "png")),
        transform=trans,
        mode=mode
    )
    
    # Validate dataset size
    total_samples = len(dataset)
    num_classes = len(dataset.classes)
    avg_per_class = total_samples / num_classes
    
    print(f"{'='*50}")
    print(f"Dataset Statistics")
    print(f"{'='*50}")
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {num_classes}")
    print(f"Average per class: {avg_per_class:.1f}")
    
    if avg_per_class < 50:
        print("\n CRITICAL: Less than 50 images per class!")
        print("   Recommendation: Get more data or reduce number of classes")
    elif avg_per_class < 100:
        print("\n  WARNING: Small dataset (50-100 per class)")
        print("   Training will be challenging but possible")
    elif avg_per_class < 500:
        print("\n✓ Moderate dataset size (100-500 per class)")
        print("   Should work with proper training")
    else:
        print("\n✓✓ Good dataset size (500+ per class)")
    
    print(f"{'='*50}\n")
    
    
        # --- Preview 3 images from the dataset ---
    print("Previewing 3 images from the dataset...")

    # Take first 3 images
    images, labels = zip(*[dataset[i] for i in range(3)])

    # Convert to HWC + unnormalize
    images = torch.stack(images).permute(0, 2, 3, 1).cpu().numpy()
    images = (images * 0.5 + 0.5).clip(0, 1)

    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i])
        plt.title(dataset.classes[labels[i]])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 2)
    )
    

  
   

    
    return dataloader