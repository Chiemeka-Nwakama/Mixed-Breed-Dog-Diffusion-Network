import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET


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

        # Collect image paths with labels
        self.samples = []
        for cls in self.classes:
            folder = self.root / cls
            for ext in suffix:
                for img_path in folder.rglob(f"*.{ext}"):
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def load_bbox(self, img_path):
        """Load bounding box from matching XML annotation."""
        # Example: Images/n02085620-Chihuahua/n02085620_10074.jpg
        # Matching annotation: Annotation/n02085620-Chihuahua/n02085620_10074.xml
        cls = img_path.parent.name
        xml_name = img_path.stem + ".xml"
        xml_path = self.ann_root / cls / xml_name

        if not xml_path.exists():
            return None  # fallback: no crop

        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        return (xmin, ymin, xmax, ymax)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert(self.mode)

        # Load bounding box
        bbox = self.load_bbox(img_path)
        if bbox is not None:
            image = image.crop(bbox)

        if self.transform:
            image = self.transform(image)

        return image, label


def create_custom_dataset(data_path, batch_size, **kwargs):
    """
    Creates a DataLoader for the custom dataset.

    Args:
        data_path (str): Root folder of dataset
        batch_size (int): Batch size for DataLoader
        kwargs:
            mode (str): "RGB" or "L"
            image_size (tuple): Resize image size
            suffix (tuple): Allowed image extensions
            shuffle (bool)
            drop_last (bool)
            pin_memory (bool)
            num_workers (int)
    """
    mode = kwargs.get("mode", "RGB")
    norm = (0.5,) if mode == "L" else (0.5, 0.5, 0.5)
    image_size = kwargs.get("image_size", (256, 256))

    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(norm, norm)
    ])

    dataset = ImageDataset(
        root=data_path,  # corrected parameter name
        ann_root= Path(data_path).parent / "Annotation",
        suffix=kwargs.get("suffix", ("png", "jpg")),
        transform=trans,
        mode=mode  # support for RGB/L
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 4)
    )

    return dataloader
