import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root, suffix=("jpg", "png"), transform=None, mode="RGB"):
        self.root = Path(root)
        self.transform = transform
        self.suffix = suffix
        self.mode = mode  # support for different image modes

        # Find classes (subfolders)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect image paths with labels
        self.samples = []
        for cls in self.classes:
            folder = self.root / cls
            for ext in suffix:
                # recursive search in case there are nested subfolders
                for img_path in folder.rglob(f"*.{ext}"):
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # convert image to the specified mode (RGB/L/CMYK)
        image = Image.open(img_path).convert(self.mode)

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
