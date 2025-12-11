# utils/tools.py - Complete version

from typing import Optional, Union
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from PIL import Image
from pathlib import Path
import yaml


def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg
    
def update_ema(ema_model, model, decay=0.999):
    """
    Update EMA (Exponential Moving Average) weights.
    Keeps a shadow copy of the model that updates slowly for stable sampling.
    """
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def train_one_epoch(trainer, loader, optimizer, device, epoch, cfg_dropout=0.1):
    """
    Conditional training loop (WITH class labels)
    
    Used for:
    - Single-stage training
    - Stage 2 of two-stage training
    """
    trainer.train()
    total_loss = 0.
    num_batches = 0
    nan_count = 0

    with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
        for batch_idx, batch in enumerate(data):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, labels = batch
                labels = labels.to(device)
            else:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                labels = None
            
            optimizer.zero_grad()
            x_0 = images.to(device)
            
            if torch.isnan(x_0).any() or torch.isinf(x_0).any():
                print(f"\nWARNING: Skipping batch {batch_idx}: NaN/Inf in input")
                nan_count += 1
                continue
            
            # Classifier-free guidance training
            if labels is not None:
                drop_mask = torch.rand(labels.size(0), device=device) < cfg_dropout
                num_classes = trainer.model.num_class
                unconditional_label = torch.tensor(num_classes, device=device)
                labels = torch.where(drop_mask, unconditional_label, labels)
            
            loss = trainer(x_0, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: Skipping batch {batch_idx}: NaN/Inf loss")
                nan_count += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            data.set_description(f"Epoch: {epoch}")
            data.set_postfix(ordered_dict={
                "train_loss": total_loss / num_batches,
                "nan_batches": nan_count
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    if nan_count > 0:
        print(f"\nWarning: {nan_count} batches skipped due to NaN/Inf")
    
    return avg_loss


def train_one_epoch_unconditional(trainer, loader, optimizer, device, epoch):
    """
    Unconditional training loop (WITHOUT class labels)
    
    Used for Stage 1 of two-stage training
    """
    trainer.train()
    total_loss = 0.
    num_batches = 0
    nan_count = 0
    
    with tqdm(loader, dynamic_ncols=True, colour="#6565b5") as data:
        for batch_idx, batch in enumerate(data):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, _ = batch  # Ignore labels
            else:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
            
            optimizer.zero_grad()
            x_0 = images.to(device)
            
            if torch.isnan(x_0).any() or torch.isinf(x_0).any():
                print(f"\nWARNING: Skipping batch {batch_idx}: NaN/Inf in input")
                nan_count += 1
                continue
            
            # Train WITHOUT labels (unconditional)
            loss = trainer(x_0, class_labels=None)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: Skipping batch {batch_idx}: NaN/Inf loss")
                nan_count += 1
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            data.set_description(f"[Stage 1 - Unconditional] Epoch: {epoch}")
            data.set_postfix(ordered_dict={
                "train_loss": total_loss / num_batches,
                "nan_batches": nan_count
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    if nan_count > 0:
        print(f"\nWarning: {nan_count} batches skipped due to NaN/Inf")
    
    return avg_loss


def save_image(images: torch.Tensor, nrow: int = 8, show: bool = False, path: Optional[str] = None,
               format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """Save concatenated images."""
    images = images * 0.5 + 0.5
    images = torch.clamp(images, 0, 1)
    
    grid = make_grid(images, nrow=nrow, **kwargs)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        im.save(path, format=format)
    if show:
        im.show()
    return grid


def save_sample_image(images: torch.Tensor, show: bool = False, path: Optional[str] = None,
                      format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """Save images  intermediate diffusion steps."""
    images = images * 0.5 + 0.5
    images = torch.clamp(images, 0, 1)

    grid = []
    for i in range(images.shape[0]):
        t = make_grid(images[i], nrow=images.shape[1], **kwargs)
        grid.append(t)
    
    grid = torch.stack(grid, dim=0)
    grid = make_grid(grid, nrow=1, **kwargs)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        im.save(path, format=format)
    if show:
        im.show()
    return grid
