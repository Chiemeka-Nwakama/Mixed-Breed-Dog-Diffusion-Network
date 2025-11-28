import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from PIL import Image
from pathlib import Path
import yaml
from typing import Optional, Union


def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg


def train_one_epoch(trainer, loader, optimizer, device, epoch, cfg_dropout=0.1):
    """
    training loop optimized for small datasets (100+ images per class)
    
    Key changes:
    - Lower CFG dropout (0.1 instead of 0.15)
    - Gradient clipping
    - Better error handling
    - Loss validation
    """
    trainer.train()
    total_loss, total_num = 0., 0
    nan_count = 0

    with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
        for batch_idx, batch in enumerate(data):
            # Unpack batch
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, labels = batch
                labels = labels.to(device)
            else:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                labels = None
            
            optimizer.zero_grad()
            x_0 = images.to(device)
            
            # Validate input
            if torch.isnan(x_0).any() or torch.isinf(x_0).any():
                print(f"\n  Skipping batch {batch_idx}: NaN/Inf in input")
                nan_count += 1
                continue
            
            # Classifier-free guidance training
            # LOWER dropout for small datasets
            if labels is not None:
                drop_mask = torch.rand(labels.size(0), device=device) < cfg_dropout
                num_classes = trainer.model.num_class
                unconditional_label = torch.tensor(num_classes, device=device)
                labels = torch.where(drop_mask, unconditional_label, labels)
            
            # Forward pass
            loss = trainer(x_0, labels)
            
            # Validate loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  Skipping batch {batch_idx}: NaN/Inf loss")
                nan_count += 1
                continue
            
            # Backward pass
            loss.backward()
            
            # CRITICAL: Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_num += x_0.shape[0]
            
            data.set_description(f"Epoch: {epoch}")
            data.set_postfix(ordered_dict={
                "train_loss": total_loss / total_num,
                "nan_batches": nan_count
            })

    avg_loss = total_loss / total_num if total_num > 0 else float('inf')
    
    if nan_count > 0:
        print(f"\nWarning: {nan_count} batches skipped due to NaN/Inf")
    
    return avg_loss


def save_image(images: torch.Tensor, nrow: int = 8, show: bool = False, path: Optional[str] = None,
               format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    Save concatenated images.
    NOTE: show=False by default to avoid blocking during training
    """
    images = images * 0.5 + 0.5
    images = torch.clamp(images, 0, 1)  # Ensure valid range
    
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
    """
    Save images including intermediate diffusion steps.
    """
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


def validate_training_setup(model, dataloader, device):
    """
    Run validation checks before starting training
    """
    print("\n" + "="*50)
    print("Validating Training Setup")
    print("="*50)
    
    # Check model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check dataset
    dataset_size = len(dataloader.dataset)
    num_classes = len(dataloader.dataset.classes)
    print(f"\nDataset size: {dataset_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Average per class: {dataset_size / num_classes:.1f}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.to(device)
    model.eval()
    
    try:
        batch = next(iter(dataloader))
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        # Test with timestep
        t = torch.randint(0, 1000, (images.shape[0],), device=device)
        with torch.no_grad():
            output = model(images, t, labels)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {images.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check for NaN
        if torch.isnan(output).any():
            print(" WARNING: NaN detected in output!")
            return False
        
    except Exception as e:
        print(f" Forward pass failed: {e}")
        return False
    
    print("="*50 + "\n")
    return True


def get_optimizer_for_small_dataset(model, base_lr=1e-4):
    """
    Create optimizer optimized for small datasets
    """
    # Lower learning rate for stability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,  # Small weight decay for regularization
        eps=1e-8
    )
    return optimizer


def get_scheduler(optimizer, total_epochs, warmup_epochs=10):
    """
    Learning rate scheduler with warmup
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler