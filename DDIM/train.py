from dataset.Custom import create_custom_dataset
from model.UNet import UNet
from utils.engine import GaussianDiffusionTrainer
from utils.tools import train_one_epoch, train_one_epoch_unconditional, load_yaml
import torch
from utils.callbacks import ModelCheckpoint


def train(config):
    """Single-stage conditional training"""
    consume = config.get("consume", False)
    if consume:
        cp = torch.load(config["consume_path"])
        config = cp["config"]
    
    print(config)
    device = torch.device(config["device"])
    
    loader = create_custom_dataset(**config["Dataset"])
    
    start_epoch = 1
    model = UNet(**config["Model"]).to(device)
    
    # Get learning rate and cfg_dropout from config
    lr = config.get("lr", 0.0001)
    cfg_dropout = config.get("cfg_dropout", 0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)
    model_checkpoint = ModelCheckpoint(**config["Callback"])
    
    if consume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1
    
    for epoch in range(start_epoch, config["epochs"] + 1):
        print(f"Starting Epoch {epoch}/{config['epochs']}")
        
        # Pass cfg_dropout to training function
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch, cfg_dropout)
        
        print(f"Epoch {epoch} finished. Avg Loss: {loss:.6f}")
        model_checkpoint.step(
            loss, 
            model=model.state_dict(), 
            config=config,
            optimizer=optimizer.state_dict(), 
            start_epoch=epoch,
            model_checkpoint=model_checkpoint.state_dict()
        )


def train_stage1(config):
    """Stage 1: Unconditional training (all dogs, no labels)"""
    print("\n" + "="*60)
    print("STAGE 1: UNCONDITIONAL TRAINING")
    print("Learning general dog features from all images")
    print("="*60 + "\n")
    
    device = torch.device(config["device"])
    loader = create_custom_dataset(**config["Dataset"])
    
    model = UNet(**config["Model"]).to(device)
    
    lr = config.get("stage1_lr", 0.0002)
    print(f"Learning Rate: {lr}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)
    
    # Use separate checkpoint callback for Stage 1
    stage1_callback_config = {
        "save_path": "./checkpoints_stage1.pth",
        "save_freq": config.get("save_interval", 20)
    }
    model_checkpoint = ModelCheckpoint(**stage1_callback_config)
    
    epochs = config.get("stage1_epochs", 80)
    
    for epoch in range(1, epochs + 1):
        print(f"Starting Epoch {epoch}/{epochs}")
        
        # Use unconditional training
        loss = train_one_epoch_unconditional(trainer, loader, optimizer, device, epoch)
        
        print(f"Epoch {epoch} finished. Avg Loss: {loss:.6f}")
        model_checkpoint.step(
            loss,
            model=model.state_dict(),
            config=config,
            optimizer=optimizer.state_dict(),
            start_epoch=epoch,
            model_checkpoint=model_checkpoint.state_dict()
        )
    
    print("\n" + "="*60)
    print("STAGE 1 COMPLETE!")
    print(f"Best checkpoint: ./checkpoints_stage1/")
    print("="*60 + "\n")
    
    # Return path to best checkpoint
    return "./checkpoints_stage1/best.pth"


def train_stage2(config, stage1_checkpoint):
    """Stage 2: Conditional fine-tuning (breed-specific)"""
    print("\n" + "="*60)
    print("STAGE 2: CONDITIONAL FINE-TUNING")
    print("Adding breed-specific conditioning")
    print("="*60 + "\n")
    
    device = torch.device(config["device"])
    loader = create_custom_dataset(**config["Dataset"])
    
    # Load Stage 1 model
    print(f"Loading Stage 1 checkpoint: {stage1_checkpoint}")
    cp = torch.load(stage1_checkpoint)
    
    model = UNet(**config["Model"]).to(device)
    model.load_state_dict(cp["model"])
    
    # Lower learning rate for fine-tuning
    lr = config.get("stage2_lr", 0.00005)
    print(f"Learning Rate: {lr} (lower for fine-tuning)")
    
    cfg_dropout = config.get("cfg_dropout", 0.1)
    
    # Fresh optimizer for Stage 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)
    
    # Use separate checkpoint callback for Stage 2
    stage2_callback_config = {
        "filepath": "./checkpoints_stage2.pth",
        "save_freq": config.get("save_interval", 10)
    }
    model_checkpoint = ModelCheckpoint(**stage2_callback_config)
    
    epochs = config.get("stage2_epochs", 50)
    
    for epoch in range(1, epochs + 1):
        print(f"Starting Epoch {epoch}/{epochs}")
        
        # Use conditional training with CFG
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch, cfg_dropout)
        
        print(f"Epoch {epoch} finished. Avg Loss: {loss:.6f}")
        model_checkpoint.step(
            loss,
            model=model.state_dict(),
            config=config,
            optimizer=optimizer.state_dict(),
            start_epoch=epoch,
            model_checkpoint=model_checkpoint.state_dict()
        )
    
    print("\n" + "="*60)
    print("STAGE 2 COMPLETE!")
    print(f"Final model: ./checkpoints_stage2/best.pth")
    print("="*60 + "\n")


def train_two_stage(config):
    """Run complete two-stage training pipeline"""
    print("\n" + "="*70)
    print("TWO-STAGE TRAINING PIPELINE")
    print("="*70)
    print("Stage 1: Train unconditional model (all dogs)")
    print("Stage 2: Fine-tune with breed conditioning")
    print("="*70 + "\n")
    
    # Stage 1
    stage1_checkpoint = train_stage1(config)
    
    input("\nPress Enter to continue to Stage 2...")
    
    # Stage 2
    train_stage2(config, stage1_checkpoint)
    
    print("\n" + "="*70)
    print("TWO-STAGE TRAINING COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    config = load_yaml("config.yml", encoding="utf-8")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--stage1":
            train_stage1(config)
        elif sys.argv[1] == "--stage2":
            train_stage2(config, "./checkpoint/sanford.pth")
        elif sys.argv[1] == "--two-stage":
            train_two_stage(config)
    else:
        train(config)
