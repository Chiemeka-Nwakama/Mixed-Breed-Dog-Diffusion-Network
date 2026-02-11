from dataset.DeltaDataset import create_delta_dataloader
from model.UNet import UNet
from utils.engine import GaussianDiffusionTrainer
from utils.tools import train_one_epoch, train_one_epoch_unconditional, load_yaml
import torch
from utils.callbacks import ModelCheckpoint
from datetime import datetime
import uuid
from pyspark.sql import SparkSession

def train(config):
    """Single-stage conditional training"""
    consume = config.get("consume", False)
    if consume:
        cp = torch.load(config["consume_path"])
        config = cp["config"]
    
    print(config)
    device = torch.device(config["device"])
    
    # Use delta dataloader
    loader = create_delta_dataloader(**config["Dataset"])
    
    start_epoch = 1
    model = UNet(**config["Model"]).to(device)
    
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
    
    # Gold Layer: Initialize metrics tracking
    run_id = str(uuid.uuid4())
    training_stage = "single_stage_conditional"
    metrics_data = []
    
    for epoch in range(start_epoch, config["epochs"] + 1):
        print(f"Starting Epoch {epoch}/{config['epochs']}")
        epoch_start_time = datetime.now()
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch, cfg_dropout)
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        
        print(f"Epoch {epoch} finished. Avg Loss: {loss:.6f}")
        
        # Gold Layer: Collect metrics
        metrics_data.append({
            'run_id': run_id,
            'training_stage': training_stage,
            'epoch': epoch,
            'mse_loss': float(loss),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'cfg_dropout': cfg_dropout,
            'epoch_duration_seconds': epoch_duration,
            'timestamp': epoch_end_time,
            'num_classes': config["Model"].get("num_class", None),
            'batch_size': config["Dataset"].get("batch_size", None)
        })
        
        # Gold Layer: Write metrics to Delta table every 5 epochs
        if epoch % 5 == 0 or epoch == config["epochs"]:
            try:
                spark = SparkSession.builder.getOrCreate()
                metrics_df = spark.createDataFrame(metrics_data)
                metrics_df.write.mode('append').saveAsTable('dogdiffusion.gold.ddpm_training_metrics')
                print(f"✓ Metrics logged to gold.ddpm_training_metrics (run_id: {run_id})")
                metrics_data = []  # Reset after writing
            except Exception as e:
                print(f"Warning: Could not write metrics to Gold table: {e}")
        
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
    loader = create_delta_dataloader(**config["Dataset"])
    
    lr = config.get("stage1_lr", 0.0002)
    print(f"Learning Rate: {lr}")
    model = UNet(**config["Model"]).to(device)
    cfg_dropout = config.get("cfg_dropout", 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)
    stage1_callback_config = {
        "filepath": "./checkpoints/stage1_unconditional.pth",
        "save_freq": config.get("save_interval", 20)
    }
    model_checkpoint = ModelCheckpoint(**stage1_callback_config)
    
    # Gold Layer: Initialize metrics tracking
    run_id = str(uuid.uuid4())
    training_stage = "stage1_unconditional"
    metrics_data = []
    
    epochs = config.get("stage1_epochs", 80)
    for epoch in range(1, epochs + 1):
        print(f"Starting Epoch {epoch}/{epochs}")
        epoch_start_time = datetime.now()
        loss = train_one_epoch_unconditional(trainer, loader, optimizer, device, epoch)
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        
        print(f"Epoch {epoch} finished. Avg Loss: {loss:.6f}")
        
        # Gold Layer: Collect metrics
        metrics_data.append({
            'run_id': run_id,
            'training_stage': training_stage,
            'epoch': epoch,
            'mse_loss': float(loss),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'cfg_dropout': None,  # Not used in unconditional training
            'epoch_duration_seconds': epoch_duration,
            'timestamp': epoch_end_time,
            'num_classes': None,  # Unconditional
            'batch_size': config["Dataset"].get("batch_size", None)
        })
        
        # Gold Layer: Write metrics to Delta table every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            try:
                spark = SparkSession.builder.getOrCreate()
                metrics_df = spark.createDataFrame(metrics_data)
                metrics_df.write.mode('append').saveAsTable('dogdiffusion.gold.ddpm_training_metrics')
                print(f" Metrics logged to gold.ddpm_training_metrics (run_id: {run_id})")
                metrics_data = []  # Reset after writing
            except Exception as e:
                print(f"Warning: Could not write metrics to Gold table: {e}")
        
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
    print(f"Final model: ", stage1_callback_config["filepath"])
    print("="*60 + "\n")
    return 

def train_stage2(config, stage1_checkpoint):
    """Stage 2: Conditional fine-tuning (breed-specific)"""
    print("\n" + "="*60)
    print("STAGE 2: CONDITIONAL FINE-TUNING")
    print("Adding breed-specific conditioning")
    print("="*60 + "\n")
    
    device = torch.device(config["device"])
    loader = create_delta_dataloader(**config["Dataset"])
    
    print(f"Loading Stage 1 checkpoint:  ./checkpoints/stage1_unconditional.pth")
    cp = torch.load("./checkpoints/stage1_unconditional.pth")
    model = UNet(**config["Model"]).to(device)
    model.load_state_dict(cp["model"])
    lr = config.get("stage2_lr", 0.00005)
    print(f"Learning Rate: {lr} (lower for fine-tuning)")
    cfg_dropout = config.get("cfg_dropout", 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)
    stage2_callback_config = {
        "filepath": "./checkpoints_stage2_conditional.pth",
        "save_freq": config.get("save_interval", 10)
    }
    model_checkpoint = ModelCheckpoint(**stage2_callback_config)
    
    # Gold Layer: Initialize metrics tracking
    run_id = str(uuid.uuid4())
    training_stage = "stage2_conditional_finetuning"
    metrics_data = []
    
    epochs = config.get("stage2_epochs", 50)
    for epoch in range(1, epochs + 1):
        print(f"Starting Epoch {epoch}/{epochs}")
        epoch_start_time = datetime.now()
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch, cfg_dropout)
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        
        print(f"Epoch {epoch} finished. Avg Loss: {loss:.6f}")
        
        # Gold Layer: Collect metrics
        metrics_data.append({
            'run_id': run_id,
            'training_stage': training_stage,
            'epoch': epoch,
            'mse_loss': float(loss),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'cfg_dropout': cfg_dropout,
            'epoch_duration_seconds': epoch_duration,
            'timestamp': epoch_end_time,
            'num_classes': config["Model"].get("num_class", None),
            'batch_size': config["Dataset"].get("batch_size", None)
        })
        
        # Gold Layer: Write metrics to Delta table every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            try:
                spark = SparkSession.builder.getOrCreate()
                metrics_df = spark.createDataFrame(metrics_data)
                metrics_df.write.mode('append').saveAsTable('dogdiffusion.gold.ddpm_training_metrics')
                print(f"✓ Metrics logged to gold.ddpm_training_metrics (run_id: {run_id})")
                metrics_data = []  # Reset after writing
            except Exception as e:
                print(f"Warning: Could not write metrics to Gold table: {e}")
        
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
    print(f"Final model: ", stage2_callback_config["filepath"])
    print("="*60 + "\n")
    return

def train_two_stage(config):
    """Run complete two-stage training pipeline"""
    print("\n" + "="*70)
    print("TWO-STAGE TRAINING PIPELINE")
    print("="*70)
    print("Stage 1: Train unconditional model (all dogs)")
    print("Stage 2: Fine-tune with breed conditioning")
    print("="*70 + "\n")
    stage1_checkpoint = train_stage1(config)
    input("\nPress Enter to continue to Stage 2...")
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
            train_stage2(config, "./checkpoints/stage1_unconditional.pth")
        elif sys.argv[1] == "--two-stage":
            train_two_stage(config)
    else:
        train(config)
