# Two-Stage Conditional Diffusion Model for Mixed Dog Breed Generation
**Sniffing Out the Breed: Seeing What a Computer's Nose Knows – Dog Breed Diffusion and Development**  
Chukwuemeka Ugwu, Chiemeka Nwakama, Alec Bennyhoff

---

## Overview
This project extends a PyTorch implementation of **Denoising Diffusion Implicit Models (DDIM)** to train a **two-stage dog-breed generator** on the **Stanford Dogs** dataset, using a Pyspark **Delta Lake medallion (Bronze → Silver → Gold)** Azure Databricks pipeline for reproducible data preparation and efficient training.

Key contributions:
- **Data pipeline**: Medallion architecture (Bronze → Silver → Gold) backed by **Delta Lake**
- **Stage 1**: Unconditional dog generator
- **Stage 2**: Conditional fine-tuning with **classifier-free guidance**
- **Three generation modes**: unconditional, single-breed, and mixed-breed interpolation
- **Delta Lake integration** for dataset versioning, ACID writes, and auditability
- Expanded configuration system (`config.yml`)
- Updated training pipeline with `DeltaDataset` loader
- Sampling/checkpointing utilities

The resulting model can generate specific breeds or interpolate between two breeds at arbitrary ratios.

---

## Base Code Acknowledgment
This project builds on the PyTorch DDIM implementation by **Zhao Di (Alokia)**:  
- Repo: https://github.com/Alokia/diffusion-DDIM-pytorch  
Includes the UNet backbone, diffusion process, DDPM/DDIM samplers, and the original training/sampling structure.

Original paper: **Denoising Diffusion Implicit Models (2020)** — https://arxiv.org/abs/2010.02502

---

## Dataset
**Stanford Dogs Dataset**
- Breeds: 120
- Images: ~20,580
- Annotations: XML bounding boxes

Dataset path (raw images):
- `data/Training Images (Stanford)`

Supported formats:
- `.jpg`, `.png`

Training preprocessing note:
- Images are converted to **RGB** and resized to **128×128** for training (after Silver preprocessing and/or at Gold load time).

Preprocessing notebook: `notebooks/data_preparation.ipynb`

---

## Data Architecture Medallion Architecture (Bronze → Silver → Gold)

We implement a three-layer medallion architecture using Delta Lake to keep data transformations **traceable, reproducible, and safe**.

### Bronze (Raw)
**Purpose:** Store the raw dataset exactly as provided.  
**What’s in it:** original images + XML annotations (no edits).  
**Location (example):**
```
DDPM/data/
├── Training Images (Stanford)/
└── Annotation/
```

### Silver (Cleaned + Validated)
**Purpose:** Clean, validate, and standardize the dataset so training is consistent.  
**What we do:**
- Filter to valid image extensions
- Parse breed label from folder structure
- Parse XML to extract bounding boxes
- Crop to bbox when available
- Resize to **64×64** and convert to **RGB**
- Encode as PNG bytes for consistent downstream loading
- Generate stable **breed → integer** label mapping (0–119)
- Skip corrupted/unreadable files

**Output schema (example):**
```
root
 |-- path: string
 |-- cls: string
 |-- label: integer
 |-- img_bytes_prepared: binary
 |-- width: integer
 |-- height: integer
```

**Stored as a Delta table** (see below).

### Gold (Training-Ready)
**Purpose:** Optimized access pattern for PyTorch training/inference.  
**What we do at training time:**
- Read from the Delta table efficiently (batched, column-pruned reads)
- Resize to training size (e.g., **128×128**)
- Normalize to model input range
- Feed to PyTorch `DataLoader` via `DeltaDataset`
- Creates gold metric Delta table: **run_id, training_stage, epoch, mse_loss, learning_rate, cfg_dropout, epoch_duration_seconds, timestamp, num_classes, batch_size**

---

## Why a Delta Table?
A Delta table is used for the Silver/Gold dataset because it provides features that are helpful for an ML pipeline:
- **ACID transactions:** prevents partial/corrupt writes during preprocessing
- **Time travel (versioning):** lets you reproduce training runs against the exact dataset version used
- **Audit history:** track when/why data changed (useful for debugging)
- **Schema evolution:** safely add metadata columns without breaking existing readers
- **Concurrent read/write safety:** stable reads while new versions are being written

Delta table example location:
```
/Volumes/dogdiffusion/default/diffusion_data/silver_delta/
├── _delta_log/
└── *.parquet
```

---

## Repository Structure
```
DDPM/
├── README.md
├── config.yml
├── train.py
├── generate.py
├── dataset/
│   └── DeltaDataset.py
├── model/
│   ├── __init__.py
│   └── UNet.py
├── utils/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── engine.py
│   └── tools.py
├── checkpoints/
│   ├── unconditional/
│   │   └── stage1_unconditional.pth
│   └── conditional/
│       └── stage2_conditional.pth
└── data/   # Bronze and Silver layer data
    ├── Training Images (Stanford)/
    └── Annotation/
    └── silver_delta/
```

---

## Breed Index Mapping
Breed IDs range **0–119**. The mapping is generated during preprocessing and stored in the dataset table.

Example:
- 0: Chihuahua  
- 45: Golden Retriever  
- 67: Poodle  
- 119: Brabancon Griffon  

Full mapping:
- Include as `breed_index.txt` (recommended) or extend this README with the complete list.

---

## Setup

### Prerequisites
```bash
pip install torch torchvision pyspark delta-spark pillow pyyaml
```

### Configuration (`config.yml`)
Example fields:
```yaml
device: cuda:0

# Single-stage (optional)
epochs: 80
lr: 0.0001
cfg_dropout: 0.1

# Two-stage (recommended)
stage1_epochs: 200
stage1_lr: 0.0001
stage2_epochs: 80
stage2_lr: 0.00005

guidance_scale: 3.0
sample_interval: 10
save_interval: 3

data_path: "/Volumes/dogdiffusion/default/diffusion_data/silver_delta"
use_delta: true
batch_size: 16
image_size: 128
mode: "RGB"
shuffle: true
drop_last: true
pin_memory: true
num_workers: 2

# Model
in_channels: 3
model_channels: 64
out_channels: 3
num_res_blocks: 2
num_class: 120
attention_resolutions: [8, 16]
dropout: 0.1
channel_mult: [1, 2, 2, 2]
conv_resample: true
num_heads: 4
```

---

## Training

### 1) Single-Stage Training (Optional)
Runs combined conditional training.

```bash
python train.py
```

Default config:
- `epochs`: 80  
- `lr`: 0.0001  
- `cfg_dropout`: 0.1  

### 2) Two-Stage Training (Recommended)

**Stage 1 — Unconditional**
```bash
python train.py --stage1
```

Defaults:
- `stage1_epochs`: 200  
- `stage1_lr`: 0.0001  

Checkpoint saved to:
`./checkpoints/unconditional/stage1_unconditional.pth`

Learns:
- general dog anatomy/pose
- fur textures/patterns
- broad image distribution

**Stage 2 — Conditional Fine-Tuning**
```bash
python train.py --stage2
```

Loads from previously unconditional model:
`./checkpoints/unconditional/stage1_unconditional.pth`

Defaults:
| Parameter | Value |
|----------|-------|
| stage2_epochs | 80 |
| stage2_lr | 0.00005 |
| guidance_scale | 3.0 |

Checkpoint saved to:
`./checkpoints/conditional/stage2_conditional.pth`

Learns:
- breed-specific features
- conditional control + classifier-free guidance
- interpolation behavior

**Full Two-Stage Pipeline**
```bash
python train.py --two-stage
```

---

## Sampling / Image Generation

### 1) Unconditional
```bash
python generate.py --unconditional -cp stage1_unconditional.pth
```

### 2) Single-Breed
```bash
python generate.py --single --class_1 5 -cp stage2_conditional.pth
```

### 3) Mixed-Breed
```bash
python generate.py --mixed --class_1 3 --class_2 10 --mix_ratio 0.6 -cp stage2_conditional.pth
```

---

## Mixed-Breed Generation (Technical Details)
Mixed-breed generation blends two conditional predictions **throughout** sampling using a time-dependent mixing weight.

**Sigmoid mixing weight**
```python
w(t) = 1 / (1 + exp(-10 * (t/T - 0.5)))
```

Where:
- `t` = current timestep  
- `T` = total sampling steps  

Meaning:
- Early steps favor Breed A
- Later steps favor Breed B
- Midpoint transitions smoothly

**Latent mixing**
```python
x_t = (1 - w(t)) * xA_t + w(t) * xB_t
```

Where:
- `xA_t` = UNet prediction using Breed A conditioning  
- `xB_t` = UNet prediction using Breed B conditioning  

Result:
- Structural features come from early-weighted breed
- Texture/coloration come from later-weighted breed
- Produces realistic mixed-breed dogs

---

## Checkpoints
- Stage 1: `./checkpoints/unconditional/stage1_unconditional.pth`
- Stage 2: `./checkpoints/conditional/stage2_conditional.pth`

Saved contents typically include:
- model state dict
- optimizer state
- epoch
- loss history
- config snapshot

---

## Credits & Acknowledgments
- **Base implementation:** Alokia/diffusion-DDIM-pytorch  
- **Dataset:** Stanford Dogs (120 breeds)
- **Tools:** Azure, Databricks, Delta Lake, PySpark, PyTorch, Matplotlib, Numpy, torchvision, Pillow, PyYAML
- **Extensions:** Medallion pipeline (Bronze/Silver/Gold), Delta Lake integration, two-stage training, CFG conditioning, mixed-breed interpolation, Delta-backed dataset loader  
- **Team:** Chukwuemeka Ugwu, Chiemeka Nwakama, Alec Bennyhoff

---

## Future Enhancements
- Add incremental Delta updates and data quality monitoring
- Higher resolution support (256×256, 512×512)

---

## License
This project extends the original DDIM implementation. Refer to the base repository for licensing terms.
