# Two-Stage Conditional Diffusion Model for Mixed Dog Breed Generation
**Sniffing Out the Breed: Seeing What a Computer's Nose Knows – Dog Breed Diffusion and Development**  
Chukwuemeka Ugwu, Chiemeka Nwakama, Alec Bennyhoff

---

## Overview
This project extends a PyTorch implementation of **Denoising Diffusion Implicit Models (DDIM)** to train a **two-stage dog-breed generator** on the **Stanford Dogs** dataset, using a PySpark **Delta Lake medallion (Bronze → Silver → Gold)** Azure Databricks pipeline for reproducible data preparation and efficient training.

Key contributions:
- **Data pipeline**: Medallion architecture (Bronze → Silver → Gold) backed by **Delta Lake**
- **Stage 1**: Unconditional dog generator
- **Stage 2**: Conditional fine-tuning with **classifier-free guidance**
- **Three generation modes**: unconditional, single-breed, and mixed-breed interpolation
- **Delta Lake integration** for dataset versioning, ACID writes, and auditability
- **FastAPI web UI** for browser-based image generation with live progress tracking
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

## Web App (FastAPI + Docker)

A browser-based UI for generating dog images without needing to run Python scripts directly.

### Features
- Single-breed, mixed-breed, and random generation modes
- Live progress bar synced to real diffusion step counts
- Skeleton loading cards while images generate
- Per-image download links
- GPU and CPU support

### App Preview

![Mixed Breed Dog App 1](Images/Mixed%20breed%20dog%20app%201.png)
![Mixed Breed Dog App 2](Images/Mixed%20breed%20dog%20app%202.png)
![Mixed Breed Dog App 3](Images/Mixed%20breed%20dog%20app%203.png)

---

## Docker Deployment

The model checkpoint is baked into the Docker image at build time — no volume mounts needed at runtime.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

### Project structure (relevant files)
```
DDPM/
├── Dockerfile
├── server.py          # FastAPI app + HTML UI
├── inference.py       # Model loading + image generation
├── config.yml
├── requirements.txt
├── model/
├── utils/
└── checkpoints/
    └── conditional/
        └── stage2_conditional.pth   # baked into image at build time
```

### 1. Build the image

Run from inside the `DDPM/` directory (where your `Dockerfile` lives):

```bash
docker build -t ddpm-web .
```

> The checkpoint is copied into the image during the build so you never need to mount it. Build time will be longer the first time — the `.pth` file adds to the image size.

### 2. Run the container

**With GPU (recommended):**
```bash
docker run --rm -it --name ddpm-web --gpus all -p 8081:8000 ddpm-web
```

**Without GPU (CPU only — generation will be slower):**
```bash
docker run --rm -it --name ddpm-web -p 8081:8000 ddpm-web
```

### 3. Open the UI

| URL | What it is |
|-----|-----------|
| http://localhost:8081 | Main web UI |
| http://localhost:8081/health | Health check — shows device and model status |

### Stopping the container
```bash
docker stop ddpm-web
```

### Rebuilding after code changes

If you change `server.py`, `inference.py`, or any other code file, rebuild the image:
```bash
docker build -t ddpm-web .
docker run --rm -it --name ddpm-web --gpus all -p 8081:8000 ddpm-web
```

If you only change the checkpoint, rebuild too — the `.pth` is baked in at build time.

### Troubleshooting

**`model not loaded` on the health check** — the checkpoint path inside the container is `/checkpoints/conditional/stage2_conditional.pth`. Make sure your `Dockerfile` has:
```dockerfile
COPY checkpoints/conditional/stage2_conditional.pth /checkpoints/conditional/stage2_conditional.pth
```
and that the file exists at that path relative to your `Dockerfile`.

**`--gpus all` errors** — you either don't have an NVIDIA GPU or the NVIDIA Container Toolkit isn't set up. Drop `--gpus all` to run on CPU.

**Port already in use** — change `-p 8081:8000` to another port, e.g. `-p 8082:8000`, and visit http://localhost:8082.

---

## Dataset
**Stanford Dogs Dataset**
- Breeds: 120
- Images: ~20,580
- Annotations: XML bounding boxes

Dataset path (raw images):
- `data/Training Images (Stanford)`

Supported formats: `.jpg`, `.png`

Training preprocessing note:
- Images are converted to **RGB** and resized to **128×128** for training (after Silver preprocessing and/or at Gold load time).

Preprocessing notebook: `Bronze To Silver Layer.py`

---

## Data Architecture — Medallion (Bronze → Silver → Gold)

We implement a three-layer medallion architecture using Delta Lake to keep data transformations **traceable, reproducible, and safe**.

### Bronze (Raw)
**Purpose:** Store the raw dataset exactly as provided — no edits.

```
DDPM/data/
├── Training Images (Stanford)/
└── Annotation/
```

### Silver (Cleaned + Validated)
**Purpose:** Clean, validate, and standardize the dataset so training is consistent.

What we do:
- Filter to valid image extensions
- Parse breed label from folder structure
- Parse XML to extract bounding boxes
- Crop to bbox when available
- Resize to **64×64** and convert to **RGB**
- Encode as PNG bytes for consistent downstream loading
- Generate stable **breed → integer** label mapping (0–119)
- Skip corrupted/unreadable files

Output schema (example):
```
root
 |-- path: string
 |-- cls: string
 |-- label: integer
 |-- img_bytes_prepared: binary
 |-- width: integer
 |-- height: integer
```

Stored as a Delta table.

### Gold (Training-Ready)
**Purpose:** Optimized access pattern for PyTorch training/inference.

What we do at training time:
- Read from the Delta table efficiently (batched, column-pruned reads)
- Resize to training size (e.g., **128×128**)
- Normalize to model input range
- Feed to PyTorch `DataLoader` via `DeltaDataset`
- Creates gold metric Delta table: **run_id, training_stage, epoch, mse_loss, learning_rate, cfg_dropout, epoch_duration_seconds, timestamp, num_classes, batch_size**

---

## Why Delta Lake?
- **ACID transactions:** prevents partial/corrupt writes during preprocessing
- **Time travel (versioning):** reproduce training runs against the exact dataset version used
- **Audit history:** track when/why data changed
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
├── Dockerfile
├── server.py              # FastAPI web app
├── inference.py           # Model inference + progress tracking
├── train.py
├── generate.py
├── requirements.txt
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
└── data/
    ├── Training Images (Stanford)/
    ├── Annotation/
    └── silver_delta/
```

---

## Breed Index Mapping
Breed IDs range **0–119**. The mapping is generated during preprocessing and stored in the dataset table.

| ID | Breed |
|----|-------|
| 0 | Chihuahua |
| 56 | Golden Retriever |
| 57 | Labrador Retriever |
| 84 | German Shepherd |
| 99 | Siberian Husky |
| 119 | African Hunting Dog |

Full mapping available in `Class Index - Breed Name Mapping` or in `server.py` as `BREED_MAP`.

---

## Setup (without Docker)

### Prerequisites
```bash
pip install torch torchvision pyspark delta-spark pillow pyyaml fastapi uvicorn
```

### Configuration (`config.yml`)
```yaml
device: cuda:0
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

### Two-Stage Training (Recommended)

**Stage 1 — Unconditional**
```bash
python train.py --stage1
```
Learns general dog anatomy, fur textures, and the broad image distribution.  
Checkpoint saved to: `./checkpoints/unconditional/stage1_unconditional.pth`

**Stage 2 — Conditional Fine-Tuning**
```bash
python train.py --stage2
```
Loads from Stage 1 and learns breed-specific features, CFG conditioning, and interpolation behavior.  
Checkpoint saved to: `./checkpoints/conditional/stage2_conditional.pth`

| Parameter | Default |
|-----------|---------|
| stage2_epochs | 80 |
| stage2_lr | 0.00005 |
| guidance_scale | 3.0 |

**Full Two-Stage Pipeline**
```bash
python train.py --two-stage
```

---

## Sampling / Image Generation

### CLI
```bash
# Unconditional
python generate.py --unconditional -cp stage1_unconditional.pth

# Single-breed
python generate.py --single --class_1 5 -cp stage2_conditional.pth

# Mixed-breed
python generate.py --mixed --class_1 3 --class_2 10 --mix_ratio 0.6 -cp stage2_conditional.pth
```

### Web UI (Docker)
See [Docker Deployment](#docker-deployment) above.

---

## Mixed-Breed Generation (Technical Details)

Mixed-breed generation blends two conditional predictions throughout sampling using a time-dependent mixing weight.

**Sigmoid mixing weight:**
```python
w(t) = 1 / (1 + exp(-10 * (t/T - 0.5)))
```

**Latent mixing:**
```python
x_t = (1 - w(t)) * xA_t + w(t) * xB_t
```

Result:
- Structural features come from early-weighted breed
- Texture/coloration comes from later-weighted breed
- Produces realistic mixed-breed dogs

---

## Checkpoints
- Stage 1: `./checkpoints/unconditional/stage1_unconditional.pth`
- Stage 2: `./checkpoints/conditional/stage2_conditional.pth`

Saved contents typically include: model state dict, optimizer state, epoch, loss history, config snapshot.

---

## Credits & Acknowledgments
- **Base implementation:** Alokia/diffusion-DDIM-pytorch
- **Dataset:** Stanford Dogs (120 breeds)
- **Tools:** Azure, Databricks, Delta Lake, PySpark, PyTorch, FastAPI, Matplotlib, NumPy, torchvision, Pillow, PyYAML
- **Extensions:** Medallion pipeline, Delta Lake integration, two-stage training, CFG conditioning, mixed-breed interpolation, Delta-backed dataset loader, FastAPI web UI with live progress tracking
- **Team:** Chukwuemeka Ugwu, Chiemeka Nwakama, Alec Bennyhoff

---

## Future Enhancements
- Add incremental Delta updates and data quality monitoring
- Higher resolution support (256×256, 512×512)
- Multi-user support for the web UI

---

## License
This project extends the original DDIM implementation. Refer to the base repository for licensing terms.
