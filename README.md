# Two-Stage Conditional Diffusion Model for Mixed Dog Breed Generation
CSCI 5561 / Computer Vision — Final Project - Sniffing Out the Breed: Seeing What a Computer’s Nose Knows – Dog Breed Diffusion and Development
Chukwuemeka Ugwu, Chiemeka Nwakama, Alec Bennyhoff — University of Minnesota

---

## Overview
This project extends an existing PyTorch implementation of Denoising Diffusion Implicit Models (DDIM) to train a two-stage dog-breed generator using the Stanford Dogs dataset.  
Our contributions include:

- Unconditional dog generator (Stage 1)
- Conditional fine-tuning with classifier-free guidance (Stage 2)
- Support for three generation modes: unconditional (general dog), single-breed, and mixed-breed interpolation
- Expanded configuration system (config.yml)
- Updated training pipeline and dataset loader
- Utilities for sampling and checkpointing

The resulting model can generate specific breeds or interpolate between two breeds at arbitrary ratios.

---

## Base Code Acknowledgment
This project builds directly on the PyTorch DDIM implementation by Zhao Di (Alokia):
- Repository: https://github.com/Alokia/diffusion-DDIM-pytorch

This includes the original UNet backbone, diffusion process, DDPM/DDIM samplers, and core training/sampling structure.  
Our project adapts these components and extends them for conditional breed modeling and breed mixture for the Standford Dog Breed Dataset.

The original DDIM paper:  
Denoising Diffusion Implicit Models (2020) — https://arxiv.org/abs/2010.02502

---

## Dataset
We use the Stanford Dogs Dataset, containing 120 breeds with high-resolution, color images.

Dataset path:
```
data/Training Images (Stanford)
```

Supported formats: `.png`, `.jpg`  
Images are automatically resized to **128×128 RGB**.

---

## Breed Index Mapping
The model uses breed IDs (0–119).  
Full mapping should be included in the repository as `breed_index.txt` or a README section.

---

# Training Pipeline

## 1. Single-Stage Training (Optional)
Runs combined conditional training.

```
python train.py
```

Default config:
```
epochs: 80
lr: 0.0001
cfg_dropout: 0.1
```

---

## 2. Two-Stage Training (Recommended)

### Stage 1 — Unconditional Training
Learns general dog structure from all images without labels.

```
python train.py --stage1
```

Defaults:
| Parameter | Value |
|----------|-------|
| stage1_epochs | 200 |
| stage1_lr | 0.0001 |

Checkpoint saved to:
```
./checkpoints/unconditional/stage1_unconditional.pth
```

---

### Stage 2 — Conditional Fine-Tuning
Fine-tunes Stage 1 with class labels and classifier-free guidance.

```
python train.py --stage2
```

---
Loads from previously uncondtional model from: ./checkpoints/unconditional/stage1_unconditional.pth
```

Defaults:
| Parameter | Value |
|----------|-------|
| stage2_epochs | 80 |
| stage2_lr | 0.00005 |
| guidance_scale | 3.0 |

---

Checkpoint saved to:
```
./checkpoints/conditional/stage2_conditional.pth
```


### Full Two-Stage Pipeline (combined stage 1 and stage 2)
```
python train.py --two-stage
```

---

# Sampling / Image Generation

## 1. Unconditional Generation
```
python generate.py --unconditional -cp stage1_unconditional.pth
```

## 2. Single-Breed Generation
```
python generate.py --single --class_1 5 -cp stage2_conditional.pth
```

## 3. Mixed-Breed Generation
```
python generate.py --mixed --class_1 3 --class_2 10 --mix_ratio 0.6 -cp stage2_conditional.pth
```

---

# Mixed-Breed Generation (Detailed Explanation)

Mixed-breed generation combines two breeds using latent-space interpolation throughout the diffusion process.  
This allows smooth blending between two dog breeds.

---

## Sigmoid-Based Mixing Function

A time-dependent sigmoid weight controls how much each breed contributes:

```
w(t) = 1 / (1 + exp(-10 * (t/T - 0.5)))
```

Where:
- t = current timestep
- T = total sampling steps

Meaning:
- Early steps favor Breed A
- Later steps favor Breed B
- Midpoint transitions smoothly

---

## Latent Mixing Equation
```
x_t = (1 - w(t)) * xA_t + w(t) * xB_t
```

Where:
- xA_t = UNet prediction using Breed A conditioning
- xB_t = prediction using Breed B conditioning

---

## Result
- Structural features come from early-weighted breed
- Texture/coloration come from later-weighted breed
- Produces realistic mixed-breed dogs

---

# Configuration File (config.yml)

Example:
```
device: cuda:0
epochs: 80
lr: 0.0001
cfg_dropout: 0.1

stage1_epochs: 200
stage1_lr: 0.0001
stage2_epochs: 80
stage2_lr: 0.00005

guidance_scale: 3.0
sample_interval: 10
save_interval: 3
```

---

## Model Configuration
```
Model:
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

You can modify:
- model_channels (capacity)
- channel_mult (depth)
- num_class (dataset size)
- dropout (regularization)
- attention layers for specific resolutions

---

# Checkpoint Locations
Stage 1:
```
./checkpoints/unconditional/stage1_unconditional.pth
```
Stage 2:
```
./checkpoints/conditional/stage2_conditional.pth
```

---

# Credits
This project is a collaboration of:
- Alokia/diffusion-DDIM-pytorch
- Original DDPM/DDIM sampling logic
- Stanford Dogs dataset

Extended for conditional and mixed-breed modeling.

