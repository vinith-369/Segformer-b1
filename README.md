# SegFormer-B1 — Cityscapes Semantic Segmentation

> A **self-contained, pure PyTorch** reimplementation of [SegFormer](https://arxiv.org/abs/2105.15203) (NVlabs, NeurIPS 2021) targeting the Cityscapes benchmark.  
> **Zero** `mmcv` / `mmseg` dependencies — only PyTorch, `timm`, and standard scientific Python.

---

## Table of Contents

- [Highlights](#highlights)
- [Project Structure](#project-structure)
- [Architecture — Base Model](#architecture--base-model)
  - [Mix Vision Transformer (MiT-B1) Encoder](#mix-vision-transformer-mit-b1-encoder)
  - [All-MLP Decoder Head](#all-mlp-decoder-head)
  - [Full SegFormer Model](#full-segformer-model)
- [Training](#training)
  - [Prerequisites](#prerequisites)
  - [Dataset Setup (Cityscapes)](#dataset-setup-cityscapes)
  - [Configuration](#configuration)
  - [Launch Training](#launch-training)
  - [Training Details](#training-details)
- [Inference / Evaluation](#inference--evaluation)
- [Pipeline — End-to-End Flow](#pipeline--end-to-end-flow)
- [Requirements](#requirements)

---

## Highlights

| Feature | Details |
|---|---|
| **Backbone** | Mix Vision Transformer B1 (MiT-B1) with Efficient Self-Attention |
| **Decoder** | Lightweight All-MLP head (no heavy FPN / ASPP) |
| **Dependencies** | Pure PyTorch + `timm` — no `mmcv`, `mmseg`, or OpenMMLab toolbox |
| **Multi-device** | Auto-detects CUDA, Apple MPS, or CPU; supports multi-GPU via DDP |
| **Augmentations** | Random scaling, cropping, flipping, PhotoMetric distortion — all NumPy/OpenCV |
| **LR Schedule** | Poly-warmup AdamW (follows official SegFormer recipe) |
| **Metrics** | mIoU, pixel accuracy, per-class IoU |

---

## Project Structure

```
segformer_b1_cityscapes/
│
├── config.yaml                  # All hyperparameters in one place
│
├── model/                       # Neural network architecture
│   ├── __init__.py              # Exports SegFormer class
│   ├── mix_transformer.py       # MiT-B1 encoder (from official NVlabs, mmcv-free)
│   ├── segformer_head.py        # All-MLP decoder head (pure PyTorch)
│   └── segformer.py             # Full model = encoder + decoder + weight loading
│
├── data/                        # Dataset loading & augmentation
│   ├── __init__.py              # Exports CityscapesSegDataset
│   ├── cityscapes.py            # Cityscapes dataset class + 34→19 label mapping
│   └── transforms.py            # Augmentations: scale, crop, flip, color jitter
│
├── utils/                       # Training utilities
│   ├── __init__.py              # Exports metrics, optimizer, logger
│   ├── metrics.py               # mIoU, pixel accuracy via confusion matrix
│   ├── optimizer.py             # PolyWarmupAdamW LR scheduler
│   └── logger.py                # Dual file + console logging
│
├── train.py                     # Training entry point (single GPU / DDP)
├── test.py                      # Standalone evaluation script
└── README.md
```

### Runtime Directories (auto-created)

| Directory | Purpose |
|---|---|
| `logs/` | Timestamped `.log` files with full training output |
| `checkpoints/` | Model weights (`best_model.pth`, `latest_model.pth`, periodic saves) |
| `pretrained/` | Place ImageNet pretrained `mit_b1.pth` here before training |

---

## Architecture — Base Model

SegFormer is a Transformer-based semantic segmentation model that combines a **hierarchical Vision Transformer encoder** with a **lightweight MLP decoder**. It avoids positional encodings entirely (making it resolution-agnostic) and uses efficient self-attention to keep computation manageable at high resolutions.

### Mix Vision Transformer (MiT-B1) Encoder

**File:** `model/mix_transformer.py`

The encoder is a 4-stage hierarchical transformer that produces multi-scale feature maps at resolutions **1/4, 1/8, 1/16, and 1/32** of the input.

#### Stage Architecture

Each of the 4 stages consists of:

1. **Overlapping Patch Embedding** (`OverlapPatchEmbed`)  
   Uses a strided convolution (kernel 7×7 for stage 1, 3×3 for stages 2–4) to extract patch tokens with overlap, preserving local continuity — unlike ViT's non-overlapping patches.

2. **Transformer Blocks** (`Block`)  
   Each block follows the pattern:
   ```
   x → LayerNorm → Efficient Self-Attention → + residual
     → LayerNorm → Mix-FFN (MLP + DWConv)    → + residual
   ```

3. **Efficient Self-Attention** (`Attention`)  
   Applies **Spatial Reduction (SR)** to keys and values — a strided convolution shrinks the spatial dimensions before computing attention, reducing complexity from O(N²) to O(N²/R²) where R is the SR ratio.

4. **Mix-FFN** (`Mlp`)  
   A feed-forward network with a **3×3 depth-wise convolution** between the two linear layers. This injects positional information implicitly, eliminating the need for explicit positional encodings.

#### MiT-B1 Configuration

| Stage | Embed Dim | Heads | Depth | SR Ratio | Output Resolution |
|-------|-----------|-------|-------|----------|-------------------|
| 1 | 64 | 1 | 2 | 8 | H/4 × W/4 |
| 2 | 128 | 2 | 2 | 4 | H/8 × W/8 |
| 3 | 320 | 5 | 2 | 2 | H/16 × W/16 |
| 4 | 512 | 8 | 2 | 1 | H/32 × W/32 |

> MiT-B1 has **depths [2, 2, 2, 2]** — a lightweight variant suitable for balancing speed and accuracy.

### All-MLP Decoder Head

**File:** `model/segformer_head.py`

The decoder is intentionally simple — it replaces heavy decoders like FPN or ASPP with pure MLP layers:

```
Stage Outputs (C1–C4)
   │
   ├── C4 (H/32) → MLP → Upsample to H/4
   ├── C3 (H/16) → MLP → Upsample to H/4
   ├── C2 (H/8)  → MLP → Upsample to H/4
   └── C1 (H/4)  → MLP (already at H/4)
   │
   ▼
   Concatenate along channel dimension (4 × embedding_dim)
   │
   ▼
   Conv1×1 + BatchNorm + ReLU  (fuse to embedding_dim)
   │
   ▼
   Dropout → Conv1×1  (predict num_classes)
   │
   ▼
   Output logits at H/4 × W/4 resolution
```

Each **MLP** is a linear projection: `flatten spatial → Linear(C_in, embedding_dim) → reshape back`. All four scales are projected to a common `embedding_dim=256`, upsampled to the finest resolution (H/4 × W/4), concatenated, and fused with a single convolution.

### Full SegFormer Model

**File:** `model/segformer.py`

The `SegFormer` class wires the encoder and decoder together:

```python
class SegFormer(nn.Module):
    def forward(self, x):           # x: [B, 3, H, W]
        features = self.encoder(x)  # List of 4 multi-scale feature maps
        out = self.decoder(features) # Logits [B, num_classes, H/4, W/4]
        return out
```

**Key features:**
- **Pretrained weight loading** with flexible key handling — automatically strips prefixes (`backbone.`, `encoder.`) and removes classification heads from community checkpoints
- **Differential learning rates** via `get_param_groups()`:
  - Encoder non-norm params → base LR
  - Encoder norm params → base LR, **no weight decay**
  - Decoder params → **10× base LR** (trains faster since initialized from scratch)

---

## Training

### Prerequisites

1. **Python 3.8+** with PyTorch ≥ 1.12
2. **Cityscapes dataset** — requires registration at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)
3. **Pretrained MiT-B1 weights** (ImageNet) — download from (https://github.com/anibali/segformer/releases/tag/v0.0.0)

### Dataset Setup (Cityscapes)

Organize the Cityscapes data in the standard directory structure:

```
cityscapes/
├── leftImg8bit/
│   ├── train/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       └── ...
└── gtFine/
    ├── train/
    │   ├── aachen/
    │   │   ├── aachen_000000_000019_gtFine_labelIds.png
    │   │   ├── aachen_000000_000019_gtFine_labelTrainIds.png  (preferred)
    │   │   └── ...
    │   └── ...
    └── val/
        └── ...
```

> The dataset loader automatically prefers `_labelTrainIds.png` files if available. If only `_labelIds.png` exists, the raw 34-class labels are mapped to 19 train classes on-the-fly using a precomputed LUT.

### Configuration

All hyperparameters are centralized in **`config.yaml`**:

```yaml
exp:
  backbone: mit_b1
  embedding_dim: 256           # Decoder embedding dimension
  pretrained: true
  pretrained_path: pretrained/mit_b1.pth

dataset:
  name: cityscapes
  root_dir: /path/to/cityscapes   # ← Update this path
  num_classes: 19
  crop_size: 1024
  resize_range: [512, 2048]
  rescale_range: [0.5, 2.0]
  ignore_index: 255

train:
  batch_size: 2
  max_iters: 160000
  eval_iters: 10000            # Validate every N iterations
  log_iters: 50                # Log loss/LR every N iterations
  save_iters: 10000            # Save checkpoint every N iterations
  seed: 42

optimizer:
  type: AdamW
  learning_rate: 6e-5
  betas: [0.9, 0.999]
  weight_decay: 0.01

scheduler:
  warmup_iter: 1500            # Linear warmup iterations
  warmup_ratio: 1e-6           # Starting LR ratio
  power: 1.0                   # Poly decay power
```

### Launch Training

```bash
# Single device (auto-detects CUDA / MPS / CPU)
python train.py --config config.yaml

# Multi-GPU with DistributedDataParallel
torchrun --nproc_per_node=4 train.py --config config.yaml
```

### Training Details

#### Data Augmentation Pipeline (training only)

| Augmentation | Description |
|---|---|
| **Random Scaling** | Resize image by a random ratio in `[0.5, 2.0]` |
| **Random Horizontal Flip** | 50% probability |
| **PhotoMetric Distortion** | Random brightness, contrast, saturation, and hue jitter |
| **Random Crop** | Crop to `1024×1024` with padding; uses `cat_max_ratio=0.75` to ensure class diversity |
| **ImageNet Normalize** | `mean=[123.675, 116.28, 103.53]`, `std=[58.395, 57.12, 57.375]` |

#### Optimizer & LR Schedule

- **Optimizer:** AdamW with weight decay `0.01` (`β₁=0.9, β₂=0.999`)
- **Schedule:** Linear warmup (1500 iters, start ratio `1e-6`) → polynomial decay (`power=1.0`, linear)
- **Differential LR:** Encoder at `6e-5`, decoder at `6e-4` (10×), norm layers with zero weight decay

#### Loss Function

**Cross-Entropy Loss** with `ignore_index=255` — pixels labeled 255 (unlabeled/void) are excluded from the loss computation. Model output at 1/4 resolution is bilinearly upsampled to match the label resolution before computing loss.

#### Checkpointing

The training script saves multiple checkpoint types:

| File | Contents | Use Case |
|---|---|---|
| `checkpoint_iter_N.pth` | Full state (model + optimizer + iteration + val score) | Resume training |
| `model_iter_N.pth` | Model weights only | Inference |
| `latest_checkpoint.pth` | Most recent full state | Quick resume |
| `latest_model.pth` | Most recent weights | Quick inference |
| `best_checkpoint.pth` | Best mIoU full state | Best model for fine-tuning |
| `best_model.pth` | Best mIoU weights | Best model for deployment |

#### Logging

All training output is logged to both **console** and a **timestamped log file** under `logs/`. Log entries include:
- Current iteration, loss, learning rate
- Elapsed time and ETA
- Validation results: pixel accuracy, mean accuracy, mIoU, and per-class IoU for all 19 Cityscapes classes

---

## Inference / Evaluation

Run standalone evaluation on the validation set:

```bash
# Using model weights only
python test.py --config config.yaml --checkpoint checkpoints/best_model.pth

# Using full checkpoint (also works — model weights are extracted automatically)
python test.py --config config.yaml --checkpoint checkpoints/best_checkpoint.pth
```

The evaluation script:

1. Loads the model and checkpoint (handles both full checkpoints and weight-only files)
2. Iterates over the entire Cityscapes `val` split (500 images)
3. For each image:
   - Forward pass → logits at 1/4 resolution
   - Bilinear upsample to original label resolution
   - `argmax` to get per-pixel predictions
4. Computes and reports:
   - **Pixel Accuracy** — overall percentage of correctly classified pixels
   - **Mean Accuracy** — average per-class accuracy
   - **Mean IoU (mIoU)** — the primary segmentation metric
   - **Per-class IoU** — IoU for each of the 19 Cityscapes classes

### Cityscapes Classes (19)

| ID | Class | ID | Class | ID | Class |
|---|---|---|---|---|---|
| 0 | road | 7 | traffic sign | 14 | motorcycle |
| 1 | sidewalk | 8 | vegetation | 15 | bus |
| 2 | building | 9 | terrain | 16 | train |
| 3 | wall | 10 | sky | 17 | motorcycle |
| 4 | fence | 11 | person | 18 | bicycle |
| 5 | pole | 12 | rider | — | — |
| 6 | traffic light | 13 | car | — | — |

---

## Pipeline — End-to-End Flow

The complete training and evaluation pipeline follows this flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        config.yaml                                  │
│  (backbone, dataset paths, hyperparams, LR schedule, logging)       │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     1. DATA LOADING                                 │
│                                                                     │
│  cityscapes.py                                                      │
│    ├── Scan leftImg8bit/{split}/ for image paths                    │
│    ├── Pair with gtFine/ labels (prefer _labelTrainIds.png)         │
│    └── Map raw 34 label IDs → 19 train IDs via LUT (if needed)     │
│                                                                     │
│  transforms.py                                                      │
│    ├── Random scale (0.5–2.0×)                                      │
│    ├── Random horizontal flip                                       │
│    ├── PhotoMetric distortion (brightness, contrast, sat, hue)      │
│    ├── Random crop (1024×1024) with cat_max_ratio                   │
│    └── ImageNet normalize + HWC → CHW                               │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     2. MODEL FORWARD PASS                           │
│                                                                     │
│  Input image [B, 3, 1024, 1024]                                     │
│       │                                                             │
│       ▼                                                             │
│  MiT-B1 Encoder (4 stages)                                         │
│       ├── Stage 1 → [B, 64,  256, 256]   (1/4)                     │
│       ├── Stage 2 → [B, 128, 128, 128]   (1/8)                     │
│       ├── Stage 3 → [B, 320, 64,  64]    (1/16)                    │
│       └── Stage 4 → [B, 512, 32,  32]    (1/32)                    │
│       │                                                             │
│       ▼                                                             │
│  All-MLP Decoder                                                    │
│       ├── 4× MLP projection to 256-d                                │
│       ├── Upsample all to 256×256 (1/4)                             │
│       ├── Concatenate (1024-d) → Fuse Conv1×1 (256-d)              │
│       └── Dropout → Predict Conv1×1 (19 classes)                    │
│       │                                                             │
│       ▼                                                             │
│  Output logits [B, 19, 256, 256]                                    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     3. OPTIMIZATION                                 │
│                                                                     │
│  ├── Bilinear upsample logits to label resolution                   │
│  ├── CrossEntropyLoss (ignore_index=255)                            │
│  ├── Backpropagation                                                │
│  └── PolyWarmupAdamW step                                           │
│       ├── Warmup: 0 → 1500 iters (linear ramp from 1e-6 ratio)     │
│       └── Poly decay: LR *= (1 - iter/max_iter)^1.0                │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     4. VALIDATION & CHECKPOINTING                   │
│                                                                     │
│  Every 10,000 iterations:                                           │
│    ├── Run full val set (500 images, no augmentation)               │
│    ├── Compute mIoU, pixel acc, per-class IoU                       │
│    ├── Save checkpoint (model + optimizer state)                    │
│    └── Track & save best mIoU model                                 │
│                                                                     │
│  Logging: console + logs/segformer_b1_YYYYMMDD_HHMMSS.log          │
└─────────────────────────────────────────────────────────────────────┘
```

### Quick Start Summary

```bash
# 1. Install dependencies
pip install torch timm omegaconf tqdm pillow numpy opencv-python

# 2. Download pretrained MiT-B1 weights (ImageNet)
mkdir -p pretrained/
# Place mit_b1.pth in pretrained/

# 3. Update dataset path in config.yaml
#    dataset.root_dir → /path/to/cityscapes

# 4. Train
python train.py --config config.yaml

# 5. Evaluate
python test.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

---

## Requirements

```
torch >= 1.12
timm
omegaconf
tqdm
Pillow
numpy
opencv-python
```

---

## References

- **Paper:** [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) — Xie et al., NeurIPS 2021
- **Official repo:** [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)
- **Cityscapes:** [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)
