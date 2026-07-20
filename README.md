# SAA: Statistical-level Artifact Augmentation for Generalizable AI-Generated Image Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-green)

Official PyTorch implementation of **SAA: Statistical-level Artifact Augmentation for Generalizable AI-Generated Image Detection**.

> **Abstract:** AI-generated image detection remains challenging due to the rapid evolution of generative models, which introduces diverse and unseen artifact patterns during inference. We propose **Statistical-level Artifact Augmentation (SAA)**, a novel training-time strategy that enhances the generalization capability of deepfake detectors. Unlike pixel-space augmentations or image-level mixing, SAA operates at the feature-statistics level within a Vision Transformer (ViT) backbone. By applying **MixStyle** — mixing the mean and variance of patch token features — exclusively to fake samples at intermediate ViT layers, SAA perturbs generator-specific artifact signatures while preserving semantic content. This forces the detector to learn more fundamental, generalizable forgery patterns. Built on a frozen CLIP ViT-Large backbone with lightweight LoRA adapters, SAA achieves strong cross-generator generalization across diverse benchmarks.

---

## Motivation

<div align="center">
  <img src="resources/tsne_all_24_layers_grid_adapter.png" alt="t-SNE visualization of feature distributions" width="800"/>
  <p><em>Figure 1: t-SNE visualization of CLIP ViT feature distributions across 24 layers. Real and AI-generated image features from different generators exhibit distinct statistical patterns, motivating layer-wise statistical augmentation.</em></p>
</div>

---

## Key Features

- **Statistical-level Artifact Augmentation** — ViTMixStyle perturbs feature statistics (mean/std) of fake samples at intermediate ViT layers, improving generalization to unseen generators
- **Fake-Specific Augmentation** — MixStyle is applied only to fake samples; real image features remain untouched, preserving their natural distribution
- **CLIP ViT-Large Backbone** — Frozen vision backbone with parameter-efficient LoRA fine-tuning (rank=8) on attention projections
- **Multi-Benchmark Evaluation** — Comprehensive evaluation across UFD (19 categories), GenImage (8 categories), and AIGI (25 categories) benchmarks

---

## Project Structure

```
SAA/
├── check_points/                  # Model checkpoints and training logs (auto-generated)
├── Dataset/
│   ├── train/                     # Training data (user-provided)
│   │   ├── <generator_1>/
│   │   │   ├── 0_real/            # Real images
│   │   │   └── 1_fake/            # AI-generated images
│   │   └── <generator_2>/...
│   ├── val/                       # Validation data (user-provided)
│   ├── eval_UFD/                  # UFD evaluation benchmark
│   ├── eval_GenImage/             # GenImage evaluation benchmark
│   └── eval_AIGI/                 # AIGI evaluation benchmark
├── models/
│   ├── network/
│   │   └── network.py             # SAP_Net, ViTMixStyle, model definition
│   └── trainer.py                 # SAP_Trainer: training loop, metrics, checkpoints
├── options/
│   └── options.py                 # Argument parsing and configuration
├── resources/
│   ├── network_architecture.pdf   # Network architecture diagram
│   └── tsne_all_24_layers_grid_adapter.png  # t-SNE motivation figure
├── evaluate.py                    # Evaluation entry point (per-category Acc & AP)
├── evaluate.sh                    # Evaluation shell script wrapper
├── train.py                       # Training entry point
├── train.sh                       # Training shell script wrapper
└── util.py                        # Data loading, transforms, balanced sampler, logger
```

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU recommended)

### Setup

1. Clone the repository:

```bash
git clone <repository_url>
cd SAA
```

2. Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft tensorboardX scikit-learn tqdm numpy huggingface_hub
pip install opencv-python pillow pyyaml
```

---

## Dataset Preparation

Organize your datasets following the structure below. Each generator category directory should contain `0_real/` and `1_fake/` subdirectories with the corresponding images.

```
Dataset/
├── train/
│   ├── progan/
│   │   ├── 0_real/              # Real images for training
│   │   └── 1_fake/              # ProGAN-generated images
│   ├── stable_diffusion_v_1_4/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   ├── biggan/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── adm/
│       ├── 0_real/
│       └── 1_fake/
├── val/                         # Same structure for validation
├── eval_UFD/                    # UFD benchmark (19 generator types)
├── eval_GenImage/               # GenImage benchmark (8 generator types)
└── eval_AIGI/                   # AIGI benchmark (25 generator types)
```

The training and validation data must be downloaded separately. Please refer to the relevant benchmarks:
- [UFD (Universal Fake Detector)](https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/main)
- [GenImage](https://github.com/GenImage-Dataset/GenImage)
- [AIGI (AI-Generated Image)](https://github.com/IAAR-Shanghai/AIGCDetect)

---

## Usage

### Training

Basic training with default settings:

```bash
bash train.sh
```

Custom training with configurable parameters:

```bash
GPU_ID=0 \
EXP_NAME=saa_experiment \
TRAIN_ROOT=Dataset/train \
VAL_ROOT=Dataset/val \
TRAIN_CLASSES="progan stable_diffusion_v_1_4 biggan adm" \
VAL_CLASSES="progan stable_diffusion_v_1_4 biggan adm" \
BATCH_SIZE=16 \
EPOCHS=5 \
bash train.sh
```

**Environment variables:**
| Variable | Default | Description |
|---|---|---|
| `GPU_ID` | `0` | CUDA device ID |
| `EXP_NAME` | `sap_G4_aug` | Experiment name for logging |
| `TRAIN_ROOT` | `Dataset/train` | Training data root |
| `VAL_ROOT` | `Dataset/val` | Validation data root |
| `TRAIN_CLASSES` | `progan ... adm` | Space-separated generator names for training |
| `VAL_CLASSES` | `progan ... adm` | Space-separated generator names for validation |
| `SEED` | `3407` | Random seed |
| `BATCH_SIZE` | `16` | Batch size |
| `EPOCHS` | `5` | Number of epochs |
| `NUM_WORKERS` | `4` | DataLoader workers |

Training outputs are saved to `check_points/<EXP_NAME>/train_sap/`:
- Model checkpoints: `sap_best_model.pth`, `sap_last_model.pth`, `sap_final_model.pth`
- Metrics: `sap_metrics.csv` (epoch, loss, accuracy, learning rate, time)
- TensorBoard logs

### Evaluation

1. Edit `evaluate.sh` to set the correct paths:

```bash
EVAL_DATA_BASE="/path/to/evaluation/datasets"
MODEL_BASE="/path/to/check_points/<EXP_NAME>"
```

2. Run evaluation:

```bash
bash evaluate.sh
```

This evaluates the trained model across all three benchmarks (UFD, GenImage, AIGI) and reports per-category and mean accuracy and average precision (AP).

### Python API

**Training:**

```python
from options.options import Options
from models.network.network import SAP_Net
from models.trainer import SAP_Trainer
from util import get_dataset_train, get_dataset_test, get_bal_sampler
import torch
from torch.utils.data import DataLoader

# Prepare data
train_dataset = get_dataset_train("Dataset/train", ["progan", "biggan"])
train_loader = DataLoader(train_dataset, batch_size=16, sampler=get_bal_sampler(train_dataset))

val_dataset = get_dataset_test("Dataset/val", ["progan", "biggan"])
val_loader = DataLoader(val_dataset, batch_size=16, sampler=get_bal_sampler(val_dataset))

# Initialize model
model = SAP_Net(r=8, alpha=0.1, target_layer_idx=18)

# Train
trainer = SAP_Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    epochs=5,
    save_dir="./checkpoints",
)
trainer.fit()
```

**Inference:**

```python
from models.network.network import SAP_Net
import torch

model = SAP_Net(r=8)
state_dict = torch.load("path/to/sap_best_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.cuda().eval()

with torch.no_grad():
    logits = model(images)
    probs = torch.sigmoid(logits).cpu().numpy()
```

---

## Configuration Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--experiment_name` | str | `my_experiment` | Experiment name for logging |
| `--seed` | int | `3407` | Random seed |
| `--train_data_root` | str | `''` | Training data root path |
| `--train_classes` | list[str] | `['car','cat','chair','horse']` | Training generator categories |
| `--val_data_root` | str | `''` | Validation data root path |
| `--val_classes` | list[str] | `['car','cat','chair','horse']` | Validation generator categories |
| `--batch_size` | int | `16` | Batch size |
| `--epochs` | int | `5` | Number of training epochs |
| `--rank` | int | `8` | LoRA rank for fine-tuning |
| `--alpha` | float | `0.1` | MixStyle Beta distribution α parameter |
| `--target_layer_idx` | list[int] | `[5,6,7,8,9]` | ViT encoder layers for MixStyle hook |
| `--max_samples` | int | `200` | Max samples per class (for debugging) |
| `--num_workers` | int | `4` | DataLoader worker processes |
| `--weights` | str | `''` | Path to pretrained checkpoint (evaluation) |
| `--eval_data_root` | str | `''` | Evaluation data root path |
| `--use_data_augument` | flag | `False` | Enable data augmentation (blur, JPEG) |
| `--blur_prob` | float | `0.5` | Gaussian blur probability |
| `--blur_sig` | str | `0.0,3.0` | Blur sigma range |
| `--jpg_prob` | float | `0.5` | JPEG compression probability |
| `--jpg_qual` | str | `30,100` | JPEG quality range |

> **Note:** The default values for `--train_classes` and `--val_classes` in `options.py` are placeholders. The actual training configuration is specified via the shell scripts.

---

## Model Architecture

### Network Overview

View the detailed network architecture: [📄 network_architecture.pdf](resources/network_architecture.pdf)



### SAP_Net

The core detection model consists of three components:

1. **Frozen CLIP ViT-L/14 Backbone**
   - Pre-trained `openai/clip-vit-large-patch14` from HuggingFace Transformers
   - All parameters frozen during training

2. **LoRA Adapters (Parameter-Efficient Fine-Tuning)**
   - Injected on attention projection matrices: `q_proj`, `k_proj`, `v_proj`, `out_proj`
   - Configuration: rank=8, lora_alpha=16, dropout=0.0, bias=none
   - Only LoRA weights are updated during training (≈1–2M trainable parameters)

3. **ViTMixStyle — Statistical-level Artifact Augmentation**
   - Applied via forward hooks at configurable ViT encoder layers (default: layer 18 in `train.sh`, range `[5,6,7,8,9]` in `options.py`)
   - For each ViT layer's patch token output: computes instance-level mean and standard deviation, normalizes features, then mixes statistics between paired fake samples using a Beta-distribution coefficient (α=0.1)
   - The [CLS] token is preserved unchanged to maintain global semantic information
   - During training, MixStyle is applied only to fake samples (identified via `_current_targets` attribute); during evaluation, it is bypassed entirely

4. **Binary Classification Head**
   - `Dropout(p=0.5)` → `Linear(1024, 1)`
   - Outputs a single logit per image (real vs. fake)

### Training Strategy

| Component | Detail |
|---|---|
| Loss Function | `BCEWithLogitsLoss` |
| Optimizer | AdamW (lr=1e-4) |
| Learning Rate Schedule | Cosine annealing (1e-4 → 1e-5) |
| Gradient Clipping | max_norm=1.0 |
| Data Augmentation | Gaussian blur, JPEG compression (optional) |
| Sampling | Balanced sampling via `WeightedRandomSampler` |
| Image Size | 224×224 |
| Multi-GPU | Automatic `DataParallel` support |

---

## Model Zoo

We provide a pretrained checkpoint trained on ProGAN, Stable Diffusion v1.4, BigGAN, and ADM:

| Model | Backbone | Training Data | Link |
|---|---|---|---|
| SAA (SAP_Net) | CLIP ViT-L/14 | ProGAN + SD v1.4 + BigGAN + ADM | [Google Drive](https://drive.google.com/file/d/1boXCdykjiUB8S31VgwucZSnjQcVHYZ84/view?usp=drive_link) |

After downloading, place the checkpoint at `check_points/<experiment_name>/train_sap/model/sap_best_model.pth` and run evaluation:

```bash
# Edit evaluate.sh paths, then:
bash evaluate.sh
```

---

## Supported Generative Models

The evaluation benchmarks cover a wide range of generative models:

### UFD Benchmark (19 categories)
ProGAN, StyleGAN, BigGAN, CycleGAN, StarGAN, GauGAN, DeepFake, SeeingDark, SAN, CRN, IMLE, Guided, LDM (200, 200_cfg, 100), GLIDE (100_27, 50_27, 100_10), DALL-E

### GenImage Benchmark (8 categories)
ADM, BigGAN, GLIDE, Midjourney, Stable Diffusion v1.4, Stable Diffusion v1.5, VQDM, Wukong

### AIGI Benchmark (25 categories)
BlendFace, BLIP, CommunityAI, DALL-E 3, E4S, FaceSwap, FLUX.1-dev, GLIDE, Imagen3, Infinite-ID, InstantID, InSwap, IP-Adapter, Midjourney, PhotoMaker, ProGAN, R3GAN, Stable Diffusion 3, SDXL, SimSwap, SocialRF, StyleGAN-XL, StyleGAN3, StyleSwim, WFIR

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{author2026saa,
  title={SAA: Statistical-level Artifact Augmentation for Generalizable AI-Generated Image Detection},
  author={...},
  booktitle={...},
  year={2026}
}
```

---

## License

This project is released for academic research and educational purposes. Please refer to the LICENSE file for details.

---

## Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft) libraries
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [MixStyle](https://github.com/KaiyangZhou/mixstyle) (Zhou et al., ICLR 2021) for the original MixStyle idea
- [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/main) (UFD) benchmark
- [GenImage](https://github.com/GenImage-Dataset/GenImage) benchmark
- [AIGCDetect](https://github.com/IAAR-Shanghai/AIGCDetect) (AIGI) benchmark
