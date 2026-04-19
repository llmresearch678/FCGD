<div align="center">

<img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<br/>
<br/>

# 🧠 FCGD
## Frequency-Conditioned Graph Diffusion for<br/>Robust Unsupervised Domain Adaptation in Medical Image Segmentation


</div>

---

## 🔥 Highlights

<table>
<tr>
<td width="50%">

### ✅ Near-Supervised Performance
FCGD achieves **88.3% DSC** on MMWHS MRI→CT, closing the gap to the fully supervised upper bound to only **0.1 pp** — essentially bridging the unsupervised adaptation deficit.

</td>
<td width="50%">

### ✅ Exceeds Supervised Ceiling
On MS-CMRSeg LGE→bSSFP, FCGD uniquely achieves **85.2% DSC**, surpassing even the fully supervised model through Monte Carlo ensemble regularisation.

</td>
</tr>
<tr>
<td>

### ✅ Calibrated Uncertainty Estimation
Produces spatial uncertainty maps with **ECE = 0.048** at no extra cost, enabling risk-stratified radiological review — a key requirement for clinical AI deployment.

</td>
<td>

### ✅ Zero-Shot Cross-Dataset Transfer
Without any fine-tuning, FCGD generalises across imaging centres via implicit test-time domain adaptation using its frequency conditioning mechanism.

</td>
</tr>
</table>

---

## 📌 Overview

Cross-domain variability in medical imaging — arising from differences in scanners, acquisition protocols, and patient populations — remains one of the key obstacles to deploying AI-assisted segmentation in real-world clinical settings.

FCGD addresses this through a novel **structured latent** approach: rather than translating images between domains or aligning raw feature distributions, we lift convolutional features into an **anatomical graph space** and resolve domain discrepancies through a **frequency-conditioned latent diffusion process**.

<div align="center">

```
Input Image (source / target)
         │
         ▼
┌─────────────────────┐
│  CNN Encoder  Ec    │  → dense feature map  F ∈ ℝ^{C×H'×W'}
└─────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌──────────────┐
│Wavelet│  │  Anatomical  │
│  W    │  │Graph Builder │ → G = (V, E, A)
└───────┘  └──────────────┘
    │              │
    ▼              ▼
  f ∈ ℝ^{df}   ┌──────────┐
  (frequency)  │GNN Enc Eg│ → Z ∈ ℝ^{N×dL}
               └──────────┘
                      │
    ┌─────────────────┘
    ▼
┌──────────────────────────────────┐
│  Frequency-Conditioned Diffusion │
│  Forward:  Z₀ → Z₁ → … → Zₜ    │
│  Reverse:  Zₜ → … → Z*₀  (f)   │
└──────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Seg Decoder  C     │ → ŷ ∈ ℝ^{K×H×W}
└─────────────────────┘
         │
         ▼ (×M samples)
    Ensemble Mean ȳ  +  Uncertainty Map U
```

</div>

---

## 🏆 Results

### MMWHS: Cross-Modal Cardiac Segmentation

| Method | MRI→CT DSC (%) ↑ | MRI→CT ASD (mm) ↓ | CT→MRI DSC (%) ↑ |
|:-------|:---:|:---:|:---:|
| W/o Adapt. | 15.2 | — | 0.0 |
| AdvEnt | 82.1 ± 0.08 | 5.18 ± 3.9 | 72.5 ± 0.09 |
| CycleGAN | 79.6 ± 0.08 | 5.64 ± 4.3 | 69.2 ± 0.09 |
| MPSCL | 83.6 ± 0.06 | 4.61 ± 4.0 | 78.0 ± 0.07 |
| CiSeg | 86.5 ± 0.06 | 3.84 ± 2.90 | 80.6 ± 0.07 |
| **FCGD (Ours)** | **88.3 ± 0.05** | **3.38 ± 2.70** | **82.8 ± 0.06** |
| *Supervised UB* | *88.4 ± 0.10* | *2.13 ± 0.98* | *84.5 ± 0.05* |

### CHAOS: Cross-Modal Abdominal Segmentation

| Method | MRI→CT DSC (%) ↑ | CT→MRI DSC (%) ↑ |
|:-------|:---:|:---:|
| SAFAv2 | 68.97 ± 0.17 | — |
| DCLPS | 53.28 ± 0.23 | 63.00 ± 0.11 |
| CiSeg | 83.81 ± 0.05 | 84.28 ± 0.03 |
| **FCGD (Ours)** | **85.71 ± 0.04** | **86.15 ± 0.03** |

### MS-CMRSeg: Cross-Sequence Cardiac MRI

| Method | bSSFP→LGE DSC (%) ↑ | LGE→bSSFP DSC (%) ↑ |
|:-------|:---:|:---:|
| CycleGAN | 50.54 ± 0.27 | 69.11 ± 0.20 |
| CiSeg | 74.04 ± 0.16 | 82.59 ± 0.13 |
| **FCGD (Ours)** | **76.85 ± 0.14** | **85.20 ± 0.12†** |
| *Supervised UB* | *78.40 ± 0.16* | *83.24 ± 0.15* |

> **†** FCGD uniquely surpasses the supervised upper bound via MC ensemble regularisation.

---

## 🔬 Ablation Study (MMWHS MRI→CT)

| Setting | G | S | F | D | A | DSC (%) ↑ | Gain |
|:--------|:-:|:-:|:-:|:-:|:-:|:---------:|:----:|
| W/o Adapt. | | | | | | 15.20 | — |
| Baseline (A) | | | | | ✓ | 79.80 | — |
| +G | ✓ | | | | ✓ | 81.60 | +1.8 |
| +G S | ✓ | ✓ | | | ✓ | 83.20 | +1.6 |
| +G S F | ✓ | ✓ | ✓ | | ✓ | 85.10 | +1.9 |
| +G S D (no F) | ✓ | ✓ | | ✓ | ✓ | 86.00 | +2.8 |
| +G S F D (no A) | ✓ | ✓ | ✓ | ✓ | | 87.40 | +1.4 |
| **FCGD (Full)** | ✓ | ✓ | ✓ | ✓ | ✓ | **88.30** | **+0.9** |

> **G** = Graph encoder · **S** = Structural consistency · **F** = Frequency conditioning · **D** = Latent diffusion · **A** = Adversarial alignment

---

## 🗂️ Repository Structure

```
FCGD/
├── 📁 fcgd/                        # Main Python package
│   ├── 📁 models/
│   │   ├── fcgd.py                 # Full FCGD model (entry point)
│   │   ├── encoder.py              # Convolutional encoder Ec
│   │   ├── graph.py                # Anatomical graph construction + GNN Eg
│   │   ├── wavelet.py              # Learnable multi-scale wavelet encoder W
│   │   ├── diffusion.py            # Frequency-conditioned latent diffusion Dφ
│   │   └── decoder.py              # Segmentation decoder C + discriminator Dψ
│   ├── 📁 data/
│   │   ├── datasets.py             # MMWHS · CHAOS · MS-CMRSeg loaders
│   │   └── transforms.py           # Domain-robust augmentation pipeline
│   ├── 📁 losses/
│   │   └── losses.py               # L_sup · L_str · L_D · L_G · L_diff
│   └── 📁 utils/
│       ├── metrics.py              # DSC · ASD evaluators
│       ├── trainer.py              # Full training engine
│       └── visualise.py            # Segmentation + uncertainty visualisation
│
├── 📁 configs/
│   ├── mmwhs_mri2ct.yaml           # MMWHS MRI→CT (default)
│   ├── mmwhs_ct2mri.yaml           # MMWHS CT→MRI
│   ├── chaos_mri2ct.yaml           # CHAOS MRI→CT
│   └── mscmr_bssfp2lge.yaml       # MS-CMRSeg bSSFP→LGE
│
├── 📁 scripts/
│   ├── train.py                    # Training entry point
│   ├── evaluate.py                 # Full test-set evaluation
│   ├── infer.py                    # Single-image inference + uncertainty
│   └── ablation.py                 # Reproduce Table 4 ablation
│
├── 📁 experiments/                 # Output checkpoints and logs
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python ≥ 3.8
- PyTorch ≥ 1.13
- CUDA ≥ 11.0 (recommended: NVIDIA RTX 4090 or equivalent ≥ 24 GB VRAM)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/usmanusmani/FCGD.git
cd FCGD

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .
```

---

## 📦 Datasets

| Dataset | Modalities | Structures | Download |
|:--------|:-----------|:-----------|:---------|
| **MMWHS** | CT ↔ MRI | MYO, LAC, LVC, AA | [Grand Challenge](https://zmiclab.github.io/zxh/0/mmwhs/) |
| **CHAOS** | CT ↔ MRI | Liver, RK, LK, Spleen | [CHAOS Challenge](https://chaos.grand-challenge.org/) |
| **MS-CMRSeg** | bSSFP ↔ LGE | RVC, MYO, LVC | [MS-CMRSeg](http://www.sdspeople.fudan.edu.cn/zhuanglab/mscmrseg.html) |

Organise datasets as follows:
```
/data/
├── MMWHS/
│   ├── ct_train/     image_*.nii.gz  label_*.nii.gz
│   ├── mr_train/     image_*.nii.gz  label_*.nii.gz
│   ├── ct_test/
│   └── mr_test/
├── CHAOS/
│   ├── CT/           patient_*/  DICOM_anon/  Ground/
│   └── MR/           patient_*/  T1DUAL/  T2SPIR/
└── MSCMRSeg/
    ├── bSSFP/        patient_*_C0.nii.gz  patient_*_C0_manual.nii.gz
    └── LGE/          patient_*_LGE.nii.gz patient_*_LGE_manual.nii.gz
```

---

## 🚀 Quick Start

### 1. Training

```bash
# MMWHS: MRI → CT  (default configuration)
python scripts/train.py --config configs/mmwhs_mri2ct.yaml

# CHAOS: MRI → CT
python scripts/train.py --config configs/chaos_mri2ct.yaml

# MS-CMRSeg: bSSFP → LGE
python scripts/train.py --config configs/mscmr_bssfp2lge.yaml

# Resume from checkpoint
python scripts/train.py --config configs/mmwhs_mri2ct.yaml \
                        --resume outputs/fcgd_mmwhs_mri2ct/ckpt_ep100.pth
```

### 2. Evaluation

```bash
python scripts/evaluate.py \
    --config  configs/mmwhs_mri2ct.yaml \
    --ckpt    outputs/fcgd_mmwhs_mri2ct/ckpt_best.pth \
    --mc      10 \
    --save_vis outputs/vis/mmwhs
```

Expected output:
```
═══════════════════════════════════════════════════════
  FCGD Evaluation: MMWHS   mr → ct
═══════════════════════════════════════════════════════
Structure    DSC (%)             ASD (mm)
──────────────────────────────────────────────────────
MYO          78.70 ± 0.03        2.16 ± 0.32
LAC          90.40 ± 0.02        3.01 ± 2.88
LVC          91.60 ± 0.03        1.54 ± 0.50
AA           92.60 ± 0.01        6.82 ± 3.38
Mean         88.33 ± 0.05        3.38 ± 2.70
```

### 3. Single-Image Inference

```bash
python scripts/infer.py \
    --config configs/mmwhs_mri2ct.yaml \
    --ckpt   outputs/fcgd_mmwhs_mri2ct/ckpt_best.pth \
    --image  path/to/slice.png \
    --out    outputs/inference \
    --mc     10
```

Outputs:
- `prediction.png` — segmentation overlay + uncertainty heatmap
- `review_mask.png` — binary mask of pixels flagged for expert review (U > τ)
- `prob_maps.png`  — per-class probability maps

### 4. Ablation Study

```bash
# Reproduce Table 4 from the paper
python scripts/ablation.py --config configs/mmwhs_mri2ct.yaml --epochs 200
```

---

## 🐍 Python API

```python
import torch
from fcgd import FCGD

# Instantiate model
model = FCGD(
    in_channels    = 1,
    num_classes    = 5,       # background + 4 cardiac structures
    img_size       = 256,
    num_nodes      = 256,     # N  – graph resolution
    diff_timesteps = 10,      # T  – diffusion steps
    mc_samples     = 10,      # M  – Monte Carlo inference samples
)

# Load pre-trained weights
ckpt = torch.load('ckpt_best.pth', map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

# Uncertainty-aware inference
image = torch.randn(1, 1, 256, 256)   # (B, C, H, W)
with torch.no_grad():
    y_bar, U = model.predict(image, mc_samples=10)

# y_bar : (1, 5, 256, 256) – ensemble mean class probabilities
# U     : (1, 256, 256)    – pixel-wise predictive uncertainty

pred = y_bar.argmax(dim=1)             # (1, 256, 256) – hard label map

# Flag uncertain pixels for radiologist review
review_mask = U[0] > 0.05
print(f"Pixels requiring review: {review_mask.float().mean()*100:.1f}%")
```

---

## 🧩 Key Hyperparameters

| Parameter | Symbol | Default | Effect |
|:----------|:------:|:-------:|:-------|
| `num_nodes` | N | 256 | Graph resolution. Pareto-optimal: N=256. N=64 causes −2.7 pp DSC. |
| `diff_timesteps` | T | 10 | Denoising steps. T=10 is optimal; T>20 gives <0.2 pp gain at 2× cost. |
| `mc_samples` | M | 10 | MC ensemble size. M=10 yields ECE=0.048; M>10 shows no improvement. |
| `lambda_str` | λ | 0.1 | Structural consistency weight. Stable for λ ∈ [0.05, 0.5]. |
| `mu_adv` | µ | 0.01 | Adversarial alignment weight. Stable for µ ∈ [0.005, 0.05]. |

---

## 📐 Method Details

### Problem Formulation
Given a **labelled source domain** $\mathcal{D}_s = \{(x_s^{(i)}, y_s^{(i)})\}$ and an **unlabelled target domain** $\mathcal{D}_t = \{x_t^{(j)}\}$, FCGD learns a segmentation mapping $F_\theta: x \mapsto \hat{y}$ that generalises to the target domain without access to target labels.

### Anatomical Graph Construction
The convolutional feature map $\mathbf{F} \in \mathbb{R}^{C \times H' \times W'}$ is converted to a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$:

$$h_i = \frac{1}{|\Omega_i|} \sum_{(u,v) \in \Omega_i} \mathbf{F}(:, u, v), \quad A_{ij} = \exp\!\left(\frac{h_i^\top h_j}{\|h_i\|_2 \|h_j\|_2}\right)$$

### Frequency-Conditioned Diffusion
The forward process injects Gaussian noise:
$$Z_t = \sqrt{\bar\alpha_t}\, Z_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

The reverse process denoises $Z_T \to Z^*_0$ conditioned on frequency vector $\mathbf{f}$:
$$\mathcal{L}_\text{diff} = \mathbb{E}_{t, Z_0, \varepsilon}\!\left[\|\varepsilon - \varepsilon_\phi(\sqrt{\bar\alpha_t}Z_0 + \sqrt{1-\bar\alpha_t}\varepsilon,\, \mathbf{f},\, t)\|_2^2\right]$$

### Combined Training Objective
$$\mathcal{L}_\text{total} = \mathcal{L}_\text{sup} + \lambda \mathcal{L}_\text{str} + \mu \mathcal{L}_G$$

where $\mathcal{L}_\text{sup} = \mathcal{L}_\text{CE} + \mathcal{L}_\text{Dice}$, and $\mathcal{L}_\text{str} = \|\mathbb{E}[\mathbf{Z}_s] - \mathbb{E}[\mathbf{Z}_t]\|_F^2$.

### Uncertainty Estimation
$M$ independent reverse-chain runs produce predictions $\{\hat{y}^{(m)}\}_{m=1}^M$. The pixel-wise uncertainty map is:
$$U(u,v) = \frac{1}{M-1}\sum_{m=1}^M \|\hat{y}^{(m)}(u,v) - \bar{y}(u,v)\|_2^2$$

---

## 📊 Computational Cost

| Setting | GPU | Training Time | Inference (per slice) | Memory |
|:--------|:----|:-------------|:---------------------|:-------|
| Default (T=10, N=256, M=10) | RTX 4090 | ~18h / 200 ep | 12 ms | ~6 GB |
| Fast (T=5, N=128, M=5) | RTX 3090 | ~10h / 200 ep | 6 ms | ~4 GB |

---



---

## 📝 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

The authors gratefully acknowledge the institutional support and research environment provided by **Xiamen University Malaysia** and the **University of Memphis, USA**.

Baseline implementations reference:
[AdaOutput](https://github.com/wasidennis/AdaptSegNet) ·
[AdvEnt](https://github.com/valeoai/ADVENT) ·
[CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) ·
[SAFAv2](https://github.com/xmed-lab/HiDe-Prompt)

---

<div align="center">

Made with ❤️ at **Xiamen University Malaysia**

⭐ If you find this work helpful, please consider giving it a star!

</div>
