# Project Journey: Comparing Deep Learning Architectures for Continuous ECoG Motor Decoding

A detailed chronicle of our research progress — what we built, what we tried, what worked, what failed, and what we learned.

---

## Phase 0–2: Foundation (Days 1–3)

### What We Built
Set up a clean research codebase from scratch in `020926-miller-library-2019/`:
- **Config system**: YAML-based configs for all experiments
- **Seed management**: `set_seed(seed)` for reproducibility
- **Data pipeline**: Two preprocessing modes (`ours` and `lomtev`) with proper train/val/test temporal splits
- **Dataset classes**: `FingerFlexDataset` (raw ECoG) and `LomtevDataset` (Morlet spectrograms)
- **W&B integration**: optional logging with `--no-wandb` flag

### Key Data Findings
- **9 Miller Library patients**: 38–64 ECoG channels, 179–610 seconds, 1000 Hz
- **3 BCI-IV subjects**: 48–64 channels, pre-split train/test, 25 Hz finger data (needs interpolation)
- PSD analysis confirmed data is raw/unfiltered: 60 Hz powerline + harmonics present
- All patients clean — no NaNs, dead channels, or artifacts in sampled windows

### Two Preprocessing Pipelines
| Step | `ours` mode | `lomtev` mode |
|------|-------------|---------------|
| Order | filter → zscore → CAR | zscore → CAR → filter |
| Bandpass | 1–200 Hz | 40–300 Hz |
| Target scaling | z-score | MinMax [0,1] |
| Input | Raw ECoG @ 1000 Hz | Morlet spectrograms @ 100 Hz |

The Lomtev pipeline turned out to be far superior (spectrograms >> raw), so all serious experiments use it.

---

## Phase 3: U-Net Baseline (Days 3–5)

### Porting FingerFlex
Ported the Lomtev U-Net architecture from `../FingerFlex/Lightning_BCI-autoencoder.ipynb`:
- `ConvBlock`: Conv1d (no bias) → LayerNorm → GELU → Dropout → MaxPool
- `UpConvBlock`: ConvBlock → nn.Upsample(mode='linear')
- `AutoEncoder1D`: spatial_reduce + 5 encoder blocks + 5 decoder blocks with skip concatenation
- ~652K params for 62 channels

### BCI-IV Results (Our Anchor Baseline)
```
python scripts/train.py --config configs/bci4_lomtev.yaml --subject 1 --gpu 0
```

| Subject | Channels | Val r (σ=6) | Test r (σ=1) |
|---------|----------|------------|-------------|
| S1 | 62 | 0.686 | 0.575 |
| S2 | 48 | 0.633 | 0.520 |
| S3 | 64 | 0.737 | 0.692 |
| **Mean** | — | **0.685** | **0.597** |

Val r at σ=6 matches FingerFlex arXiv exactly (0.686, 0.633, 0.737). This proved our pipeline is correct.

### Miller Library Results (9 Patients)
```bash
# Batch runner for all 9 patients
bash scripts/run_all_miller.sh
```

| Patient | Ch | Test r | Notes |
|---------|-----|--------|-------|
| cc | 63 | **0.725** | Best patient |
| zt | 61 | 0.600 | Strong |
| jp | 58 | 0.548 | Good |
| jc | 47 | 0.536 | Good |
| mv | 43 | 0.460 | Short recording (179s) |
| wc | 64 | 0.383 | Moderate |
| bp | 46 | 0.369 | Moderate |
| ht | 64 | 0.287 | Weak |
| wm | 38 | 0.083 | Very weak (fewest channels) |
| **Mean** | — | **0.443** | σ=0.192 |

**Key insight**: Huge patient variability (0.08–0.73). This is expected — electrode placement and cortical coverage vary massively.

### Raw ECoG U-Net (Failed Experiment)
Tested whether raw ECoG input (no spectrograms) could work:
- Mean test r = 0.227 vs 0.443 (spectrograms) — roughly **half** the performance
- Raw models early-stopped at epoch 18–33 — they plateau fast
- **Conclusion**: Morlet wavelets perform critical time-frequency feature extraction that Conv1d can't replicate

---

## Phase 4: TCN Architecture (Days 5–6)

### Implementation
Built a standard Temporal Convolutional Network following Bai et al. 2018:
- 6 blocks with dilations [1, 2, 4, 8, 16, 32]
- Receptive field: 505 timesteps (covers full 256-step window)
- ~601K params (matched to U-Net)
- Acausal convolutions (padding='same')

### Results: TCN Failed
```
python scripts/train.py --config configs/fingerflex_tcn.yaml --patient bp --gpu 0
```

| Variant | Mean Test r | vs U-Net |
|---------|------------|----------|
| TCN v1 (6 blocks, 64ch, drop=0.1) | 0.360 | -0.071 |
| TCN v2 (more regularization) | ~0.34 | -0.09 |

TCN v1 lost on 7/9 Miller patients. v2 (dropout 0.25, 48ch, higher weight decay) was even worse — killed after 3 patients.

**Root cause**: Flat single-resolution architecture lacks the multi-scale inductive bias that U-Net's encoder-decoder provides. More regularization can't fix a missing inductive bias.

**Operational lesson learned**: Monitor first 2–3 patients. If consistently worse, kill early. We wasted GPU hours on the full 9-patient v1 run.

---

## Phase 5: Transformer Architecture (Days 6–15) — The Big Scaling Story

### The Architecture
```
Input: (B, n_ch, n_freq, T=256)
  → reshape: (B, n_ch*n_freq, T)
  → spatial_reduce: Conv1d → LN → GELU → Drop
  → transpose: (B, T, d_model)
  → sinusoidal positional encoding
  → N × TransformerEncoderLayer (pre-LN, bidirectional)
  → transpose + output Conv1d → (B, 5, T)
```

Key design choices:
- Pre-LN (`norm_first=True`) — stable without warmup
- Bidirectional attention (no causal mask) — matches U-Net/TCN
- GELU activation, sinusoidal PE, no downsampling

### Critical Bug: Sliding-Window Evaluation
**Initial disaster**: val r ~0.05 after 30 epochs. The model appeared to learn nothing.

**Root cause**: `evaluate_fullsig` fed the entire signal (5K–20K timesteps) through the Transformer at once. For convnets this is fine (translation-equivariant), but for Transformers the attention distribution and positional encoding completely shift at test-time sequence lengths vs training windows (256 steps).

**Fix**: Added `_sliding_window_forward()` with 50% overlapping windows, matching the 256-step training context. After fix: val r jumped from 0.01 to 0.32 at epoch 2.

**Lesson**: Sliding-window eval is mandatory for Transformers on this task.

### The Scaling Journey (BCI-IV, 11 Rounds of Tuning)

This was the most extensive and informative phase. We systematically scaled the Transformer from d=64 (0.404) to d=1024+h16 (0.621):

| Round | What Changed | Best Mean r | Key Finding |
|-------|-------------|-------------|-------------|
| 1 | Initial grid (d=64/128, L=2/4, 2 LRs) | 0.444 | d=128 >> d=64; lr=8.42e-5 best |
| 2 | Depth (L=2→5) | 0.473 | L=4 optimal, monotonic to L=4 |
| 3 | FF width (256→1024) | 0.505 | ff=512 biggest jump (+0.031) |
| 4 | Heads (4 vs 8) | 0.502 | No difference |
| 5 | Spatial kernel (k=1 vs k=3) | 0.505 | k=3 hurt |
| 6 | d=256 | 0.517 | S1 jumped 0.435→0.505 |
| 7 | Dropout sweep | 0.517 | drop=0.1 still best |
| 8 | Hybrid (downsample+transformer+upsample) | 0.434 | **Failed** — lost information |
| 9 | d=512 | 0.567 | ff=2048 improved all subjects |
| 10 | d=1024 | **0.599** | **First time beating U-Net!** |
| 11 | Depth+heads ablation | **0.621** | L6+h16 synergistic |

**Scaling trend**: d64(0.404) → d128(0.468) → d256(0.517) → d512(0.567) → d1024(0.599) → h16(0.609) → L6+h16(**0.621**)

### Round 11 Deep Dive: The Breakthrough
The L6+h16 configuration was the key insight:

| Config | S1 | S2 | S3 | Mean | Params |
|--------|------|------|------|------|--------|
| L4, h8 (baseline) | 0.583 | 0.484 | 0.731 | 0.599 | 53M |
| L6, h8 | 0.580 | 0.471 | **0.745** | 0.599 | 78M |
| L4, **h16** | **0.612** | 0.482 | 0.732 | **0.609** | 53M |
| **L6, h16** | **0.620** | **0.495** | **0.748** | **0.621** | 78M |
| L8, h16 | 0.605 | — | — | — | 104M |

**Key insights**:
1. **h16 (16 heads, head_dim=64) is the primary lever** — +0.029 on S1
2. **L6 depth helps S3** specifically — +0.014
3. **L6+h16 is synergistic** — improves ALL subjects simultaneously
4. **L8 is over-capacity** — L6 is the depth sweet spot
5. **h32 no gain** — h16 is the head sweet spot

### What Failed in Phase 5
- **Hybrid U-Net+Transformer**: skip shortcut caused overfitting (0.434 mean)
- **Window=512**: longer context hurt (S1: -0.052)
- **ff=8192**: larger FFN hurt generalization
- **Spatial bottleneck 128**: destroyed spatial info (mean=0.569)
- **Multiscale transformer**: "dead finger" bug, too complex for 33K samples (mean≈0.553)
- **SwiGLU activation**: worse than GELU (-0.018)
- **d=1536 with ff=4096**: under-scaled FFN ratio, too slow
- **Higher LRs (1e-4, 2e-4, 3e-4)**: all worse — noisier val_r → premature LR decay
- **Scheduler tracking val_r instead of val_loss**: counterintuitively worse (-0.013)

### The Scheduler Discovery
The ReduceLROnPlateau scheduler tracks val_loss (mode=min), but we checkpoint on val_r. These diverge, and we thought this was a bug. Testing showed it's actually a **feature**:

- val_loss plateaus at epoch 6 → fires LR halving at epoch 15
- At that point val_r is at 0.653 and still climbing
- The half-LR creates a "precision phase": model climbs 0.653→0.688 in 5 more epochs
- Tracking val_r delays LR drop to epoch 31 → noisy oscillations → never recovers

**Conclusion: keep `scheduler(val_loss, mode=min)` + `checkpoint(val_r, mode=max)`.** The mismatch is intentional and beneficial.

### BCI-IV Architecture Comparison (Final)

| Model | S1 | S2 | S3 | Mean | Params |
|-------|------|------|------|------|--------|
| TCN | — | — | — | 0.408 | ~600K |
| U-Net | 0.575 | **0.520** | 0.692 | 0.596 | ~650K |
| **Transformer** | **0.620** | 0.495 | **0.748** | **0.621** | 78M |

Transformer beats U-Net by +0.025 mean but at 120× the parameter cost. S2 remains U-Net's stronghold.

---

## Phase 5 Continuation: Paper Reproduction Track (Days 15–20)

### Milestone 5.7 — Literature Review

Read every relevant paper on BCI-IV Dataset 4 SOTA. Key findings:

**Corrected SOTA Table (official test set, Pearson r)**:
| Method | Year | Mean r | Status |
|--------|------|--------|--------|
| Flamary & Rakotomamonjy | 2012 | 0.43 | Competition winner |
| Liang & Bougrain | 2012 | 0.48 | 2nd place |
| Xie et al. | 2018 | 0.52 | CNN-LSTM |
| FingerFlex v1 | 2022 | 0.64 | We reproduce exactly |
| **DTCNet** | **2025** | **0.69** | **Credible SOTA** |
| DeepFingerNet | 2025 | 0.69 | NOT reproducible |
| BC4D4 | 2025 | 0.85† | Bogus (70% data removed) |
| Tragoudaras | 2023 | 0.886† | Suspicious NAS paper |

**Critical discovery**: DTCNet's abstract claims "82%" — but this is the **single best finger on the single best subject** (S3 Thumb). The actual mean across subjects is 0.69. The abstract is misleading.

**Evaluation protocol trap**: Our test σ=1 numbers aren't directly comparable to published σ=6 numbers. Our U-Net val_r (σ=6) = 0.685 matches FingerFlex exactly. Converting consistently: our Transformer σ=1 test_r of 0.621 ≈ σ=6 of ~0.71–0.73.

### Milestone 5.8 — FingerFlex v1 Exact Reproduction
**Goal**: Reproduce FingerFlex's published S1=0.686, S2=0.633, S3=0.737 with their exact setup (seed=42, 20 epochs, fixed LR, 100% train split).

```
python scripts/train.py --config configs/bci4_fingerflex_v1_reproduction.yaml --subject 1 --gpu 0
```

**Results**:
| Subject | Our result (σ=6) | Published | Gap |
|---------|-----------------|-----------|-----|
| S1 | 0.605 | 0.686 | -0.081 |
| S2 | 0.535 | 0.633 | -0.098 |
| S3 | 0.701 | 0.737 | -0.036 |

**Findings**:
1. Cannot reproduce with single seed. S3 peaked at epoch 1 (0.701) then declined to 0.614 by epoch 20 — fixed LR causes overfitting
2. FingerFlex checkpoint shows `epoch=16, val_r=0.679` (not 0.686) — their own run didn't hit 0.686
3. Published numbers are likely best-of-multiple-seeds
4. **Our ReduceLROnPlateau scheduler is what makes our runs better** than naive FingerFlex

**Verdict**: Partially reproduced. Architecture confirmed identical. Gap = seeds/luck, not missing component.

### Milestone 5.9 — DeepFingerNet Reproduction (FAILED)
**Goal**: Reproduce claimed mean=0.69 for nested U-Net architecture.

#### Attempt 1: Summed Wavelet Power (n_wavelets=0)
The paper equation "X_wavelet = Σ W_k * X" interpreted as summing 40 frequency bands → single channel per electrode.

```
python scripts/train.py --config configs/bci4_deepfingernet.yaml --subject 1 --gpu 0
```

| Subject | Our result | Paper claim | Gap |
|---------|-----------|-------------|-----|
| S1 | 0.169 | 0.71 | -0.54 |
| S2 | 0.225 | 0.59 | -0.37 |
| S3 | 0.363 | 0.77 | -0.41 |
| **Mean** | **0.252** | **0.69** | **-0.44** |

Summing frequencies destroys all frequency discrimination → model can't learn.

#### Attempt 2: Full Spectrogram (n_wavelets=40)
**Hypothesis**: Paper uses full 40-frequency spectrogram, not summed power.

```
python scripts/train.py --config configs/bci4_deepfingernet_fullspec.yaml --subject 1 --gpu 0
```

| Subject | Paper setup (fixed lr) | Our setup (adaptive lr) | Paper claim |
|---------|----------------------|------------------------|-------------|
| S1 | 0.427 | 0.396 | 0.71 |
| S2 | 0.312 | — | 0.59 |
| S3 | 0.541 | — | 0.77 |
| **Mean** | **0.427** | — | **0.69** |

**Root causes of failure**:
1. Fixed lr=2e-5 → rapid overfitting (train loss drops 20× in 5 epochs, val_r peaks at epoch 2–4)
2. k=1 decoder convolutions → no temporal receptive field in reconstruction path
3. Our U-Net (same preprocessing) gets 0.575 vs NestedUNet's best 0.427

#### Attempt 3 (v2): Architecture Fix — Dropout + Pre-Norm Decoder — FAILED

Re-examined Fig. 2 block diagrams and found two bugs: (1) missing Dropout layers in encoder and decoder, and (2) wrong decoder ordering — paper uses pre-norm (LN→GELU→Conv→Drop), not post-norm. Fixed both in `nested_unet.py` and re-ran.

| Config | S1 | S2 | S3 | Mean |
|--------|-----|-----|-----|------|
| v2 raw (n_wv=0, dropout+prenorm) | 0.188 | 0.264 | 0.404 | **0.285** |
| v2 fullspec (n_wv=40, dropout+prenorm) | 0.342 | 0.351 | 0.547 | **0.413** |

Dropout helped the raw config slightly but hurt the fullspec config. The same overfitting pattern persists under fixed lr=2e-5 regardless of architecture fixes.

**Verdict**: ❌ DeepFingerNet NOT reproducible after 4 attempts. Best result across all attempts: mean=0.427 (gap of 0.26 vs claimed 0.69). Architecture is inferior to our plain U-Net (0.427 vs 0.597). The paper likely reports validation-set performance with undisclosed hyperparameter search — their claimed 0.69 is comparable to our U-Net's val_r=0.685 on the same preprocessing, not a genuine improvement.

### Milestone 5.10 — DTCNet Reproduction (IN PROGRESS)
**Goal**: Reproduce DTCNet's mean=0.69 (S1=0.71, S2=0.59, S3=0.77).

#### Implementation
Built `src/models/dtcnet.py` with the paper's exact architecture:
- Encoder: 6 stages with dims [64, 64, 128, 256, 512, 512]
- Sawtooth dilations: [1, 1, 2, 3, 1, 2]
- Kernels: [3, 7, 7, 5, 5, 5]
- Stage 0 (Feature Reduction): Conv + LN + GELU + Dropout, **no MaxPool**
- Stages 1–5: Conv + LN + GELU + Dropout + **MaxPool(2)**
- Decoder: ConvTranspose1d(stride=2) + skip concatenation + Conv + LN + GELU
- Head: Conv1d(64, 5, k=1)

#### Architecture Bug: First vs Corrected Version
The first implementation was **wrong** — missing MaxPool, using BatchNorm, no ConvTranspose in decoder. This gave 4.5M params (vs paper's claim of ~550K). After reading the paper more carefully:

| Feature | Wrong v1 | Corrected v2 | Paper |
|---------|----------|--------------|-------|
| Downsampling | None | MaxPool(2) per stage | MaxPool(2) |
| Normalization | BatchNorm | LayerNorm | LayerNorm |
| Decoder upsampling | Upsample(linear) | ConvTranspose1d | ConvTranspose1d |
| Params (62ch) | 4.5M | 5.2M | ~550K (suspect) |

**Param count discrepancy**: Paper claims 550–790K but with stated dims [64→512], the first Conv1d(2480,64,k=3) alone costs 476K params. Our corrected implementation = 5.2M. The paper's count appears to be an error.

Corrected model passes sanity checks:
```
python -c "
from src.models.dtcnet import DTCNet
import torch
m = DTCNet(62*40, 5)
x = torch.randn(2, 62*40, 256)
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
print(f'Output: {m(x).shape}')
"
# → Params: 5,237,189  Output: torch.Size([2, 5, 256])
```

**Status**: Model implemented and verified. Training not yet launched (waiting for GPU access).

---

## Common Commands Reference

### Training
```bash
# Single subject, specific GPU
python scripts/train.py --config configs/bci4_lomtev.yaml --subject 1 --gpu 0 --no-wandb

# When GPU NVML fails (GPU 1 hardware error)
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/bci4_dtcnet.yaml --subject 1 --gpu 0 --no-wandb

# Background training with logging
nohup python scripts/train.py --config configs/bci4_transformer_d1024_L6_h16.yaml --subject 1 --gpu 0 --no-wandb > results/run_s1.log 2>&1 &

# Explicit Python path (for shell scripts where conda isn't active)
/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python scripts/train.py ...
```

### GPU Management
```bash
nvidia-smi                                    # Check GPU availability
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
watch -n 5 nvidia-smi                         # Monitor during training
```

### Monitoring
```bash
tail -f results/run_s1.log                    # Watch training progress
grep "val_r" results/run_s1.log               # Check validation metrics
grep "test_r" results/run_s1.log | tail -1    # Check final test result
```

### Model Sanity Checks
```bash
python -c "
from src.models.dtcnet import DTCNet
import torch
m = DTCNet(62*40, 5)
x = torch.randn(2, 62*40, 256)
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
print(f'Output: {m(x).shape}')
"
```

---

## Key Lessons Learned

### Scientific
1. **Spectrograms >> raw ECoG**: Morlet wavelets perform critical feature extraction (mean r 0.443 vs 0.227)
2. **Multi-scale inductive bias matters**: U-Net's encoder-decoder beats flat TCN (-0.071 mean r)
3. **Transformers scale with model size**: monotonic improvement from d=64 (0.404) to d=1024 (0.599)
4. **16 heads is a sweet spot**: finer attention granularity (head_dim=64) drives S1 performance
5. **More regularization ≠ better** when the core issue is missing inductive bias (TCN v2)
6. **Published numbers often aren't reproducible**: FingerFlex (seeds/luck), DeepFingerNet (fundamentally wrong)
7. **Paper claims need scrutiny**: DTCNet "82%" = single finger/subject; BC4D4 removes 70% of data
8. **ReduceLROnPlateau > fixed LR**: the scheduler is what makes our runs competitive
9. **Scheduler val_loss + checkpoint val_r**: the "mismatch" is intentionally beneficial

### Operational
1. **Always check `nvidia-smi` before launching** — GPUs can be taken by other users
2. **Monitor first 2–3 patients** before committing to full sweep
3. **Use explicit Python path** in shell scripts: conda env isn't always active in subshells
4. **Launch nohup jobs in separate Bash calls** — multiple in one call causes silent crashes
5. **GPU 1 hardware error**: use `CUDA_VISIBLE_DEVICES=2 --gpu 0` workaround
6. **Scheduler naming**: trainer checks for `"reduce_on_plateau"`, not `"plateau"`
7. **Transformers need sliding-window eval** — full-signal evaluation produces garbage
8. **σ=1 vs σ=6 matters**: our honest σ=1 test numbers are ~0.08 below published σ=6 numbers

### Storage
- Code on `/mnt/onefs/` (50 GB cap), data/results on `/mnt/archive/` (300 GB)
- Results dir symlinked: `results/` → `/mnt/archive/home/yyu2024/PLaCT_data/results`
- Always `conda clean --all` and `pip cache purge` after installs
- Never store .pt/.mat/.npy on home filesystem

---

## Project File Structure

```
020926-miller-library-2019/
├── ROADMAP.md              # Living project plan (single source of truth)
├── JOURNEY.md              # This file
├── configs/
│   ├── bci4_lomtev.yaml              # U-Net BCI-IV config
│   ├── fingerflex_lomtev.yaml        # U-Net Miller config
│   ├── fingerflex_tcn.yaml           # TCN config
│   ├── bci4_transformer_*.yaml       # Transformer variants (many)
│   ├── bci4_deepfingernet*.yaml      # DeepFingerNet reproduction
│   ├── bci4_nested_unet_ours.yaml    # NestedUNet with our training
│   └── bci4_dtcnet.yaml              # DTCNet reproduction
├── src/
│   ├── models/
│   │   ├── __init__.py               # Model registry + build_model()
│   │   ├── unet_lomtev.py            # U-Net (Lomtev 2023)
│   │   ├── unet_raw.py               # U-Net for raw ECoG
│   │   ├── tcn.py                    # Temporal Convolutional Network
│   │   ├── transformer.py            # Transformer + Hybrid + Multiscale
│   │   ├── nested_unet.py            # NestedUNet (DeepFingerNet)
│   │   └── dtcnet.py                 # DTCNet (Wang 2025)
│   ├── data/
│   │   ├── __init__.py               # Data routing (build_data)
│   │   ├── load_fingerflex.py        # Miller Library loader
│   │   ├── load_bci4.py              # BCI-IV loader
│   │   ├── preprocessing_lomtev.py   # Morlet + RobustScaler pipeline
│   │   ├── dataset_lomtev.py         # Spectrogram dataset
│   │   └── dataset.py                # Raw ECoG dataset
│   ├── training/
│   │   ├── trainer.py                # Training loop + evaluation
│   │   └── losses.py                 # Loss registry (mse, mse_cosine)
│   └── evaluation/
│       └── metrics.py                # Pearson r, smoothing
├── scripts/
│   ├── train.py                      # Unified entry point
│   ├── grid_transformer_bci4.sh      # Transformer grid search
│   ├── grid_phase6_bci4.sh           # Optimization ablations
│   └── run_all_miller.sh             # 9-patient batch runner
├── papers/                           # Downloaded papers for reference
│   ├── dtcnet_frontiers2025.pdf
│   ├── tao_DeepFingerNet_2025.pdf
│   └── ...
├── data -> /mnt/archive/.../PLaCT_data
└── results -> /mnt/archive/.../PLaCT_data/results
```

---

## What's Next

1. **DTCNet reproduction** (Milestone 5.10): Run corrected DTCNet on BCI-IV S1/S2/S3. Target: mean=0.69.
2. **100% train split experiment**: Train on full official BCI-IV train split (no 15% val carve). Expected +0.04 r.
3. **Sigma=6 test evaluation**: Report σ=6 test numbers for direct comparison to published papers.
4. **Miller 2019 sweep**: If DTCNet or Transformer with 100% train gets strong BCI-IV numbers, run 9-patient Miller sweep.
5. **Multi-seed statistics**: 3-seed sweep for publication-ready error bars.
6. **Phases 6–7**: Joystick and mouse tracking tasks (same 4 patients across both — paired comparison).
7. **Paper writing**: Once we have the full results matrix (3 architectures × 3 tasks × multiple patients).

---

*Last updated: 2026-03-18*
