# Experiment Coordination Log

**Last updated:** 2026-04-04
**Objective:** Beat DTCNet (0.680 repro / 0.690 paper) on BCI-IV with transformer-based architecture

## Baseline (target to beat)
| Model | S1 | S2 | S3 | Mean | Params |
|-------|------|------|------|------|--------|
| Vanilla d1024 L6 h16 chdrop=0.2 | **0.680** | **0.569** | **0.777** | **0.675** | 78M |
| DTCNet (our repro) | 0.696 | 0.598 | 0.747 | 0.680 | 4.5M |
| DTCNet (paper) | 0.71 | 0.59 | 0.77 | 0.690 | — |

## Completed Experiments (2026-04-04)

### Conformer d=512 (L6, h8, ff=2048, chdrop=0.2)
- **Params:** 40M | **Architecture:** Full Conformer (2x FFN + conv module + RoPE)
- S1: 0.628 | S2: 0.501 | S3: 0.716 | **Mean: 0.615**
- **Verdict:** Architecture beats vanilla d=512 (+0.05) but capacity-limited. NOT SOTA.

### Conformer d=1024 (L6, h16, ff=4096, chdrop=0.2) — S3 NOT RUN
- **Params:** 155M | S1: 0.648 | S2: 0.509 | S3: NOT RUN
- **Verdict:** WORSE than vanilla d1024. 155M params → massive overfitting on 33K samples.

### ConvTransformer d=512 (DTCNet conv stem + transformer at T/4)
- **Params:** ~15M | S1: 0.611
- **Verdict:** Worse than Conformer d512 and much worse than baseline. Conv stem alone isn't enough.

### Vanilla d1024 + Data Augmentation (Gaussian noise + amplitude scaling)
- S1: 0.664 (best ~ep38, still declining after)
- **Verdict:** Augmentation HURTS S1. Doesn't help a model that's already well-regularized with chdrop=0.2.

### Vanilla d1024 + SpecAugment
- S1: 0.591
- **Verdict:** SpecAug catastrophically bad for ECoG. Masking frequency bands destroys signal.

## Key Learnings
1. **Full Conformer doubles FFN params** (78M → 155M) — too much for 33K samples.
2. **Conformer's conv module IS helpful** — d512 Conformer beats d512 vanilla by +0.05.
3. **The right approach:** Add ONLY the conv module + RoPE to vanilla transformer (don't double FFN). ~97M params.
4. **Augmentation doesn't help** when model already has chdrop=0.2.
5. **SpecAug kills ECoG** — frequency bands are informative, masking them is destructive.

## Convergence Analysis (d1024 chdrop=0.2 baseline)
- **By epoch 5, model is at 95.6% of final performance.** Epochs 5-50 buy only 4.4%.
- Train loss drops to 4% of ep1 by ep5. Model memorizes training data almost immediately.
- Per-finger oracle (best epoch per finger): S1=0.694, S2=0.578, S3=0.783, **mean=0.685** — beats DTCNet!
- The model CAN decode well enough. Single-checkpoint selection loses ~0.010.
- **Channel dropout scaling trend:**
  - chdrop=0.0: peak ~ep3-5
  - chdrop=0.1: peak ~ep14 (+0.015 mean)
  - chdrop=0.2: peak ~ep15 (+0.030 mean)
  - More chdrop = more productive training epochs = better final result

## chdrop Scaling Results (FULL SWEEPS — FINAL)

| chdrop | S1 | S2 | S3 | Mean | vs DTCNet paper (0.690) |
|--------|------|------|------|------|-------------------------|
| 0.2 (baseline) | 0.680 | 0.569 | 0.777 | 0.675 | -0.015 |
| 0.3 | 0.688 | 0.588 | — | — | — |
| 0.4 | 0.7164 | 0.6054 | **0.7928** | 0.7049 | +0.015 |
| 0.5 | 0.7307 | **0.6235** | 0.7745 | 0.7096 | +0.020 |
| **0.6** | **0.7755** | 0.6223 | 0.7727 | **0.7235** | **+0.034** |
| 0.7 | running on gpu039 | — | — | — | — |

**chdrop=0.6 mean=0.7235 — NEW SOTA, +0.034 over DTCNet paper.**
Per-subject optima: **S1=0.6**, **S2=0.5 (by 0.001)**, **S3=0.4**. Mixed oracle = 0.7306.

### V2 Results (vanilla + RoPE + conv module, ~97M)
| Subject | V2 test_r | vs chdrop=0.4 |
|---------|-----------|---------------|
| S1 | 0.711 | -0.005 |
| S2 | 0.591 | -0.014 |
| S3 | ~0.761 (running, ep45) | -0.032 |

V2 beats baseline but chdrop=0.4 > V2 across the board. **Regularization > architecture.**

## Currently Running
- [gpu003] **Miller chdrop=0.6 sweep started** (Stage 2 per CLAUDE.md)
  - GPU 0: bp (46ch, 610s)
  - GPU 1: mv (43ch, 179s — fastest signal)
  - GPU 2: wc (64ch, 610s)
  - Baseline: U-Net Lomtev mean 0.44 across 9 patients
  - Monitor first 2-3, kill early if worse than baseline (CLAUDE.md workflow)
- [gpu039] **chdrop=0.7 S1** — finding hard ceiling on BCI-IV

## Miller Sweep Plan
1. Round 1 (in progress): bp, mv, wc
2. Round 2 (after): cc, ht, jp (if round 1 beats baseline)
3. Round 3: jc, wm, zt

## Next Steps
- Multi-seed on BCI-IV chdrop=0.6 once Miller sweep finishes
- Per-subject optimal chdrop is an interesting finding — may want a clean ablation
- Clinical framing: "model decodes with any 40% subset of electrodes" is the story
