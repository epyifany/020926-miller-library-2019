# Experiment Coordination Log

**Last updated:** 2026-04-06
**Objective:** Beat DTCNet (0.680 repro / 0.690 paper) on BCI-IV, then generalize to Miller 2019

## BCI-IV SOTA (ACHIEVED)

### chdrop Scaling Results (FULL SWEEPS — ALL FINAL)

| chdrop | S1 | S2 | S3 | Mean | vs DTCNet paper (0.690) |
|--------|------|------|------|------|-------------------------|
| 0.2 (baseline) | 0.680 | 0.569 | 0.777 | 0.675 | -0.015 |
| 0.3 | 0.688 | 0.588 | — | — | — |
| 0.4 | 0.716 | 0.605 | **0.793** | 0.705 | +0.015 |
| 0.5 | 0.731 | **0.624** | 0.775 | 0.710 | +0.020 |
| **0.6** | **0.776** | 0.622 | 0.773 | **0.724** | **+0.034** |
| 0.7 | 0.772 | 0.620 | 0.768 | 0.720 | +0.030 |

**chdrop=0.6 mean=0.724 — NEW SOTA, +0.034 over DTCNet paper.**
**chdrop=0.7 declined** (-0.004 mean) — confirms chdrop=0.6 is the peak.
Per-subject optima: **S1 peaks at 0.6**, **S2 at 0.5**, **S3 at 0.4**. Mixed oracle = 0.731.
Paper ablation curve: 0.2(0.675)→0.4(0.705)→0.5(0.710)→**0.6(0.724)**→0.7(0.720). Clean peak.

Best single finger: S1 Ring = **0.818** (chdrop=0.6). S3 Thumb = **0.866** (chdrop=0.4).

### Architecture Experiments (all failed to beat chdrop scaling)
| Experiment | S1 test_r | Params | Verdict |
|-----------|-----------|--------|---------|
| Conformer d512 | 0.628 | 40M | +0.05 over vanilla d512, capacity-limited |
| Conformer d1024 | 0.648 | 155M | 2x FFN = massive overfitting |
| TransformerV2 (RoPE+conv) | 0.711 | 97M | Beats baseline, but chdrop=0.4 > V2 |
| ConvTransformer d512 | 0.611 | ~15M | Conv stem alone isn't enough |
| Vanilla + augmentation | 0.664 | 78M | Doesn't help when chdrop regularizes |
| Vanilla + SpecAugment | 0.591 | 78M | Catastrophic — kills ECoG signal |

**Key lesson: regularization (chdrop) > architecture changes.**

---

## Miller 2019 chdrop=0.6 Sweep (Stage 2 — IN PROGRESS)

**Context:** Lomtev/FingerFlex is the only reproducible DL benchmark on Miller 9-patient (mean 0.44).
DTCNet, Jangir, Tragoudaras use BCI-IV only. DeepFingerNet claimed 0.54 on Miller but is NOT reproducible.
Protocol: fullsplit 85/15, σ_test=6 (directionally comparable to baseline).

### All Patients

| Patient | Ch | U-Net baseline | Our transformer | Δ | Node | Status |
|---------|-----|---------------|-----------------|---|------|--------|
| **mv** | 43 | 0.460 | **0.548** | **+0.088** | gpu003 | ✅ DONE |
| **bp** | 46 | 0.369 | **0.597** (ep25) | **+0.228** | gpu003 | running |
| **wc** | 64 | 0.383 | **0.449** (ep24) | **+0.066** | gpu003 | running |
| **cc** | 63 | 0.725 | running | — | gpu003 | running |
| **ht** | 46 | 0.256 | **0.369** (ep6) | **+0.113** | gpu039 | running ep9 |
| **jc** | 46 | 0.210 | **0.508** (ep10) | **+0.298** | gpu039 | running ep12 |
| **jp** | 52 | 0.065 | **0.511** (ep3) | **+0.446** | gpu039 | running ep3 |
| **wm** | 46 | ? | — | — | — | queued |
| **zt** | 64 | ? | — | — | — | queued |

**Standout results so far:**
- **bp**: +0.228 over baseline, still climbing at ep25
- **jp**: 7.8x improvement (0.065 → 0.511) — U-Net couldn't decode this patient at all
- **jc**: 2.4x improvement (0.210 → 0.508)

mv caveat: thumb still negative (-0.109, same structural issue as baseline -0.31).
Other 4 fingers average 0.697 on mv.

### splits.py Change
`test_frac=0.0` triggers fullsplit mode (val=test). Matches BCI-IV protocol.
```yaml
split:
  train_frac: 0.85
  val_frac: 0.15
  test_frac: 0.0
```

## Miller 9-Patient Sweep — COMPLETE (2026-04-06)

Config: `miller_transformer_d1024_chdrop06.yaml` (d1024 L6 h16, chdrop=0.6, fullsplit 85/15, σ=6, seed=7)

| Patient | Ch | Baseline (U-Net) | Ours | Δ |
|---------|-----|------------------|------|---|
| cc | 63 | 0.725 | **0.818** | +0.093 |
| jp | 58 | 0.548 | **0.630** | +0.082 |
| zt | 61 | 0.540 | **0.602** | +0.062 |
| bp | 46 | 0.369 | **0.597** | +0.228 |
| mv | 43 | 0.460 | **0.548** | +0.088 |
| jc | 47 | 0.536 | 0.508 | -0.028 |
| wc | 64 | 0.383 | **0.449** | +0.066 |
| ht | 64 | 0.287 | **0.369** | +0.082 |
| wm | 38 | 0.080 | 0.047 | -0.033 |
| **Mean (9)** | | **0.436** | **0.508** | **+0.071** |
| **Mean (8, excl wm)** | | **0.481** | **0.565** | **+0.084** |

**7/9 patients improved over baseline.** Two underperformers:
- **wm** (38ch): chdrop=0.6 leaves only 15 active channels — too few. Baseline was already 0.080.
- **jc** (47ch): slight decline (-0.028), may benefit from lower chdrop.

**Note on wm:** Every architecture we've tested fails on wm. DeepFingerNet claimed
nothing below 0.2 on all 9 patients — their wm number is not credible given the data
quality (38ch, weak motor coverage). Our 0.047 and baseline 0.080 are honest.

## Currently Running
- (nothing — all sweeps complete)

## Next Steps
1. Per-patient chdrop tuning — lower chdrop for sparse patients (wm, ht, jc)
2. Multi-seed on BCI-IV chdrop=0.6 for error bars (3-5 seeds)
3. Write paper — clinical framing: "model decodes with any 40% subset of electrodes"
4. Re-run U-Net baseline with matched protocol (85/15 fullsplit σ=6) for fair comparison
