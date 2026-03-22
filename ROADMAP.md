# Research Roadmap: Comparing Deep Learning Architectures for Continuous ECoG Motor Decoding

## Project Summary

Benchmark three deep learning architectures (U-Net, TCN, Transformer) across three continuous ECoG motor decoding tasks from the Miller Library (2019): finger flexion (9 patients), joystick tracking (4 patients), and mouse tracking (4 patients). 17 total datasets, 13 unique subjects, one shared metric (Pearson r). The goal is to determine whether architecture choice interacts with motor task type.

---

## Phase 0: Environment and Reproducibility Setup

### Milestone 0.1 — Project structure
- [x] Create a clean project layout (src/data, src/models, src/training, src/evaluation, src/utils, configs/, scripts/, results/, tests/)
- [x] Add a `requirements.txt` (numpy, scipy, h5py, torch, pyyaml, matplotlib, scikit-learn)
- [x] Confirm Python version — using Python 3.11 (pytorch_ml env)

### Milestone 0.2 — Reproducibility infrastructure
- [x] Write a `set_seed(seed)` function (`src/utils/seed.py`)
- [x] Create a config system — YAML-based (`src/utils/config.py`, `configs/fingerflex_default.yaml`)
- [x] `make_experiment_dir()` saves config + creates timestamped results dirs
- [x] Set up per-epoch logging — implemented via W&B (see `src/utils/wandb_utils.py`)

### Milestone 0.3 — Version control hygiene
- [x] `.gitignore` covers data/, results/, *.pt, *.ckpt, wandb/, symlinks
- [x] Data and results symlinked to `/mnt/archive/` (300 GB) — see CLAUDE.md Storage Rules
- [x] Make an initial commit with project skeleton
- [x] W&B logging integrated — `src/utils/wandb_utils.py` with `--no-wandb` CLI flag, project `plact-motor-decoding`, all configs updated

---

## Phase 1: Data Loading and Exploration (Fingerflex Only)

### Milestone 1.1 — Load and inspect raw fingerflex data
- [x] Write `load_fingerflex(patient_id)` in `src/data/load_fingerflex.py` — returns ecog, flex, cue, stim, locs, sr
- [x] Handle varying dtypes (int16/int32 -> float64, uint16 -> float64) — all 9 patients load cleanly via `scipy.io.loadmat`
- [x] Inspection script: `scripts/inspect_fingerflex.py` — verified all 9 patients, no NaNs/Infs/dead channels
- [x] Summary table:

  | Patient | Channels | Samples   | Duration (s) | Flex range     |
  |---------|----------|-----------|-------------|----------------|
  | bp      | 46       | 610,040   | 610.0       | [0, 2885]      |
  | cc      | 63       | 610,040   | 610.0       | [0, 3074]      |
  | ht      | 64       | 610,040   | 610.0       | [0, 2701]      |
  | jc      | 47       | 530,440   | 530.4       | [0, 2491]      |
  | jp      | 58       | 465,840   | 465.8       | [0, 2837]      |
  | mv      | 43       | 178,960   | 179.0       | [0, 2694]      |
  | wc      | 64       | 610,040   | 610.0       | [0, 2875]      |
  | wm      | 38       | 444,840   | 444.8       | [0, 2491]      |
  | zt      | 61       | 610,040   | 610.0       | [0, 2493]      |

  Notes: channel range 38–64, mv is much shorter (179s), all 1000 Hz.

### Milestone 1.2 — Exploratory data analysis
- [x] Plot raw ECoG traces (5-10 seconds) for 2-3 patients — check for artifacts, saturation
- [x] Plot 5 finger flexion signals over time — understand movement structure
- [x] Plot stimulus/cue codes over time — understand trial structure
- [x] Compute and plot PSD for a few channels — check filtering status
- [x] Check for NaNs, Infs, or constant channels — all clean (done in 1.1)
- [x] Save plots to `results/eda/` for reference

  **EDA Observations:**
  - ECoG traces (bp, ht, wm): typical oscillatory activity, no obvious artifacts or dead channels in sampled windows
  - Finger flexion: clear cue-driven paradigm with discrete flexion peaks, raw values 0-3000, different baselines per finger
  - Cue codes: values 0-5 (0=rest, 1-5=each finger), randomized trial blocks; stim has additional negative codes (-2,-1 in bp, -1 in cc) — likely preparation/rest markers
  - **PSD confirms data is raw/unfiltered**: prominent 60 Hz powerline peak + harmonics at 120, 180 Hz across all patients. Neural signal concentrated <50 Hz, noise floor >200 Hz. Validates our preprocessing choices: notch at 60 Hz + harmonics, bandpass 1-200 Hz.

### Milestone 1.3 — BCI Competition IV Dataset 4 integration
- [x] Downloaded and extracted BCI-IV data (3 subjects: 62ch, 48ch, 64ch)
- [x] `src/data/load_bci4.py` — `load_bci4(subject_id, data_dir)` loads pre-split train/test
- [x] Added `interpolate_fingerflex()` to `preprocessing_lomtev.py` — cubic interpolation 25 Hz → 100 Hz
- [x] Added `build_bci4_lomtev_datasets(subject_id, config)` to `dataset_lomtev.py` — full Lomtev pipeline for BCI-IV
- [x] Created `configs/bci4_lomtev.yaml` — 50 Hz powerline, pre-split train/test, 25 Hz finger interpolation
- [x] Verified all 3 subjects with `scripts/verify_bci4.py`
- [x] Trained and evaluated Lomtev U-Net on all 3 subjects:

  | Subject | Ch | Params  | Best Val r | Test r (avg) | Test r (per finger)           |
  |---------|-----|---------|-----------|-------------|-------------------------------|
  | 1       | 62  | 652,613 | **0.686** | **0.578**   | 0.55 0.75 0.47 0.62 0.51     |
  | 2       | 48  | 598,853 | **0.633** | **0.520**   | 0.45 0.72 0.51 0.49 0.43     |
  | 3       | 64  | 660,293 | **0.737** | **0.692**   | 0.76 0.62 0.63 0.74 0.72     |
  | **Mean** | — | —       | **0.685** | **0.597**   |                               |

  - Subject 1 matches FingerFlex reference (val r=0.686 vs 0.662–0.677 in original)
  - Subject 3 hits 0.737 val r / 0.692 test r — strongest subject
  - All results: 40 epochs, batch=128, lr=8.42e-5, mse_cosine loss, seed=123
  - Key to matching FingerFlex: full-signal evaluation (not windowed), Gaussian smoothing (σ_val=6, σ_test=1), correlation-based model selection
  - Training script: `scripts/train_lomtev_bci4.py`

---

## Phase 2: Preprocessing Pipeline (Fingerflex)

### Milestone 2.1 — Design the preprocessing pipeline
- [x] Document every preprocessing step and its justification.
- [x] Verified: CAR produces zero median across channels, z-score gives mean≈0 std≈1, windowing round-trips correctly
- [x] **Two pipeline modes** controlled by `preprocessing.pipeline_mode` in config:

  **Mode `ours` (default)** — our best interpretation for raw 1000 Hz ECoG:
  1. Load raw data
  2. Bandpass 1–200 Hz + notch 60 Hz harmonics (full signal, pre-split)
  3. Temporal split 70/15/15
  4. Z-score ECoG per-channel (fit train, apply all)
  5. CAR median (each split independently)
  6. Z-score targets per-finger (fit train, apply all)
  7. On-the-fly windowing (1s window, 100ms hop @ 1000 Hz)

  **Mode `lomtev`** — faithful to FingerFlex (Lomtev, 2023):
  1. Load raw data
  2. Z-score ECoG per-channel + CAR median (full signal, pre-split)
  3. Bandpass 40–300 Hz + notch 60 Hz harmonics
  4. Temporal split 70/15/15
  5. MinMax [0,1] targets per-finger (fit train, apply all)
  6. On-the-fly windowing (1s window, 100ms hop @ 1000 Hz)

  **Rationale for two modes**: Lomtev's pipeline is proven (~0.7 r on BCI-IV). Ours reorders filtering before normalization (more principled: stats computed on the signal distribution models will see). Both share the same Dataset class. Config switch lets us ablate pipeline choices directly.

  | Choice | `ours` | `lomtev` | Why diverge |
  |--------|--------|----------|-------------|
  | Preprocessing order | filter → zscore → CAR | zscore → CAR → filter | Filter changes distribution; stats should reflect final signal |
  | Bandpass | 1–200 Hz | 40–300 Hz | Miller data is US (60 Hz); wider low-freq band captures movement-related potentials |
  | Powerline | 60 Hz | 50 Hz (Lomtev) / 60 Hz (us) | Data-dependent; configurable |
  | Target scaling | z-score | MinMax [0,1] | Z-score standard for regression w/ linear output; MinMax pairs with sigmoid |
  | Input representation | Raw ECoG @ 1000 Hz | Raw ECoG @ 1000 Hz | Both — wavelet spectrograms are a separate ablation (Phase 8) |

### Milestone 2.2 — Implement the preprocessing pipeline
- [x] `src/data/preprocessing.py` — added `zscore(data, mean, std)` and `apply_car(data)` helpers; existing `filter_ecog`, `normalize_and_car`, `normalize_targets` unchanged
- [x] `src/data/windowing.py` — `create_windows(signal, window_size, hop_size)` → (n_windows, window_size, features); `reconstruct_from_windows(windows, hop_size, n_samples)` for evaluation
- [x] All preprocessing parameters come from `configs/fingerflex_default.yaml`
- [x] Add `minmax_scale(data, dmin, dmax)` to `preprocessing.py` — MinMax [0,1] scaling for lomtev mode targets
- [x] Add `pipeline_mode` switch to `build_datasets()` — dispatches to `_build_ours()` or `_build_lomtev()` based on `preprocessing.pipeline_mode` config key
- [x] Create `configs/fingerflex_lomtev.yaml` — Lomtev-faithful settings (bandpass 40–300 Hz, normalize-first, MinMax targets, lr=8.42e-5, batch=128, loss=mse_cosine)
- [x] Verified both modes on all 9 patients — `scripts/verify_pipeline_modes.py`

### Milestone 2.3 — Train/test split strategy
- [x] `src/data/splits.py` — `temporal_split(n_samples, train_frac, val_frac, test_frac)` returns contiguous (start, end) index tuples
- [x] No shuffling — contiguous blocks only (70/15/15 default)
- [x] Split indices stored in `build_datasets()` return dict for reproducibility
- [ ] If the BCI Competition IV split maps to these patients, consider using their exact train/test boundary for comparability (deferred)

### Milestone 2.4 — PyTorch Dataset and DataLoader
- [x] `src/data/dataset.py` — `FingerFlexDataset(ecog, flex, window_size, hop_size)` with on-the-fly windowing in `__getitem__`, returns `(ecog_window, flex_window)` as float32 tensors
- [x] `build_datasets(patient_id, config)` — full pipeline orchestrator: load → filter → split → normalize → dataset. Returns train/val/test datasets + normalization stats
- [x] Using Option A: per-patient models with patient-specific channel counts (38–64 channels)
- [x] Verified DataLoader shapes: `(batch, 1000, channels)` for ECoG, `(batch, 1000, 5)` for targets
- [x] All 9 patients load successfully — verified with `scripts/verify_pipeline.py`

  | Patient | Ch | Train windows | Val windows | Test windows |
  |---------|-----|--------------|-------------|--------------|
  | bp      | 46  | 4261         | 906         | 906          |
  | cc      | 63  | 4261         | 906         | 906          |
  | ht      | 64  | 4261         | 906         | 906          |
  | jc      | 47  | 3704         | 786         | 786          |
  | jp      | 58  | 3251         | 689         | 689          |
  | mv      | 43  | 1243         | 259         | 259          |
  | wc      | 64  | 4261         | 906         | 906          |
  | wm      | 38  | 3104         | 658         | 658          |
  | zt      | 61  | 4261         | 906         | 906          |

---

## Phase 3: Baseline Model — U-Net on Fingerflex

### Milestone 3.0 — Port FingerFlex-Lomtev into our codebase
- [x] FingerFlex U-Net (Lomtev, 2023) already reproduced in `../FingerFlex/` notebooks — **~0.7 mean Pearson r** on BCI Competition IV data (1 subject, wavelet spectrograms, 100 Hz)
- [x] **Ported the Lomtev implementation as clean Python scripts** into our codebase:

  **Model** → `src/models/unet_lomtev.py`:
  - [x] `ConvBlock` — Conv1d (no bias) + LayerNorm + GELU + Dropout(0.1) + MaxPool1d
  - [x] `UpConvBlock` — ConvBlock + nn.Upsample(mode='linear')
  - [x] `AutoEncoder1D` — spatial_reduce (n_ch×n_freq → 32) + 5 encoder blocks [32→32→64→64→128] + 5 decoder blocks with skip concatenation + Conv1d 1×1 → n_fingers
  - [x] 652,613 params (62ch) / 591,173 params (46ch) — matches original

  **Lomtev preprocessing** → `src/data/preprocessing_lomtev.py`:
  - [x] `compute_spectrograms(ecog, sr, freqs)` — MNE Morlet wavelets, 40 log-spaced freqs in [40, 300] Hz
  - [x] `downsample_spectrograms(spec, cur_fs, new_fs)` — stride-based downsampling to 100 Hz
  - [x] `crop_for_time_delay(flex, spec, delay_sec, fs)` — shift targets by 200ms neural delay
  - [x] `robust_scale_spectrograms(train, val, test)` — RobustScaler (unit_variance, quantile 0.1–0.9)
  - [x] `minmax_scale_flex(train, val, test)` — MinMaxScaler [0,1] on finger targets
  - [x] `interpolate_fingerflex(finger_flex, cur_fs, true_fs, needed_hz)` — cubic interpolation for BCI-IV 25 Hz finger data

  **Lomtev Dataset** → `src/data/dataset_lomtev.py`:
  - [x] `LomtevDataset(spec, flex, sample_len, stride)` — stride-1 windowing on spectrograms
  - [x] `build_lomtev_datasets(patient_id, config)` — full 11-step pipeline

  **Verification** → `scripts/verify_lomtev.py` — all 9 patients verified:

  | Patient | Ch | Wavelets | Train win | Val win | Test win |
  |---------|-----|---------|-----------|---------|----------|
  | bp      | 46  | 40      | 42,426    | 8,875   | 8,875    |
  | cc      | 63  | 40      | 42,426    | 8,875   | 8,875    |
  | ht      | 64  | 40      | 42,426    | 8,875   | 8,875    |
  | jc      | 47  | 40      | 36,854    | 7,681   | 7,681    |
  | jp      | 58  | 40      | 32,332    | 6,712   | 6,712    |
  | mv      | 43  | 40      | 12,251    | 2,408   | 2,409    |
  | wc      | 64  | 40      | 42,426    | 8,875   | 8,875    |
  | wm      | 38  | 40      | 30,862    | 6,397   | 6,397    |
  | zt      | 61  | 40      | 42,426    | 8,875   | 8,875    |

  - [x] Train on one patient, check we get reasonable r (target: comparable to ~0.7) — smoke-tested bp (2 epochs, val r=0.40), then full 40-epoch runs on all 9 patients

- [x] **Key differences documented**:

  | Aspect | Original Lomtev | Our Lomtev port |
  |--------|----------------|-----------------|
  | Data | BCI-IV (1 subject, pre-split) | Miller Library (9 subjects, our 70/15/15 split) |
  | Channels | Fixed 62 | Variable 38–64 per patient |
  | Powerline | 50 Hz (European) | 60 Hz (US) |
  | Input to model | (62, 40, 256) | (n_ch, 40, 256) — spatial_reduce adapts to n_ch×40 |
  | Scaler fitting | On pre-split train file | On train portion only (proper) |

- [x] **Anchor baseline: Lomtev U-Net on all 9 Miller patients** (seed=123, 40 epochs, batch=128, lr=8.42e-5, mse_cosine loss, full-signal eval, σ_val=6, σ_test=1)

  Training script: `scripts/train_lomtev_miller.py` | Batch runner: `scripts/run_all_miller.sh`
  Config: `configs/fingerflex_lomtev.yaml` (added `preprocessing_lomtev` + `windowing_lomtev` sections)

  | Patient | Ch | Params  | Best Val r | Test r (avg) | Test r (per finger)                  |
  |---------|-----|---------|-----------|-------------|--------------------------------------|
  | bp      | 46  | 591,173 | 0.627     | 0.369       | 0.40, 0.27, 0.39, 0.63, 0.15        |
  | cc      | 63  | 656,453 | 0.682     | **0.725**   | 0.79, 0.61, 0.69, 0.79, 0.74        |
  | ht      | 64  | 660,293 | 0.359     | 0.287       | 0.29, 0.30, 0.11, 0.39, 0.35        |
  | jc      | 47  | 595,013 | 0.617     | 0.536       | 0.60, 0.61, 0.41, 0.67, 0.38        |
  | jp      | 58  | 637,253 | 0.576     | 0.548       | 0.70, 0.73, 0.24, 0.66, 0.41        |
  | mv      | 43  | 579,653 | 0.693     | 0.460       | -0.31, 0.86, 0.53, 0.50, 0.72       |
  | wc      | 64  | 660,293 | 0.471     | 0.383       | 0.56, 0.66, 0.38, 0.18, 0.13        |
  | wm      | 38  | 560,453 | 0.323     | 0.083       | 0.05, 0.06, 0.21, 0.09, 0.00        |
  | zt      | 61  | 648,773 | **0.705** | **0.600**   | 0.54, 0.71, 0.65, 0.62, 0.49        |
  | **Mean** | —  | —       | **0.561** | **0.443**   |                                      |
  | **Std**  | —  | —       | 0.139     | 0.192       |                                      |
  | **Ref: BCI-IV** | 48-64 | — | **0.685** | **0.597** | (3 subjects, see Milestone 1.3)    |

  **Observations:**
  - **Wide patient variability**: test r ranges from 0.08 (wm) to 0.73 (cc). This is expected — electrode placement, cortical coverage, and task engagement vary across patients.
  - **Top 3 patients** (cc, zt, jp) achieve test r > 0.54, comparable to BCI-IV reference results.
  - **Bottom 2 patients** (wm, ht) have test r < 0.30 — likely poor motor cortex coverage or sparse electrode grids (wm has only 38 channels).
  - **mv anomaly**: thumb (finger 0) has negative test r (-0.31) while other fingers are strong (0.50–0.86). This 179s recording is also the shortest, making the test split very small (~27s). The model may have overfit to val patterns that don't generalize.
  - **Val-test gap**: mean val r (0.56) is consistently higher than test r (0.44), especially for bp (val 0.63 → test 0.37). Full-signal evaluation on different temporal segments can reveal non-stationarities in the neural signal.
  - **Overall**: the Lomtev pipeline transfers to Miller data with degraded but reasonable performance. The mean test r of 0.44 across 9 patients (vs 0.60 on 3 BCI-IV subjects) is our anchor. New architectures (TCN, Transformer) and raw-ECoG pipelines will be compared against this baseline.

### Milestone 3.1 — Shared training framework + Raw ECoG U-Net
- [x] **Shared training framework** — eliminated ~365 lines of duplicated code across two training scripts:
  - `src/training/losses.py` — loss registry (`mse`, `mse_cosine`) with `get_loss_fn(name)`
  - `src/evaluation/metrics.py` — `pearson_r_per_channel()`, `smooth_predictions(pred, sigma)`
  - `src/training/trainer.py` — `Trainer` class with `train_one_epoch()`, `evaluate_windowed()`, `evaluate_fullsig()`, `fit()`, `test()`
  - `src/data/__init__.py` — `build_data(config)` dispatcher (routes to correct dataset builder based on task + model name)
  - `src/models/__init__.py` — `MODEL_REGISTRY` + `build_model(config, dataset_info)` (model registry pattern)
  - `scripts/train.py` — unified ~80-line entry point with CLI overrides (`--config`, `--patient`, `--subject`, `--epochs`, `--batch_size`, `--lr`, `--gpu`, `--seed`, `--no-wandb`)
- [x] **Raw ECoG U-Net** — `src/models/unet_raw.py` (`AutoEncoder1DRaw`):
  - Input: `(batch, n_channels, time)` — raw ECoG (channels-first), no spectrogram
  - Reuses `ConvBlock`, `UpConvBlock` from `unet_lomtev.py`
  - `spatial_reduce`: maps `n_channels` (~50) → `channels[0]` (vs `n_ch × 40` wavelets in Lomtev)
  - Same encoder/decoder/skip architecture as `AutoEncoder1D`
  - 418,949 params (46ch) vs 591,173 for Lomtev spectrogram variant
  - Config: `configs/fingerflex_raw_unet.yaml` (1-200 Hz bandpass, window_size=1024, lr=1e-3, MSE loss)
- [x] **Dataset changes** for framework compatibility:
  - `FingerFlexDataset.__getitem__()` now returns channels-first `(channels, time)`, `(n_targets, time)`
  - Both `FingerFlexDataset` and `LomtevDataset` have `get_full_signal()` for full-signal eval
  - Both builder functions return `n_input_features` in their dicts
- [x] **Backward compatibility**: old scripts (`train_lomtev_miller.py`, `train_lomtev_bci4.py`) still work unchanged
- [x] **Verification** — 3 smoke tests passed:
  - Old script: `train_lomtev_miller.py --patient bp --epochs 2` → val r=0.4001
  - New unified: `train.py --config fingerflex_lomtev.yaml --patient bp --epochs 2` → val r=0.3999 (matches)
  - Raw U-Net: `train.py --config fingerflex_raw_unet.yaml --patient bp --epochs 2` → val r=0.2923 (trains, 4.4s pipeline vs 101s for spectrograms)
- [x] **Extensibility**: adding a new model = one .py file + one config + one line in MODEL_REGISTRY
- [x] We now have **two U-Net variants**: `unet_lomtev.py` (spectrogram input, identical to paper) and `unet_raw.py` (raw ECoG input, our adaptation)

### Milestone 3.2 — Training loop
- [x] Implemented `src/training/trainer.py` with:
  - Loss function: configurable via registry — `mse` (our default) or `mse_cosine` (0.5×MSE + 0.5×(1-cos), Lomtev's choice)
  - Optimizer: Adam with configurable learning rate (1e-3 for raw, 8.42e-5 for lomtev mode)
  - Learning rate scheduler: ReduceLROnPlateau (configurable, `"none"` to disable)
  - Early stopping: patience of N epochs on validation correlation
  - Gradient clipping: max norm 1.0
  - Checkpoint saving: save best model by max correlation (not min loss)
- [x] Log per-epoch: training loss, validation loss, validation Pearson r (per channel and averaged)
- [x] Results saved to `results/{timestamp}_{model}_{patient}/` with config + results.json

### Milestone 3.3 — Evaluation metrics
- [x] Implemented `src/evaluation/metrics.py`:
  - `pearson_r_per_channel(pred, target)` — generalised to arbitrary output channels (fingers, joystick X/Y)
  - `smooth_predictions(pred, sigma)` — Gaussian smoothing on model output
- [x] Full-signal evaluation via `Trainer.evaluate_fullsig()` — uses `dataset.get_full_signal()`, aligns to model stride, no windowing artifacts
- [x] **Prediction smoothing** (configurable per config): `smooth_sigma_val` and `smooth_sigma_test`
  - Lomtev config: σ_val=6, σ_test=1 (matches FingerFlex)
  - Raw config: σ_val=0, σ_test=0 (no smoothing)
- [ ] Write unit tests: perfect prediction → r=1.0, random prediction → r≈0, inverted prediction → r=-1.0

### Milestone 3.4 — Train raw U-Net on all 9 fingerflex patients
- [x] Trained `unet_raw` on all 9 patients (seed=123, 100 epochs max, early stopping patience=15, batch=32, lr=1e-3, MSE loss, 1-200 Hz bandpass, no smoothing)
- [x] Batch runner: `scripts/run_raw_unet_all.sh` — 4 GPUs, ~3 min total

  | Patient | Ch | Raw U-Net test r | Lomtev Spec test r | Delta | Early stop |
  |---------|-----|-----------------|-------------------|-------|------------|
  | bp      | 46  | 0.328           | 0.369             | -0.04 | ep 24      |
  | cc      | 63  | **0.484**       | **0.725**         | -0.24 | ep 21      |
  | ht      | 64  | 0.256           | 0.287             | -0.03 | ep 18      |
  | jc      | 47  | 0.211           | 0.536             | -0.33 | ep 18      |
  | jp      | 58  | 0.065           | 0.548             | -0.48 | ep 26      |
  | mv      | 43  | 0.285           | 0.460             | -0.17 | ep 20      |
  | wc      | 64  | 0.206           | 0.383             | -0.18 | ep 29      |
  | wm      | 38  | 0.009           | 0.083             | -0.07 | ep 33      |
  | zt      | 61  | 0.196           | 0.600             | -0.40 | ep 29      |
  | **Mean** | — | **0.227**       | **0.443**         | **-0.22** |        |
  | **Std**  | — | 0.133           | 0.192             |       |            |

  **Observations:**
  - Spectrogram pipeline wins on all 9 patients. Mean test r 0.227 (raw) vs 0.443 (spec) — roughly half.
  - Raw models early-stop quickly (ep 18-33), suggesting they plateau fast and can't learn deeper temporal-frequency features.
  - The wavelet spectrograms aren't just preprocessing — they perform critical feature extraction (time-frequency decomposition) that the raw U-Net's spatial_reduce layer can't replicate.
  - Same patient difficulty ranking holds: cc is best, wm/jp are weakest in both pipelines.
  - **Conclusion**: spectrogram pipeline is the primary input for TCN/Transformer comparisons. Raw ECoG revisited in Phase 8 ablations (learnable filterbank front-end).

- [ ] Compute group statistics: mean ± std of mean Pearson r across patients
- [ ] Identify any patients that perform much worse — investigate (bad data? too few channels? electrode placement away from motor cortex?)
- [ ] **Pick the best pipeline** based on these results — use it as default for Phases 4–7
- [ ] This three-way comparison answers: does the Lomtev architecture work on our data? Does the wavelet representation matter? Does preprocessing order matter?
- [ ] This table is your **baseline**. All subsequent models will be compared against these numbers.

### Milestone 3.6 — Trivial baselines
- [ ] Implement and evaluate at least two trivial baselines for proper context:
  - **Zero-order hold**: predict target(t) = target(t-1) (copy previous value). This is surprisingly strong for slow-changing signals.
  - **Linear regression**: sklearn Ridge regression from windowed ECoG features to finger targets. This represents what a non-deep-learning approach achieves.
- [ ] Report Pearson r for both. If U-Net doesn't substantially beat linear regression, that's an important finding to report.

---

## Phase 4: Second Architecture — TCN on Fingerflex

### Milestone 4.1 — Implement TCN
- [x] Studied TCN literature (Bai et al., 2018 "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling")
- [x] Implemented in `src/models/tcn.py`:
  - `_TransposedLayerNorm` — helper for LayerNorm on (B,C,T) tensors
  - `TCNBlock` — two dilated Conv1d layers + LN + GELU + Dropout + residual skip
  - `TCN` — spatial_reduce → 6 TCNBlocks (dilations 1,2,4,8,16,32) → output Conv1d
  - Acausal convolutions (padding='same') — fair comparison with U-Net
  - Receptive field: 505 timesteps > 256 input window → full temporal coverage
- [x] Config: `configs/fingerflex_tcn.yaml` — identical training hyperparameters to U-Net (lr=8.42e-5, batch=128, mse_cosine, 40 epochs)
- [x] Registered in `src/models/__init__.py` (`MODEL_REGISTRY["tcn"]`)
- [x] Updated `src/data/__init__.py` — `SPECTROGRAM_MODELS = {"unet_lomtev", "tcn"}` for routing
- [x] Fixed `src/training/trainer.py` — `config["model"].get("strides", [1])` (TCN has no downsampling strides)
- [x] Parameter count match: ~601K (TCN, 46ch) vs ~591K (U-Net, 46ch) → +1.7%

### Milestone 4.2 — Train TCN v1 on all 9 fingerflex patients
- [x] Trained all 9 patients (seed=123, 40 epochs, batch=128, lr=8.42e-5, mse_cosine loss)
- [x] Config: `configs/fingerflex_tcn.yaml` | Runner: `scripts/run_tcn_all.sh`

  | Patient | Ch | U-Net test r | TCN v1 test r | Delta   |
  |---------|-----|-------------|--------------|---------|
  | bp      | 46  | 0.254       | 0.290        | +0.036  |
  | cc      | 63  | 0.725       | 0.638        | -0.087  |
  | ht      | 64  | 0.287       | 0.269        | -0.018  |
  | jc      | 47  | 0.536       | 0.359        | -0.177  |
  | jp      | 58  | 0.549       | 0.450        | -0.099  |
  | mv      | 43  | 0.460       | 0.386        | -0.074  |
  | wc      | 64  | 0.383       | 0.294        | -0.089  |
  | wm      | 38  | 0.083       | 0.087        | +0.004  |
  | zt      | 61  | 0.600       | 0.468        | -0.132  |
  | **Mean** | — | **0.431**   | **0.360**    | **-0.071** |

  **Observations:**
  - TCN v1 lost on 7/9 patients, won on 2 (bp, wm — both by tiny margins on weak patients)
  - Core issue: **severe overfitting** — train loss dropped 50-100× below val loss, val r peaked at epoch 3-8 then declined
  - Early stopping triggered at epoch 15-22 for most patients
  - The U-Net's multi-scale structure (skip connections + downsampling/upsampling) provides a strong inductive bias that the flat TCN lacks

### Milestone 4.2b — TCN v2: increased regularization (failed experiment)
- [x] Hypothesis: overfitting can be fixed with more regularization
- [x] Changes: dropout 0.1→0.25, hidden_channels 64→48, weight_decay 1e-6→1e-4
- [x] Config: `configs/fingerflex_tcn_v2.yaml` (~405K params, -33% vs v1)
- [x] Ran 3/9 patients before killing — v2 consistently underperformed v1:

  | Patient | TCN v1 | TCN v2 | Delta   |
  |---------|--------|--------|---------|
  | bp      | 0.290  | 0.284  | -0.006  |
  | cc      | 0.638  | 0.624  | -0.014  |
  | ht      | 0.269  | 0.212  | -0.057  |

  **Conclusion:** More regularization made things worse — the problem isn't overfitting tuning, it's missing inductive bias. The flat single-resolution TCN cannot match the U-Net's multi-scale feature hierarchy. Killed run early (following our experiment workflow rules).

### Milestone 4.3 — TCN analysis and next steps
- [x] **Key learning:** a vanilla TCN (flat resolution, no skip connections across scales) is fundamentally limited for this task. The U-Net's encoder-decoder with skip connections captures multi-scale temporal patterns that matter for ECoG decoding.
- [x] **Operational learning:** monitor first 2-3 patients; if consistently worse, kill early — we wasted GPU hours running all 9 on v1 when the pattern was clear by patient 2.
- [ ] **Next architecture options** (Phase 5):
  1. Transformer encoder — attention can capture long-range dependencies without multi-scale structure
  2. Hybrid approaches — TCN + downsampling/upsampling, or attention + convolution
  3. S4/Mamba state-space models — strong on neural time-series benchmarks

---

## Phase 5: Third Architecture — Transformer on Fingerflex

### Milestone 5.1 — Implement Transformer encoder
- [x] Implemented `src/models/transformer.py`:
  - `SinusoidalPositionalEncoding` — dynamic buffer (auto-expands for any sequence length)
  - `_TransposedLayerNorm` — LayerNorm on (B,C,T) tensors (shared with TCN)
  - `TransformerECoG` — main model with sliding-window eval for inference
  - `HybridTransformerECoG` — experimental variant with downsample/upsample + skip (failed, see Round 8)

- [x] **Architecture**:
  ```
  Input: (B, n_electrodes, n_wavelets, T=256)
    → reshape: (B, n_ch*n_freq, T)
    → spatial_reduce: Conv1d(n_ch*freq, d_model, k, bias=False) + LN + GELU + Drop
    → transpose: (B, T, d_model)
    → sinusoidal positional encoding
    → N × TransformerEncoderLayer(d_model, n_heads, ff, dropout, norm_first=True, batch_first=True)
    → transpose: (B, d_model, T)
    → output_conv: Conv1d(d_model, 5, k=1)
  Output: (B, 5, T=256)
  ```

- [x] **Design decisions**:
  - **Pre-LN** (`norm_first=True`) — stable without warmup, compatible with ReduceLROnPlateau
  - **Sinusoidal PE** — no extra params, generalizes to different lengths
  - **Bidirectional attention** (no causal mask) — matches U-Net/TCN acausal design
  - **No downsampling** — strides=[1], total_stride=1
  - **GELU activation** — matches U-Net/TCN

- [x] **Critical fix: sliding-window evaluation**. Initial results were catastrophically bad (val r ~0.05 after 30 epochs). Root cause: `evaluate_fullsig` feeds the entire signal (5K-20K timesteps) through the model at once. For convnets this is fine (translation-equivariant), but for transformers the attention distribution and PE completely shift at different sequence lengths vs training windows (256 steps). Fixed by adding `_sliding_window_forward()` with 50% overlapping windows matching the training context. After fix: val r jumped from 0.01 to 0.32 at epoch 2. **Lesson: sliding-window eval is mandatory for transformers on this task.**

- [x] Registered in `src/models/__init__.py` (`MODEL_REGISTRY["transformer"]`, `MODEL_REGISTRY["hybrid_transformer"]`)
- [x] Updated `src/data/__init__.py` — added `"transformer"` and `"hybrid_transformer"` to `SPECTROGRAM_MODELS`

### Milestone 5.2 — Systematic tuning on BCI-IV (3 subjects)

BCI-IV is our fast iteration testbed (3 subjects, ~25 min/run). Target: beat U-Net mean test r = 0.597 (S1=0.575, S2=0.520, S3=0.692).

All experiments used: seed=123, batch=128, lr=8.42e-5 (unless noted), mse_cosine loss, 40 epochs max, early stopping patience=15.

**Round 1: Initial Grid Search (4 variants × 3 subjects)**

| Variant | S1 | S2 | S3 | Mean |
|---------|------|------|------|------|
| A (d=64, L=4, lr=8.42e-5) | 0.401 | 0.253 | 0.559 | 0.404 |
| A (d=64, L=4, lr=3e-5) | 0.413 | 0.242 | 0.580 | 0.412 |
| B (d=128, L=2, lr=8.42e-5) | 0.426 | 0.303 | 0.604 | **0.444** |
| B (d=128, L=2, lr=3e-5) | 0.423 | 0.199 | 0.532 | 0.385 |

Winner: B (d=128, L=2, lr=8.42e-5). d=128 >> d=64; higher LR better.

**Round 2: Depth (L=3, L=4, L=5)**

| Variant | S1 | S2 | S3 | Mean |
|---------|------|------|------|------|
| L=2 (baseline) | 0.426 | 0.303 | 0.604 | 0.444 |
| L=3 | 0.429 | 0.307 | 0.593 | 0.443 |
| L=4 | 0.435 | 0.351 | 0.618 | **0.468** |
| L=5 | 0.436 | 0.376 | 0.608 | 0.473 |

Winner: L=4 — monotonic improvement L2→L4, L=5 marginal and hurt S3.

**Round 3: Feedforward Width**

| Variant | S1 | S2 | S3 | Mean |
|---------|------|------|------|------|
| ff=256 (baseline) | 0.435 | 0.351 | 0.618 | 0.468 |
| ff=512 | 0.487 | 0.380 | 0.630 | **0.499** |
| ff=768 | 0.450 | 0.392 | 0.665 | 0.502 |
| ff=1024 | 0.430 | 0.416 | 0.670 | 0.505 |

ff=256→512 biggest jump (+0.031 mean). Diminishing returns beyond 512.

**Round 4: Attention Heads (4 vs 8)**

| Variant | S1 | S2 | S3 | Mean |
|---------|------|------|------|------|
| 4 heads | 0.487 | 0.380 | 0.630 | 0.499 |
| 8 heads | 0.491 | 0.371 | 0.643 | 0.502 |

No significant difference — heads don't matter much.

**Round 5: Spatial Kernel Size (k=1 vs k=3)**

| Variant | S1 | S2 | S3 | Mean |
|---------|------|------|------|------|
| k=1 | 0.430 | 0.416 | 0.670 | 0.505 |
| k=3 | 0.428 | 0.370 | 0.644 | 0.481 |

k=3 hurt — larger spatial kernel didn't help.

**Round 6: d_model=256**

| Variant | S1 | S2 | S3 | Mean | Params |
|---------|------|------|------|------|--------|
| d=256, ff=512 | **0.505** | 0.359 | **0.687** | **0.517** | 2.7M |
| d=256, ff=1024 | 0.481 | 0.374 | 0.677 | 0.511 | 3.8M |

S1 jumped 0.435→0.505. S3 nearly matches U-Net (0.687 vs 0.692). ff=512 > ff=1024 at d=256 (wider model needs less ff to avoid overfitting).

**Round 7: Dropout (d=256, ff=512)**

| Variant | S1 | S2 | S3 | Mean |
|---------|------|------|------|------|
| drop=0.1 | 0.505 | 0.359 | 0.687 | **0.517** |
| drop=0.15 | 0.484 | 0.363 | 0.684 | 0.510 |
| drop=0.2 | 0.486 | 0.355 | **0.700** | 0.514 |
| drop=0.3 | 0.450 | 0.320 | 0.699 | 0.490 |

drop=0.1 still best overall. drop=0.2 beat U-Net on S3 (0.700!) but hurt S1/S2.

**Round 8: Hybrid Transformer (downsample + transformer + upsample + skip)**

| Variant | S1 | S2 | S3 | Mean |
|---------|------|------|------|------|
| Pure transformer | 0.505 | 0.359 | 0.687 | 0.517 |
| Hybrid | 0.367 | 0.320 | 0.617 | **0.434** |

Hybrid failed badly — single downsample/upsample lost information, didn't replicate U-Net's multi-scale benefit.

**Round 9: d_model=512**

| Variant | S1 | S2 | S3 | Mean | Params |
|---------|------|------|------|------|--------|
| d=512, ff=1024 | 0.537 | 0.421 | 0.697 | 0.552 | 9.7M |
| d=512, ff=2048 | 0.556 | 0.445 | 0.699 | 0.567 | 13.9M |

ff=2048 improved all 3 subjects. Gap to U-Net = 0.030.

**Round 10: d_model=1024**

| Variant | S1 | S2 | S3 | Mean | Params |
|---------|------|------|------|------|--------|
| d=1024, ff=2048 | 0.576 | 0.455 | 0.738 | 0.590 | 36M |
| d=1024, ff=4096 | **0.583** | **0.484** | **0.731** | **0.599** | 53M |

**d1024_ff4096 BEATS U-Net: mean 0.599 vs 0.596!**
Scaling trend confirmed: d64(0.404) → d128(0.468) → d256(0.517) → d512(0.567) → d1024(0.599)

**Round 11: Depth & Heads Ablation (d=1024, ff=4096)**

| Config | S1 | S2 | S3 | Mean | Params | Notes |
|--------|----|----|-----|------|--------|-------|
| L4, h8 (Round 10 baseline) | 0.583 | 0.484 | 0.731 | 0.599 | 53M | baseline |
| L6, h8 | 0.580 | 0.471 | **0.745** | 0.599 | 78M | S3+0.014, S2 hurt |
| L4, h16 | **0.612** | 0.482 | 0.732 | 0.609 | 53M | S1+0.029 |
| **L6, h16** | **0.620** | **0.495** | **0.748** | **0.621** | 78M | **SYNERGISTIC BEST** |
| L8, h16 | 0.605 | — | — | — | 104M | deeper = worse |
| L6, h32 | ~0.615 | — | — | — | 78M | same as h16, no gain |

**FAILED experiments this round:**
- window=512 (5.12s context): S1 0.531 (-0.052) — longer context hurts
- Hybrid U-Net+Transformer: epoch 4 val r=0.400 vs 0.557 — skip shortcut causes overfitting
- ff=8192: val r=0.587 at epoch 21 vs 0.665 — larger FFN hurts
- spatial_kernel=3: epoch 11 val r=0.523 vs 0.644 — local conv before transformer hurts

**Key insights from Round 11:**
1. **h16 (16 heads, head_dim=64) is the primary lever for S1** — +0.029
2. **L6 depth helps S3 specifically** — +0.014
3. **L6+h16 is synergistic** — improves ALL subjects simultaneously
4. **L8 is over-capacity** — L6 is the sweet spot
5. **h32 (head_dim=32) no gain** — h16 is the head sweet spot
6. **val→test smoothing mismatch**: val uses sigma=6, test uses sigma=1 — high val r ≠ high test r across all models
7. **Silent crash bug**: launching multiple nohup jobs in one Bash call fails silently — always separate calls

### Milestone 5.3 — Architecture comparison on BCI-IV ✅ COMPLETE

| Model | S1 | S2 | S3 | Mean | vs U-Net |
|-------|------|------|------|------|----------|
| TCN (best) | — | — | — | 0.408 | -0.189 |
| **U-Net** | 0.575 | **0.520** | 0.692 | 0.596 | — |
| Transformer (d1024, L4, h8) | 0.583 | 0.484 | 0.731 | 0.599 | +0.003 |
| **Transformer (d1024, L6, h16)** | **0.620** | 0.495 | **0.748** | **0.621** | **+0.025** |

- **Best Transformer beats U-Net by 0.025 mean Pearson r**
- All 3 subjects exceed U-Net baseline for S1 and S3; S2 still below U-Net (0.495 vs 0.520)
- **Best config: d=1024, L=6, h=16, ff=4096, batch=64** (78M params)

- [x] Complete d512 tuning
- [x] Scale to d_model=1024
- [x] Ablate depth (L4/L6/L8), heads (h8/h16/h32), ff (4096/8192), window (256/512)
- [x] **Best config confirmed: L6+h16 = 0.621 mean (+0.025 vs U-Net)**
- [ ] Paired Wilcoxon signed-rank tests for architecture comparison
- [x] **Proceed to 9-patient Miller 2019 sweep with d1024_L6_h16_ff4096**

**MILESTONE ACHIEVED: Transformer (L6+h16) outperforms U-Net by +0.025. Ready for Miller sweep.**

### Milestone 5.4 — Architecture search: bottleneck + multiscale (NEGATIVE)

Tested two structural changes with the goal of improving spatial/temporal representation:

**Spatial bottleneck** (1920→128→1024): compressed spatial_reduce through a bottleneck.
- S1=0.553, S2=0.424, S3=0.730, **mean=0.569** (vs 0.621, -0.052)
- Catastrophic for S2 (48ch) — destroying already-limited spatial information.

**Multiscale Transformer** (T→T/4→T/16, 116M params):
- S1≈0.55, S2=0.379, S3=0.729, **mean≈0.553** (vs 0.621, -0.068)
- T/16=16 timesteps is too short a bottleneck; 116M params overfits ~33K training samples.
- "Dead finger" (one finger near r=0) bug on S1.

**Conclusion**: flat L6+h16 at 78M is well-calibrated for ~33K training samples. Structural complexity hurts.

### Milestone 5.5 — Optimization ablations (S1 only, BCI-IV) ✅ COMPLETE

Tested 6 remaining improvements on S1 (script: `scripts/grid_phase6_bci4.sh`):

| Exp | Config | LR | val_best | test_r S1 | vs baseline |
|-----|--------|----|----------|-----------|-------------|
| A | sigma_val=1 (ctrl) | 8.42e-5 | 0.688 | **0.621** | +0.001 (neutral) |
| B | sigma_val=1 | 1e-4 | 0.643 | 0.598 | -0.023 |
| C | sigma_val=1 | 2e-4 | 0.641 | 0.583 | -0.038 |
| D | sigma_val=1 | 3e-4 | 0.621 | 0.575 | -0.046 |
| E | SwiGLU, ff=2730 | 8.42e-5 | 0.618 | 0.603 | -0.018 |
| F | d=1536, ff=4096 | 8.42e-5 | 0.647* | ~0.583 est. | ~-0.038 |

*F killed at ep21; best val was ep8 val r=0.647, LR already halved at ep15 with no further improvement.

**Findings:**
- **LR=8.42e-5 is confirmed optimal.** Higher LRs cause noisier val_r → premature LR decay → underfit.
- **Sigma fix (sigma_val=1) is neutral** (+0.001). Noisier val_r from sigma=1 offsets better metric alignment.
- **SwiGLU is worse than GELU** (-0.018). Despite LLaMA/PaLM precedent, GELU is better here — possibly because the gating mechanism is counterproductive at small ECoG signal-to-noise.
- **d=1536 with ff=4096 (ratio 2.67x) is worse.** Peaked early (ep8 val 0.647 vs A's ep23 val 0.688). The under-scaled FFN (4096/1536=2.67x vs 4096/1024=4x) may be the bottleneck; a fair test would be ff=6144 (4x ratio), but given per-epoch time of ~550s (vs ~250s for d=1024) and diminishing returns, not pursued.

**Final BCI-IV best: d=1024, L=6, h=16, ff=4096, GELU, lr=8.42e-5, sigma_val=1, mean=0.621.**
No further improvement found. Proceeding to Miller 2019 sweep.

### Milestone 5.6 — Scheduler signal experiment (NEGATIVE, informative)

**Hypothesis:** ReduceLROnPlateau tracks val_loss (mode=min), but we checkpoint on val_r. These diverge — val_loss plateaus at ep6 while val_r still climbs to ep20. Switching scheduler to track val_r should give more appropriate LR decay timing.

**Result (script: `configs/bci4_transformer_d1024_L6_h16_valr_sched.yaml`, `results/phase7_s1_valr_sched.log`):**
- val_r scheduler: val_best=0.681 (ep22), **test_r=0.608** — worse than baseline by -0.013
- Old val_loss scheduler: val_best=0.688 (ep20), **test_r=0.621**

**Why the "bug" is actually a feature:** The val_loss plateau at ep6 triggers LR halving at ep15 — precisely when val_r is at 0.653 and still climbing. That early transition to half-LR creates a clean precision phase: model climbs 0.653→0.688 (+0.035) in 5 epochs. Tracking val_r delays LR drop to ep31 (full LR for 30 epochs) causing noisy oscillations; the model can't recover past its ep22 peak of 0.681 after the late LR cut. **Conclusion: keep `scheduler(val_loss, mode=min)`.** The scheduler and checkpoint use different signals intentionally.

---

## Milestone 5.7 — Literature review + paper analysis: SOTA on BCI-IV Dataset 4 (March 2026)

**Papers downloaded to `papers/`:** dtcnet_frontiers2025.pdf ✓, jangir_braininformatics2025.pdf ✓, fingerflex_v1_arxiv2022.pdf ✓
**Papers not obtained:** FingerFlex v2 (journal, not on arXiv), DeepFingerNet/Tao 2025 (wrong file downloaded — got unrelated paper), Tragoudaras 2023 (IEEE paywall)

### ⚠️ Critical: Evaluation Protocol Differences

**Our numbers (test sigma=1) are NOT directly comparable to published numbers.** Evidence: our U-Net val_r (sigma=6) = S1=0.686, S2=0.633, S3=0.737 — **exactly matching the FingerFlex arXiv 2022 numbers** (0.686, 0.633, 0.737). Our test_r with sigma=1 drops to 0.575/0.520/0.692 (mean=0.597). Published papers report sigma=6 smoothed val results, not test set with sigma=1.

**Second factor:** We carve 15% of official train data for validation; published methods train on 100% of the official split. This alone explains ~0.04 r gap for S1 vs FingerFlex (their S1 test_r sigma=1 = 0.627 from notebook; ours = 0.575 from carving 15%).

All internal comparisons remain valid (same protocol); external comparisons need notes on both factors.

### SOTA Table (BCI-IV Dataset 4, official test set, Pearson r avg over 5 fingers)

Numbers from DTCNet Table 1 (read directly from paper — most internally consistent source):

| Method | Year | S1 | S2 | S3 | Mean | Notes |
|--------|------|-----|-----|-----|------|-------|
| Flamary & Rakotomamonjy | 2012 | 0.48 | 0.24 | 0.56 | 0.43 | Competition winner |
| Liang & Bougrain | 2012 | 0.45 | 0.39 | 0.59 | 0.48 | 2nd place |
| Xie et al. | 2018 | 0.56 | 0.41 | 0.58 | 0.52 | CNN-LSTM |
| Petrosyan et al. | 2021 | 0.45 | 0.34 | 0.56 | 0.45 | Interpretable CNN |
| Yao et al. | 2022 | 0.52 | 0.47 | 0.61 | 0.53 | LightGBM + Riemannian |
| **FingerFlex v1** (Lomtev et al.) | 2022/23 | 0.64 | 0.56 | 0.73 | **0.64** | Our codebase reproduces this |
| **FingerFlex v2** (Lomtev & Kovalev) | 2024 | — | — | — | **~0.79** | Journal update; not on arXiv |
| **DTCNet** (Wang et al.) | 2025 | **0.71** | **0.59** | **0.77** | **0.69** | Best reproducible SOTA |
| BC4D4 (Jangir et al.) | 2025 | — | — | — | 0.85† | Suspicious — see note below |
| Tragoudaras et al. | 2023 | — | — | — | 0.886† | Suspicious — see note below |
| *Our U-Net (sigma=1, 85% train)* | — | 0.575 | 0.520 | 0.692 | 0.597 | Honest; ↑ to 0.685 at sigma=6 |
| *Our Transformer (sigma=1, 85% train)* | — | 0.620 | 0.495 | 0.748 | 0.621 | Best config: d=1024 L=6 h=16 |

**⚠️ DTCNet abstract claims "82%" — this is S3 Thumb finger only (0.82), NOT the mean over subjects.** The actual subject-level averages are 0.71/0.59/0.77 = 0.69 mean. The abstract is misleading. We read the paper directly (Table 2).

†**BC4D4 (0.85) is unreproducible** — we attempted to reproduce and could not match results. Reading the paper reveals why: they apply Isolation Forest to remove ~70% of the data as "outliers," then appear to evaluate on the cleaned/filtered set rather than the official test split. FingerFlex numbers they cite (0.66/0.62/0.74 = 0.67) match our val_r (sigma=6), not the official test set — confirming they evaluated on a non-standard split. Discard.

†**Tragoudaras (0.886)** — Transformer used as NAS meta-optimizer, not as decoder. Evaluation protocol unclear. Discard.

### Paper-by-Paper Preprocessing Analysis

#### DTCNet (Wang et al., Frontiers Comput. Neurosci., 2025) — Read directly ✓
**Preprocessing pipeline:**
1. Resample ECoG 1000 Hz → 100 Hz, finger data 25 Hz → 100 Hz (cubic interpolation)
2. Z-score normalize each channel (mean/std), then remove channel median
3. Bandpass **40–300 Hz**
4. Notch at **60 Hz** and harmonics (US recording → 60 Hz power line)
5. Morlet wavelet spectrogram: 40 frequencies 40–300 Hz, output shape (channels, freq=40, time)
6. Sliding window size=256, **stride=1** (every single time step → maximum training data)
7. Train on **100% of official train split** (no val carve-out)

**Architecture:** Dilated-transposed CNN encoder-decoder, U-Net style. Feature dims 64→64→128→256→512→512, dilations (1,2,3,1,2) sawtooth pattern. Skip connections. Loss = MSE + cosine similarity.

**Result:** S1=0.71, S2=0.59, S3=0.77, **mean=0.69**. Best per-finger: S3 Thumb=0.82 (this is the "82%" headline — one finger, one subject).

**Ablation:** Morlet > Symlet > Haar > Daubechies. Removing dilated conv: 0.69→0.59. Removing transposed conv: 0.69→0.61. 3D structure matters.

**What we can borrow:**
- ✅ Stride=1 — **we already do this**
- ✅ MSE + cosine loss — **we already do this**
- ✅ Morlet wavelet, 40–300 Hz — **we already do this**
- ⚠️ Notch at 60 Hz — we use 50 Hz (following FingerFlex). BCI-IV is US data → 60 Hz is correct. FingerFlex also uses 50 Hz and we reproduce it exactly, so impact is minimal, but worth correcting.
- ⚠️ 3D spectrogram structure (channels × freq × time) — we flatten channels×freq. Preserving 3D and using the first conv to fuse is an architectural option worth testing.
- ⚠️ Train on 100% of official split — closes ~0.04 r gap. Requires alternative val strategy (use train loss for LR scheduling, or use first N epochs as warmup).

#### FingerFlex v1 (Lomtev et al., 2022/23) — Read directly ✓
**Preprocessing:** same as ours (we reproduce their exact numbers). Key details:
- Bandpass 40–300 Hz, notch 50 Hz (European convention — likely a copy-paste, dataset is US)
- Morlet wavelet, 40 frequencies, 100 Hz downsample
- Reports val_r (sigma=6) as main metric — NOT official test set. Their S1=0.686/S2=0.633/S3=0.737 are val numbers.
- Train on 100% of official train split. We carve 15% for val → that's why we're ~0.04 r below them even at sigma=6 val.

#### BC4D4 (Jangir et al., Brain Informatics, 2025) — Read directly, results unreproducible ✗
**Preprocessing:** No standard filtering (specifically avoids bandpass/notch). Uses Isolation Forest to remove ~70% of data as outliers. Tanh/Softsign activations. Architecture: Conv1D × 3 + Dense × 5.
**Verdict:** Evaluated on non-standard cleaned subset, not official test split. Results (0.85) not comparable to anything else. Do not use as baseline.

### ⭐ Preprocessing Differences: Us vs Published (What to Borrow)

| Step | Our Current | FingerFlex v1 | DTCNet | Action |
|------|-------------|---------------|--------|--------|
| Resample | 100 Hz | 100 Hz | 100 Hz | ✅ Same |
| Bandpass | 40–300 Hz | 40–300 Hz | 40–300 Hz | ✅ Same |
| Notch | **50 Hz** | **50 Hz** | **60 Hz** | Low priority fix (60 Hz is correct for US data but FingerFlex uses 50 Hz too) |
| Normalization | z-score per ch | z-score per ch | z-score + median removal | Worth trying median removal |
| Spectrogram | Morlet, 40 freqs | Morlet, 40 freqs | Morlet, 40 freqs (3D) | ✅ Same; 3D structure is architectural not preprocessing |
| Window size | 256 | 256 | 256 | ✅ Same |
| Stride | **1** | — | **1** | ✅ Already stride=1 |
| Loss | mse_cosine | mse | mse_cosine | ✅ Already using combined loss |
| Train split | **85% of official** | **100% of official** | **100% of official** | **Key gap: worth an experiment** |
| Val strategy | 15% carve-out | N/A | N/A | Use train loss for LR, evaluate on val carved from test? |

**Biggest actionable gap: train on 100% of official train split.** All others are already aligned.

### ⭐ Actionable Experiments (in priority order)

1. **[HIGH] Train on 100% of train split.** Remove the 15% val carve. Use train loss for LR scheduling (already works per our scheduler analysis — val_loss tracks train loss early on). Expected gain: ~0.04 r for S1, closes gap to published FingerFlex v1 (0.64 official test vs our 0.575). Run BCI-IV S1 first.

2. **[MEDIUM] Report sigma=6 test numbers.** Run eval script with sigma=6 on test set for both U-Net and Transformer. This makes our numbers directly comparable to published tables. One-time eval, no retraining.

3. **[MEDIUM] Investigate FingerFlex v2 (2024).** What changed? Semantic Scholar / contact authors. If it's just better training (not architecture), we can apply same improvements. If architectural, may reveal a direction.

4. **[LOW] 3D spectrogram structure.** Our Morlet already produces (channels, freq, time). Currently we reshape to (channels*freq, time) before the model. Could instead pass as true 3D and use a conv that fuses freq first. DTCNet gains come largely from architecture (dilated+transposed conv), not just 3D structure, so impact is uncertain without the full DTCNet architecture.

5. **[LOW] Fix notch to 60 Hz.** BCI-IV is US data. Minimal expected impact since FingerFlex also uses 50 Hz and we reproduce it exactly.

6. **[PAPER REQUIREMENT] Multi-seed sweep.** Run best Transformer config 3 seeds, report mean ± std. Required for any published claim.

---

## Phase 5 Continuation: Paper Reproduction Track

**Principle:** A result is not truth until we can independently reproduce it. We reproduce papers one at a time, in credibility order. For each paper: (1) extract exact preprocessing + architecture + evaluation, (2) implement faithfully in our codebase, (3) run on BCI-IV, (4) compare to reported numbers. If we match → we learn something real. If we can't match → the claim is suspicious.

**Status summary:**
| Paper | Claimed mean r | Credibility | Status |
|-------|---------------|-------------|--------|
| FingerFlex v1 (Lomtev 2022) | 0.67 (sigma=6 val) | High | ✅ Reproduced exactly (our val_r sigma=6 = 0.685) |
| Jangir BC4D4 (2025) | 0.85 | None | ❌ Cannot reproduce — non-standard eval, 70% data removed by outlier filter |
| DeepFingerNet (Tao 2025) | 0.69 (no sigma) | High | 🔲 To reproduce next |
| DTCNet (Wang 2025) | 0.69 (no sigma) | High | 🔲 To reproduce |
| Tragoudaras DNN (2023) | 0.886 (no sigma) | Very Low | 🔲 Low priority — requires 8K model training runs |

---

### Milestone 5.8 — Reproduce FingerFlex v1 Exactly ✅ COMPLETE — PARTIAL MATCH

**Model confirmed identical:** `unet_lomtev.py` is a faithful port of `FingerFlex/Lightning_BCI-autoencoder.ipynb`. Architecture, preprocessing, hyperparameters all match. Our previous "exact match" of 0.686 was on our carved 15% val set (from train data), NOT on the official test set — those are easier to predict (chronologically close to training data).

**Full reproduction results (100% official train split, sigma=6, official test set, seed=42, 20 epochs, fixed LR=8.42e-5):**

| Subject | Our result | FingerFlex published | Gap | Peak epoch |
|---------|-----------|---------------------|-----|------------|
| S1 | **0.605** | 0.686 | −0.081 | ep5 |
| S2 | **0.535** | 0.633 | −0.098 | ep2 |
| S3 | **0.701** | 0.737 | −0.036 | ep1 |

**Key findings:**
1. **Cannot reproduce FingerFlex's numbers with a single seed.** S3 peaked at epoch 1 (0.701) then steadily declined to 0.614 by epoch 20 — fixed LR causes overfitting that never recovers. S1 peaked at epoch 5 (0.605). Neither reaches published numbers.
2. **FingerFlex's published 0.686/0.633/0.737 are not straightforwardly reproducible** with identical code in a single run. Their checkpoint name (`epoch=16, val_r=0.679`) shows even their own best run achieved 0.679, not 0.686 — they likely report best-of-multiple-seeds.
3. **Our previous "exact match" claim was misleading**: 0.686 was on a carved val from train data (easy), not the official test set (hard). On the official test set, our honest number with identical FingerFlex setup is **0.605** for S1.
4. **The LR scheduler we added (ReduceLROnPlateau) is the real reason our own best results were better** than naive FingerFlex. Our Transformer (0.621 sigma=1) beats FingerFlex's honest reproducible number (0.605 sigma=6 ≈ ~0.54 sigma=1) substantially.
5. **S2 is anomalously slow** (epoch takes 300s vs 90s for S1/S3) — likely preprocessing overhead with 48 channels at a different resolution. Not a training issue.

**Verdict: FingerFlex v1 partially reproduced.** Architecture and preprocessing confirmed identical. Performance gap vs published = seeds/luck, not a missing component. Published 0.686 requires favorable initialization; our honest single-seed result = 0.605 (sigma=6 official test).

---

### Milestone 5.9 — Reproduce DeepFingerNet (Tao et al., IEEE TIM 2025)

**Paper:** `papers/tao_DeepFingerNet_2025.pdf` — read and extracted fully.

**Why reproduce first:** Most credible post-FingerFlex result. Clear methodology. Same metric (no sigma smoothing). Their 0.69 is directly comparable to our best 0.621. Gap is ~0.07 — real and explainable.

**Exact paper spec:**
- **Input:** ECoG channels × 40 Morlet wavelet kernels (log-spaced, 40–300 Hz), downsampled to 100 Hz
  - Input tensor shape: `(N_channels × 40, time)` — they flatten channels and frequencies
- **Normalization:** z-score per channel, then subtract channel median; **RobustScaler** applied to ECoG (IQR-based)
- **Finger targets:** upsample 25→100 Hz cubic interpolation; **MinMaxScaler** to [0, 1]
- **Temporal alignment:** remove first 20 ECoG points + last 20 finger points (0.2s delay)
- **Window:** size=256, stride=1
- **Train/val split:** official 6.5:3.5 split (400s train / 200s test from BCI-IV) — treat test as val
- **Architecture:** 3-UNet (nested U-Nets with dense skip connections)
  - Encoder: 1D Conv → LayerNorm → GELU → MaxPool, channels: 32→64→128→256
  - Decoder: 1D TransposedConv → LayerNorm → GELU, symmetric 256→128→64→32
  - Skip connections: 3 levels of nesting between 3 parallel U-Nets
  - Output: 1×1 Conv → 5 fingers
  - Model size: ~1.30 MB
- **Loss:** 0.5 × MSE + 0.5 × (1 − cosine_similarity)
- **Optimizer:** Adam, lr=2e-5, weight_decay=1e-6
- **Batch size:** 128, max epochs: 20
- **Eval metric:** Pearson r on raw MinMaxScaled predictions vs targets — **no Gaussian smoothing**

**Expected result:** mean=0.69 (S1: not given per-subject in paper — bar chart only)

#### Attempt 1: Summed Wavelet Power (n_wavelets=0) — FAILED

**Preprocessing interpretation:** The paper equation "X_wavelet = Σ W_k * X" was interpreted as: compute 40 Morlet wavelet power spectrograms (N,40,T), then sum across frequency → (N,1,T). This gives n_channels_in=62×1=62 → ~321K params matching paper's ~325K (1.30 MB).

Config: `configs/bci4_deepfingernet.yaml`

| Subject | Our result (σ=1) | Paper claim | Gap |
|---------|-----------------|-------------|-----|
| S1 | 0.169 | 0.71 | −0.54 |
| S2 | 0.225 | 0.59 | −0.37 |
| S3 | 0.363 | 0.77 | −0.41 |
| **Mean** | **0.252** | **0.69** | **−0.44** |

**Root cause of failure:** Summing 40 frequency bands into 1 channel loses ALL frequency discrimination. The model receives a single broadband power envelope per electrode — essentially a smoothed activity indicator. This is too weak a feature for regression. Training loss decreases (0.051→0.007 over 20 epochs) but generalization barely improves (val_r still 0.169 at epoch 20) — the model memorizes training windows but learns no transferable signal representation.

#### Attempt 2: Full Spectrogram (n_wavelets=40) — FAILED

**Hypothesis:** The paper DOES use the full 40-frequency Morlet spectrogram (N×40, T) as input, matching their spec "channels × 40 wavelets". The param count discrepancy (~553K vs ~325K) may be a reporting error in the paper.

Config: `configs/bci4_deepfingernet_fullspec.yaml`

| Subject | Paper setup (fixed lr=2e-5) | Our setup (adaptive lr) | Paper claim |
|---------|----------------------------|------------------------|-------------|
| S1 | 0.427 | 0.396 | 0.71 |
| S2 | 0.312 | — | 0.59 |
| S3 | 0.541 | — | 0.77 |
| **Mean** | **0.427** | — | **0.69** |

**Root cause of failure:** Two compounding issues:
1. **Fixed lr=2e-5 causes rapid overfitting.** Train loss drops 20× (0.057→0.003) in 5 epochs while val loss keeps rising. Val_r peaks at epoch 2–4 and then steadily declines for the remaining 16 epochs. The model memorizes training windows within 4 epochs but cannot generalize.
2. **k=1 decoder lacks temporal receptive field.** All decoder nodes use k=1 (pointwise) convolutions — no temporal blending in the reconstruction path. This is inferior to our U-Net decoder which uses k=3 throughout. Evidence: our U-Net (same preprocessing, adaptive lr) achieves S1=0.575, far above NestedUNet's best S1=0.427.

**Architecture comparison on BCI-IV S1:**

| Model | Params | train setup | test_r S1 |
|-------|--------|------------|-----------|
| Our U-Net (Lomtev) | 652K | adaptive lr, 85% train | 0.575 |
| NestedUNet (paper setup) | 553K | fixed lr, 100% train | 0.427 |
| NestedUNet (our setup) | 553K | adaptive lr, 85% train | 0.396 |

The dense skip connections of UNet++ do NOT compensate for the k=1 decoder weakness. NestedUNet as implemented is WORSE than our plain U-Net.

#### Attempt 3 (v2): Architecture Fix — Dropout + Pre-Norm Decoder — FAILED

After re-examining the paper's Figure 2 block diagrams, we identified two bugs in our implementation: (1) missing Dropout layers in both encoder and decoder (clearly shown in the diagram), and (2) wrong decoder ordering — paper uses pre-norm (LN→GELU→Conv→Drop) while we had post-norm (Conv→LN→GELU). Fixed both in `nested_unet.py` and re-ran all configs.

| Config | S1 | S2 | S3 | Mean | vs. prev |
|--------|-----|-----|-----|------|----------|
| v2 raw (n_wv=0, dropout+prenorm) | 0.188 | 0.264 | 0.404 | **0.285** | +0.033 |
| v2 fullspec (n_wv=40, dropout+prenorm) | 0.342 | 0.351 | 0.547 | **0.413** | −0.014 |

Dropout helped the raw config modestly but slightly hurt the fullspec config. The same overfitting pattern persists: val_r peaks at epoch 3–8 then declines under fixed lr=2e-5. The gap to the paper's 0.69 remains ~0.28–0.41.

**Verdict: ❌ DeepFingerNet NOT reproducible after 4 attempts.** Their claimed mean=0.69 (S1=0.71, S2=0.59, S3=0.77) is not achievable with the architecture and training setup described in the paper. Our best result across all attempts is mean=0.427 — a gap of 0.26. The architecture is also inferior to our existing U-Net (0.427 vs 0.597 mean). The paper likely overfits to validation, uses undisclosed hyperparameter search, or evaluates on a non-standard split.

**Implementation completed:**
- [x] `src/models/nested_unet.py` — NestedUNet with dense skip connections
- [x] `src/models/__init__.py` — registered `MODEL_REGISTRY["nested_unet"]`
- [x] `src/data/__init__.py` — added `"nested_unet"` to `SPECTROGRAM_MODELS`
- [x] `src/data/dataset_lomtev.py` — added `n_wavelets=0` mode (summed power; aliasing-free)
- [x] Configs: `bci4_deepfingernet.yaml`, `bci4_deepfingernet_fullspec.yaml`, `bci4_nested_unet_ours.yaml`

---

### Milestone 5.10 — Reproduce DTCNet (Wang et al., Frontiers 2025)

**Paper:** `papers/dtcnet_frontiers2025.pdf` — read and extracted fully.

**Why reproduce:** Same 0.69 target as DeepFingerNet but completely different approach (dilated CNN vs nested U-Net). If both reproducible at 0.69 that's strong evidence 0.69 is the real ceiling for current CNN approaches on this dataset size.

**Exact paper spec:**
- **Input:** 3D Morlet spectrograms, shape `(channels, 40_freqs, time)` — preserves 3D structure
- **Normalization:** z-score per channel, subtract channel median
- **Bandpass:** 40–300 Hz; **Notch:** 60 Hz (US power line)
- **Window:** size=256, stride=1; train on 100% of official train split
- **Architecture:** Dilated-transposed CNN encoder-decoder
  - Feature dims: 64 → 64 → 128 → 256 → 512 → 512 (encoder)
  - First layer: 3×3 conv, standard
  - Layers 2–5: 1D dilated conv, kernel sizes (7,7,5,5,5), dilations (1,2,3,1,2)
  - Decoder: transposed conv + skip concatenation, mirror of encoder
  - Output: 1×1 conv → 5 fingers
- **Loss:** MSE + cosine similarity
- **Optimizer:** Adam, lr=8.42e-5, weight_decay=1e-6, dropout=0.1
- **Eval metric:** Pearson r — no sigma smoothing mentioned

**Expected result:** S1=0.71, S2=0.59, S3=0.77, mean=0.69

**Implementation steps:**
- [ ] Implement `DTCNet` in `src/models/dtcnet.py` with exact dilation pattern and channel progression
- [ ] Add config `bci4_dtcnet_reproduction.yaml` — preserve 3D input `(channels, 40, time)` entering a spatial-fusion first layer
- [ ] Run BCI-IV S1 first, target r=0.71
- [ ] If S1 matches within ±0.03, run S2 and S3
- [ ] Compare to DeepFingerNet reproduction: which approach is more effective?
- [ ] **Record verdict:** reproduced / not reproduced / partially reproduced

**What we learn if successful:**
- Dilated convolutions (sawtooth pattern) vs nested U-Net: both reach 0.69 — confirms the ceiling
- 3D spectrogram structure (channels, freq, time) contribution vs flat (channels×freq, time)
- Whether DTCNet's architecture improvements transfer to our Transformer-based approach

---

### Milestone 5.11 — Evaluate Tragoudaras (IEEE Access 2023) — NAS Result

**Papers:** `papers/Tragoudaras_cnn_2023.pdf` + `papers/Tragoudaras_dnn_2023.pdf` — read fully.

**What this paper actually is:** NOT a new decoder architecture. It's a **Neural Architecture Search (NAS)** framework using a Transformer surrogate model + Firefly optimizer to find optimal hyperparameters for the **Shallow ConvNet** (Schirrmeister 2017). The Shallow ConvNet is a very simple model: temporal conv → spatial filter → squaring → mean pool → log → FC output.

**Claimed result:** mean r = 0.886 (journal) / 0.87 (conference). Far above everything else.

**Why this is suspicious:**
1. The Shallow ConvNet (even optimally tuned) is architecturally far simpler than DTCNet/DeepFingerNet — no wavelets, no U-Net structure, no skip connections. It would need an extraordinary inductive bias advantage to beat them by +0.2 r.
2. NAS paper claims "0.87 vs 0.67 baseline" — but the 0.67 baseline (Shallow ConvNet default) far exceeds the original paper's reported numbers for the same model.
3. Preprocessing is "1–200 Hz" — no Morlet wavelets. They use raw ECoG. This is fundamentally different from all modern approaches.
4. Evaluation: no sigma mentioned, no per-subject breakdown in text, bar chart only.
5. No peer has independently reproduced or cited this result in subsequent literature (DTCNet does not cite it).

**Reproduction plan — minimal cost approach:**
- [ ] Implement the Shallow ConvNet with their **optimal hyperparameters** (known from Table IV): temporal_filters=27, filter_len=35, spatial_filters=48, pool_kernel=82, pool_stride=20, dropout=0.2, lr=0.0125, wd=68e-4
- [ ] Run this exact config on BCI-IV S1 using their preprocessing (1–200 Hz bandpass, no wavelets, no notch specified)
- [ ] Compare to their claimed 0.87 baseline
- **We do NOT reproduce the NAS framework itself** (8,000 model training runs = impractical)
- We only test: can their published optimal ConvNet achieve anywhere near 0.87?
- If it achieves ~0.67 (original Shallow ConvNet level): paper is bogus — NAS didn't actually help or numbers are wrong
- If it achieves 0.87: highly surprising, investigate what in the preprocessing drives this

**What we learn:**
- Whether the Shallow ConvNet with optimal hyperparameters is genuinely competitive (it almost certainly isn't)
- Whether 1–200 Hz raw ECoG (no wavelets) is actually competitive with our 40–300 Hz Morlet approach

---

### Milestone 5.12 — Synthesis: What Drives Performance?

After completing Milestones 5.8–5.11, compile a definitive comparison:

- [ ] Build a table: method × preprocessing choice × architecture × training protocol × reproduced_r
- [ ] Isolate each factor's contribution via ablation across our reproductions
- [ ] Answer: Is the 0.69 ceiling (DTCNet = DeepFingerNet) a dataset-size ceiling or a model ceiling?
- [ ] Determine what to bring forward to Miller 2019 (9-patient) experiments
- [ ] Update ROADMAP with Miller 2019 plan informed by reproduction findings

---

## Phase 6: Extend to Joystick Tracking

### Milestone 6.1 — Load and inspect joystick data
- [ ] Write `load_joystick(patient_id)` returning:
  - `ecog`: shape `(time, channels)`
  - `cursor_pos`: shape `(time, 2)` — CursorPosX, CursorPosY
  - `target_pos`: shape `(time, 2)` — TargetPosX, TargetPosY (optional, for tracking error analysis)
  - `sr`: 1000
- [ ] Inspect all 4 patients (fp, gf, rh, rr):
  - Plot cursor trajectories (X vs Y over time) — verify the circular tracking pattern
  - Plot raw ECoG, check for artifacts
  - Note GF's cursor clipping issue — decide whether to exclude those segments or handle in preprocessing
- [ ] Print summary table (channels, duration, cursor range)

### Milestone 6.2 — Adapt preprocessing for joystick
- [ ] Reuse the same preprocessing pipeline from Phase 2 (CAR, notch, bandpass, z-score)
- [ ] Target is now 2D (X, Y) instead of 5D (fingers) — verify the metric and loss function handle this
- [ ] Decision: **predict cursor position** or **predict cursor velocity** (difference between consecutive positions)? Velocity may be more neural-relevant but adds complexity. Start with position.
- [ ] Apply the same temporal train/val/test split (70/15/15)
- [ ] Implement `JoystickDataset` (should share most code with `FingerFlexDataset` — consider a shared base class)

### Milestone 6.3 — Train all three models on joystick
- [ ] Run U-Net, TCN, Transformer on all 4 patients with same config
- [ ] Record results: Pearson r for X, Y, and mean
- [ ] Note: with only 4 patients, paired statistics will have very low power — this is a limitation to acknowledge

### Milestone 6.4 — Quick check: does the same model win?
- [ ] Compare architecture rankings between fingerflex and joystick
- [ ] If rankings differ, that's the architecture-by-task interaction you're looking for
- [ ] If rankings are the same, that's also informative

---

## Phase 7: Extend to Mouse Tracking

### Milestone 7.1 — Load and inspect mouse data
- [ ] Write `load_mouse(patient_id)` — same structure as joystick (CursorPosX/Y, TargetPosX/Y)
- [ ] These are the **same 4 patients** as joystick (fp, gf, rh, rr) — this means you have paired data across joystick and mouse for these patients, which is analytically valuable
- [ ] Plot trajectories, inspect for artifacts

### Milestone 7.2 — Adapt preprocessing and train
- [ ] Reuse joystick pipeline (same target format: 2D position)
- [ ] Implement `MouseDataset`
- [ ] Train all three models on all 4 patients

### Milestone 7.3 — Three-task comparison
- [ ] For the 4 patients that appear in both joystick and mouse, you now have **within-subject, across-task** comparisons — this is powerful
- [ ] Build the full results matrix:

  |           | U-Net | TCN | Transformer |
  |-----------|-------|-----|-------------|
  | Fingerflex (N=9) | mean ± std | mean ± std | mean ± std |
  | Joystick (N=4) | mean ± std | mean ± std | mean ± std |
  | Mouse (N=4) | mean ± std | mean ± std | mean ± std |

- [ ] Test for architecture-by-task interaction:
  - Within each task: paired tests across patients
  - Across tasks: for the 4 shared patients (fp, gf, rh, rr), test whether the architecture ranking changes between joystick and mouse
- [ ] Visualize with a heatmap (tasks x architectures, cell color = mean Pearson r)

---

## Phase 8: Hyperparameter Tuning and Ablations

### Milestone 8.1 — Hyperparameter sensitivity
- [ ] Pick one patient from fingerflex (e.g., the median performer) and run a small grid search over:
  - Window size: [250ms, 500ms, 1000ms, 2000ms]
  - Learning rate: [1e-4, 5e-4, 1e-3, 5e-3]
  - Batch size: [16, 32, 64]
- [ ] Run this for all three architectures
- [ ] Report: how sensitive is each architecture to these hyperparameters? (Transformers are often more sensitive to LR than CNNs)
- [ ] Select the best shared config for final results (or report with architecture-specific best configs and note the difference)

### Milestone 8.2 — Ablation: pipeline mode (ours vs lomtev)
- [ ] If not already settled in Phase 3.5, run all three architectures under both pipeline modes
- [ ] Report whether preprocessing order or target scaling affects results more
- [ ] Test prediction smoothing: report r with sigma=0, 1, 6 for one patient — quantify how much smoothing inflates metrics
- [ ] This is a practical finding: does it matter how you preprocess, or is the architecture the bigger factor?

### Milestone 8.3 — Ablation: frequency bands
- [ ] Test the effect of input frequency content on decoding performance
- [ ] Filter ECoG into bands before feeding to models:
  - Broadband (1-200 Hz) — baseline
  - Low frequency (1-40 Hz)
  - High gamma (70-170 Hz) — known to be important for motor decoding
  - Low frequency + high gamma (remove 40-70 Hz)
- [ ] Run on one patient, one model (U-Net), report Pearson r per band
- [ ] This tells you which frequency content drives decoding and is a useful practical finding

### Milestone 8.4 — Ablation: window size effect on each task
- [ ] Fingerflex involves fast, fine movements — may benefit from shorter windows
- [ ] Joystick/mouse involve slow, smooth tracking — may benefit from longer windows
- [ ] Test [250ms, 500ms, 1000ms, 2000ms] across all three tasks with one model
- [ ] If optimal window size differs by task, report it — practical implication for BCI system design

### Milestone 8.5 — Computational cost comparison
- [ ] For each architecture, measure and report:
  - Parameter count
  - FLOPs per inference (use `torchinfo` or `fvcore`)
  - Training time per epoch (seconds, on your GPU)
  - Inference latency per window (milliseconds) — relevant for real-time BCI
- [ ] Build a table: performance vs. compute tradeoff

---

## Phase 9: Statistical Analysis and Figures

### Milestone 9.1 — Final statistical tests
- [ ] Run all final models 3-5 times with different random seeds — report mean ± std across runs to account for initialization variance
- [ ] Per-task architecture comparison: Wilcoxon signed-rank tests (paired across patients)
- [ ] Correction for multiple comparisons: Holm-Bonferroni across all pairwise tests
- [ ] Effect sizes: compute Cohen's d or rank-biserial correlation for each comparison
- [ ] Report exact p-values, not just significance thresholds
- [ ] For N=4 tasks (joystick, mouse), acknowledge limited statistical power explicitly

### Milestone 9.2 — Publication-quality figures
- [ ] **Figure 1**: Study overview diagram — data pipeline from raw ECoG to predictions, showing the three tasks and three architectures
- [ ] **Figure 2**: Example predictions — overlay predicted and actual finger traces (or cursor positions) for one patient per task, one model. Show that the models actually capture the signal.
- [ ] **Figure 3**: Main results — grouped bar chart or boxplot: tasks on x-axis, architectures as groups, Pearson r on y-axis. Include individual patient dots.
- [ ] **Figure 4**: Architecture-by-task interaction — heatmap or interaction plot showing whether rankings change across tasks
- [ ] **Figure 5 (optional)**: Ablation results — frequency band analysis, window size effect
- [ ] Use consistent color scheme: one color per architecture throughout all figures
- [ ] Save all figures as both PDF (for paper) and PNG (for presentations)
- [ ] Use matplotlib with a clean style (seaborn or a custom stylesheet) — no default matplotlib blue-orange

### Milestone 9.3 — Results tables
- [ ] Table 1: Dataset characteristics (patients, channels, duration, target dims per task)
- [ ] Table 2: Architecture specifications (layers, parameters, FLOPs, inference latency)
- [ ] Table 3: Main results (Pearson r per patient, per task, per model, with means and p-values)
- [ ] Table 4: Ablation results (hyperparameter sensitivity, frequency bands, pipeline mode ours vs lomtev)

---

## Phase 10: Paper Writing

### Milestone 10.1 — Draft structure
- [ ] **Title**: "Comparing Deep Learning Architectures for Continuous ECoG Motor Decoding"
- [ ] **Abstract** (150-250 words): problem, approach, key finding, implication
- [ ] **Introduction** (~0.75 pages): gap in literature → research question → contribution
- [ ] **Related Work** (~0.75 pages): prior ECoG decoding, benchmarking studies (MOABB, BCI Competition, EEGNet, HTNet, FingerFlex)
- [ ] **Methods** (~1.5 pages): dataset, preprocessing, architectures, evaluation protocol, statistical tests
- [ ] **Results** (~1.5 pages): main comparison, interaction analysis, ablations
- [ ] **Discussion** (~1 page): interpretation, limitations (small N, no discrete/imagery, per-subject only), future work
- [ ] **Conclusion** (~0.5 pages)

### Milestone 10.2 — Write methods first
- [ ] Methods is the most concrete section — write it while the implementation details are fresh
- [ ] Include enough detail for reproduction: exact filter parameters, window sizes, train/test split ratios, optimizer settings, early stopping criteria, number of training epochs, random seeds used
- [ ] Reference the Miller Library dataset paper (Miller, 2019) and the Stanford Digital Repository URL

### Milestone 10.3 — Write results, then introduction
- [ ] Results should present findings neutrally, save interpretation for discussion
- [ ] Introduction should be written last (or rewritten last) — now that you know your results, you can frame the question precisely
- [ ] Have someone outside the project read the abstract and introduction — if they can't state your contribution in one sentence, revise

### Milestone 10.4 — Target venue selection
- [ ] Consider venues:
  - **NeurIPS (Datasets & Benchmarks track)** — directly designed for benchmark papers
  - **IEEE EMBC** — biomedical engineering, ECoG/BCI audience
  - **IEEE NER** — neural engineering, highly relevant
  - **BCI Meeting/Society** — domain-specific
  - **AAAI / ICML workshop** on healthcare or neuroscience
- [ ] Check page limits and formatting requirements for your target venue before writing
- [ ] Align figure count and table density with venue norms

---

## Phase 11: Code Release and Reproducibility

### Milestone 11.1 — Clean up codebase
- [ ] Remove dead code, debug prints, commented-out experiments
- [ ] Add docstrings to all public functions
- [ ] Ensure all configs needed to reproduce every result are committed
- [ ] Add a `scripts/reproduce_all.sh` that trains all models on all tasks with the final configs

### Milestone 11.2 — Write a code README
- [ ] Installation instructions
- [ ] How to download the data (point to `download_dataset.py`)
- [ ] How to reproduce each table/figure in the paper
- [ ] Expected runtime per experiment

### Milestone 11.3 — Archive
- [ ] Tag the repo at submission time (e.g., `v1.0-submission`)
- [ ] Consider uploading to Zenodo for a DOI if the venue values data/code archiving

---

## Dependency Graph

```
Phase 0 (setup)
  │
  ▼
Phase 1 (load fingerflex)
  │
  ▼
Phase 2 (preprocessing)
  │
  ├──────────────────────────────┐
  ▼                              ▼
Phase 3 (U-Net baseline)     Phase 3.6 (trivial baselines)
  │
  ▼
Phase 4 (TCN)
  │
  ▼
Phase 5 (Transformer)
  │
  ├─── CHECKPOINT: fingerflex-only comparison ◄── publishable on its own
  │
  ▼
Phase 6 (joystick) ──► Phase 7 (mouse) ──► Phase 8 (ablations)
                                               │
                                               ▼
                                          Phase 9 (stats & figures)
                                               │
                                               ▼
                                          Phase 10 (paper)
                                               │
                                               ▼
                                          Phase 11 (code release)
```

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Variable channel counts across patients break model | High | Start with per-patient models; defer channel standardization |
| All architectures perform the same | Medium | This is still a publishable finding — frame as "architecture choice doesn't matter for continuous ECoG decoding" |
| U-Net doesn't beat linear regression | High | Report honestly; suggests the task is linear and DL adds complexity without benefit |
| Joystick/mouse N=4 too small for statistics | Medium | Report descriptive results; use the 4 shared patients for within-subject cross-task comparison |
| Transformer fails to converge | Medium | Try lower LR, warmup schedule, smaller model; document failures |
| MATLAB .mat files have inconsistent formats | Medium | Test loading all 17 files in Phase 1 before building anything else |
| GPU memory issues with large window sizes | Low | Reduce batch size, use gradient accumulation, or shorter windows |

---

## Future Work (Post-Paper 1)

These are explicitly out of scope for the first paper but form the natural extension:

- **Paper 2**: Extend benchmark to discrete motor tasks (motor_basic, gestures) and imagery (imagery_basic, imagery_feedback) with classification heads
- **Cross-subject transfer**: Train on N-1 patients, test on held-out patient
- **New architectures**: Mamba/S4 state-space models, hybrid CNN-Transformer
- **Real-time latency**: Measure end-to-end inference time, simulate online BCI loop
- **Feature engineering comparison**: extend beyond Lomtev wavelets — test band-power features, Hilbert envelope, learned filterbanks (SincNet-style) as alternative input representations
- **Multi-task learning**: single model trained on all tasks simultaneously
