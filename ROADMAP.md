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
- [ ] Study TCN literature (Bai et al., 2018 "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling")
- [ ] Key components:
  - Causal dilated 1D convolutions (dilation factors: 1, 2, 4, 8, ...)
  - Residual connections
  - Weight normalization or batch normalization
  - Dropout between layers
- [ ] Implement in `src/models/tcn.py`
- [ ] **Match parameter count** approximately to U-Net — this is critical for fair comparison. Adjust depth/width until parameter counts are within ~20% of each other. Document the final counts.
- [ ] Verify forward pass with same input shapes as U-Net

### Milestone 4.2 — Train TCN on all 9 fingerflex patients
- [ ] Use identical training config as U-Net (same optimizer, LR, scheduler, early stopping, batch size)
- [ ] Only architecture-specific hyperparameters should differ (kernel size, dilation, number of layers)
- [ ] Record the same results table as Milestone 3.5
- [ ] Compare against U-Net per-patient using paired differences

### Milestone 4.3 — Quick sanity comparison
- [ ] For each patient, compute: TCN_r - UNet_r
- [ ] Does TCN consistently beat U-Net? Lose to it? Mixed?
- [ ] Run Wilcoxon signed-rank test on the 9 paired differences (one per patient)
- [ ] This is a preliminary signal — if there's zero difference, that's informative for the paper's narrative

---

## Phase 5: Third Architecture — Transformer on Fingerflex

### Milestone 5.1 — Implement lightweight Transformer encoder
- [ ] Design choices to document and justify:
  - **Positional encoding**: sinusoidal (Vaswani) vs. learnable vs. relative (consider that ECoG has meaningful temporal structure)
  - **Input embedding**: linear projection of each time step's channel vector, or patch-based (group N time steps into a token)
  - **Architecture**: encoder-only (no decoder needed for regression). Stack of TransformerEncoderLayers.
  - **Output head**: global average pooling → linear, or use [CLS] token, or predict per-timestep
  - **Number of heads, layers, d_model**: start small (2 layers, 4 heads, d_model=64-128) and scale up
- [ ] Implement in `src/models/transformer.py`
- [ ] **Match parameter count** to U-Net and TCN
- [ ] Verify forward pass, check attention maps are reasonable (not all uniform or all on one position)

### Milestone 5.2 — Train Transformer on all 9 fingerflex patients
- [ ] Identical training config to U-Net and TCN
- [ ] Record results table
- [ ] Transformers may need different learning rate or warmup — if default config fails to converge, try:
  - Lower LR (1e-4 instead of 1e-3)
  - Linear warmup for first 10% of training steps
  - Document any deviations from the shared config and justify them

### Milestone 5.3 — Architecture comparison on fingerflex
- [ ] Combine results into a single table (using the winning pipeline from 3.5):

  | Patient | U-Net r | TCN r | Transformer r | Lomtev U-Net (spectrograms) r |
  |---------|---------|-------|---------------|-------------------------------|
  | bp      |         |       |               |                               |
  | ...     |         |       |               |                               |
  | **Mean ± Std** |  |       |               |                               |
  | **Ref: Lomtev (BCI-IV)** | — | — | — | 0.55 (mean), 0.68 (sub3) |

- [ ] Paired Wilcoxon signed-rank tests for each pair: UNet vs TCN, UNet vs Transformer, TCN vs Transformer
- [ ] Apply Holm-Bonferroni correction for 3 comparisons
- [ ] Plot: grouped bar chart or boxplot showing distribution of Pearson r across patients for each model
- [ ] Plot: per-patient scatter (e.g., U-Net r vs. TCN r) to visualize whether the same patients are easy/hard across models
- [ ] Include Lomtev spectrogram baseline as a reference in all plots
- [ ] The Lomtev column answers: is the spectrogram representation consistently better/worse than raw ECoG across architectures?

**This is your first checkpoint — the fingerflex-only comparison. If you're under time pressure, this alone is a publishable result.**

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
