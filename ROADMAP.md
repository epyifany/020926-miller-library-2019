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
- [ ] Set up per-epoch CSV logging (will implement with training loop in Phase 3)

### Milestone 0.3 — Version control hygiene
- [x] `.gitignore` covers data/, results/, *.pt, *.ckpt, wandb/, symlinks
- [x] Data and results symlinked to `/mnt/archive/` (300 GB) — see CLAUDE.md Storage Rules
- [ ] Make an initial commit with project skeleton
- [ ] Decide on logging: CSV to start, W&B optional later

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
- [ ] Plot raw ECoG traces (5-10 seconds) for 2-3 patients — check for artifacts, saturation
- [ ] Plot 5 finger flexion signals over time — understand movement structure
- [ ] Plot stimulus/cue codes over time — understand trial structure
- [ ] Compute and plot PSD for a few channels — check filtering status
- [x] Check for NaNs, Infs, or constant channels — all clean (done in 1.1)
- [ ] Save plots to `results/eda/` for reference

### Milestone 1.3 — Understand the BCI Competition IV format
- [ ] Compare BCI Competition data to Miller Library fingerflex — determine if same patients
- [ ] This is optional/deferred — we have enough to proceed with Miller data directly

---

## Phase 2: Preprocessing Pipeline (Fingerflex)

### Milestone 2.1 — Design the preprocessing pipeline
- [ ] Document every preprocessing step and its justification. The pipeline should be:
  1. **Load raw data** (Milestone 1.1)
  2. **Common Average Reference (CAR)** — subtract mean across channels at each time point. The Miller Library already provides `car.m`; replicate this exactly in Python
  3. **Notch filter at 60 Hz** — verify if already applied; if not, apply a notch filter at 60, 120, 180 Hz (harmonics)
  4. **Bandpass filter** — verify the existing 0.15-200 Hz hardware filter is sufficient; optionally apply a tighter software bandpass (e.g., 1-200 Hz) to remove residual DC drift
  5. **Normalize ECoG** — z-score per channel (subtract mean, divide by std) computed on training set only
  6. **Normalize targets** — z-score per finger, computed on training set only. Store the mean/std for inverse transform at evaluation time
  7. **Windowing** — segment continuous data into overlapping windows (e.g., 1-second windows with 100ms hop → 10 Hz output rate). Window size is a hyperparameter.
- [ ] Write unit tests: verify CAR output has zero mean across channels, verify z-score output has mean≈0 std≈1, verify windowing produces correct number of segments

### Milestone 2.2 — Implement the preprocessing pipeline
- [ ] `src/data/preprocessing.py`:
  - `apply_car(ecog)` → np.ndarray
  - `apply_notch(ecog, sr, freqs=[60, 120, 180])` → np.ndarray
  - `apply_bandpass(ecog, sr, lo, hi)` → np.ndarray
  - `normalize(data, mean=None, std=None)` → (np.ndarray, mean, std)
- [ ] `src/data/windowing.py`:
  - `create_windows(ecog, targets, window_size, hop_size)` → (windows, labels)
  - windows shape: `(n_windows, window_size, channels)`
  - labels shape: `(n_windows, n_targets)` or `(n_windows, window_size, n_targets)` depending on whether you predict one output per window or a full sequence
- [ ] All preprocessing parameters should come from config, not hardcoded

### Milestone 2.3 — Train/test split strategy
- [ ] **Within-subject, temporal split**: For each patient, split the continuous recording into contiguous blocks:
  - First 70% → training
  - Next 15% → validation
  - Last 15% → test
- [ ] **Critical: no shuffling of time points.** This is a time series — shuffling would leak future information into training. Use contiguous blocks only.
- [ ] If the BCI Competition IV split maps to these patients, consider using their exact train/test boundary for comparability
- [ ] Store split indices (not the data itself) so they're reproducible
- [ ] Write a `src/data/splits.py` module that returns train/val/test indices given a patient ID and config

### Milestone 2.4 — PyTorch Dataset and DataLoader
- [ ] Implement `FingerfFlexDataset(patient_id, split, config)` extending `torch.utils.data.Dataset`
  - `__getitem__` returns `(ecog_window, target)` as float32 tensors
  - Preprocessing is applied on init (or cached to disk after first run)
- [ ] Handle variable channel counts:
  - **Option A (recommended to start):** Train per-patient models with patient-specific input dimensions — simplest, no interpolation needed
  - **Option B (later):** Zero-pad to max channels (64) with a channel mask
  - **Option C (later):** Select a fixed subset of channels based on anatomical region (e.g., only sensorimotor cortex)
- [ ] Verify DataLoader produces correct shapes: `(batch, window_size, channels)` for ECoG, `(batch, n_targets)` for targets
- [ ] Test on all 9 patients — confirm no crashes, print shapes

---

## Phase 3: Baseline Model — U-Net on Fingerflex

### Milestone 3.1 — Implement U-Net architecture
- [ ] Study the FingerFlex U-Net paper (Lomtev, 2020) — document the exact architecture:
  - Input: `(batch, channels, time)` — note: Conv1d expects channel-first
  - Encoder: series of Conv1d + BatchNorm + ReLU + MaxPool1d blocks
  - Bottleneck
  - Decoder: series of ConvTranspose1d + skip connections from encoder
  - Output: `(batch, n_fingers, time)` or `(batch, n_fingers)` depending on output strategy
- [ ] Implement in `src/models/unet.py`
- [ ] Print model summary: parameter count, input/output shapes
- [ ] Verify forward pass works with a random tensor matching one patient's dimensions
- [ ] Target parameter count: note it for later comparison with TCN and Transformer (important for fair comparison)

### Milestone 3.2 — Training loop
- [ ] Implement `src/training/trainer.py` with:
  - Loss function: MSE (standard for regression)
  - Optimizer: Adam with configurable learning rate (start with 1e-3)
  - Learning rate scheduler: ReduceLROnPlateau or CosineAnnealing (configurable)
  - Early stopping: patience of N epochs on validation loss
  - Gradient clipping: max norm 1.0 (ECoG data can have scale issues)
  - Checkpoint saving: save best model (by validation loss) and last model
- [ ] Log per-epoch: training loss, validation loss, validation Pearson r (per finger and averaged)
- [ ] Save training curves to `results/{experiment_id}/`

### Milestone 3.3 — Evaluation metrics
- [ ] Implement `src/evaluation/metrics.py`:
  - `pearson_r(predicted, actual)` — Pearson correlation coefficient per target dimension
  - `mean_pearson_r(predicted, actual)` — average across target dimensions (fingers or X/Y)
  - `mse(predicted, actual)` — for monitoring, not the primary metric
  - `r_squared(predicted, actual)` — coefficient of determination, useful secondary metric
- [ ] For Pearson r, compute on the **full test set continuously** (not averaged across windows) — reconstruct the continuous prediction from overlapping windows, then correlate with ground truth
- [ ] Write unit tests: perfect prediction → r=1.0, random prediction → r≈0, inverted prediction → r=-1.0

### Milestone 3.4 — Train U-Net on one patient
- [ ] Pick patient `bp` (or whichever has the most data/cleanest signals from exploration)
- [ ] Train with default hyperparameters
- [ ] Plot: training/validation loss curves, predicted vs. actual finger traces for test set
- [ ] Compute Pearson r per finger and averaged
- [ ] Sanity check: does the model clearly beat a trivial baseline? (e.g., predicting the mean, or predicting the previous time step)
- [ ] If Pearson r < 0.3 on the best finger, something is likely wrong — debug before proceeding

### Milestone 3.5 — Train U-Net on all 9 fingerflex patients
- [ ] Write a loop/script that trains on each patient independently with identical config
- [ ] Store results in a table:

  | Patient | Channels | r (thumb) | r (index) | r (middle) | r (ring) | r (little) | r (mean) |
  |---------|----------|-----------|-----------|------------|----------|------------|----------|
  | bp      | 46       |           |           |            |          |            |          |
  | cc      | 63       |           |           |            |          |            |          |
  | ...     | ...      |           |           |            |          |            |          |

- [ ] Compute group statistics: mean ± std of mean Pearson r across patients
- [ ] Identify any patients that perform much worse — investigate (bad data? too few channels? electrode placement away from motor cortex?)
- [ ] Compare against BCI Competition IV published results if applicable
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

### Milestone 5.3 — Three-way comparison on fingerflex
- [ ] Combine results into a single table:

  | Patient | U-Net r | TCN r | Transformer r |
  |---------|---------|-------|---------------|
  | bp      |         |       |               |
  | ...     |         |       |               |
  | **Mean**|         |       |               |
  | **Std** |         |       |               |

- [ ] Paired Wilcoxon signed-rank tests for each pair: UNet vs TCN, UNet vs Transformer, TCN vs Transformer
- [ ] Apply Holm-Bonferroni correction for 3 comparisons
- [ ] Plot: grouped bar chart or boxplot showing distribution of Pearson r across patients for each model
- [ ] Plot: per-patient scatter (e.g., U-Net r vs. TCN r) to visualize whether the same patients are easy/hard across models

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

### Milestone 8.2 — Ablation: frequency bands
- [ ] Test the effect of input frequency content on decoding performance
- [ ] Filter ECoG into bands before feeding to models:
  - Broadband (1-200 Hz) — baseline
  - Low frequency (1-40 Hz)
  - High gamma (70-170 Hz) — known to be important for motor decoding
  - Low frequency + high gamma (remove 40-70 Hz)
- [ ] Run on one patient, one model (U-Net), report Pearson r per band
- [ ] This tells you which frequency content drives decoding and is a useful practical finding

### Milestone 8.3 — Ablation: window size effect on each task
- [ ] Fingerflex involves fast, fine movements — may benefit from shorter windows
- [ ] Joystick/mouse involve slow, smooth tracking — may benefit from longer windows
- [ ] Test [250ms, 500ms, 1000ms, 2000ms] across all three tasks with one model
- [ ] If optimal window size differs by task, report it — practical implication for BCI system design

### Milestone 8.4 — Computational cost comparison
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
- [ ] Table 4: Ablation results (hyperparameter sensitivity, frequency bands)

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
- **Feature engineering comparison**: raw ECoG vs. spectrograms vs. wavelet features as model input
- **Multi-task learning**: single model trained on all tasks simultaneously
