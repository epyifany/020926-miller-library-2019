# Miller ECoG Library (2019)

A download utility for **"A library of human electrocorticographic data and analyses"** by Kai J. Miller, hosted on the [Stanford Digital Repository](https://purl.stanford.edu/zk881ps0522).

The library contains 204 individual ECoG datasets from 34 patients across 16 behavioral experiments, recorded with the same amplifiers at the same sampling rate and filter settings. Electrode positions are registered to brain anatomy, and MATLAB analysis scripts are included.

## Quick Start

```bash
python download_dataset.py
```

This downloads all 23 files (~7.5 GB) into a `data/` directory and extracts all `.zip` archives.

**Requirements:** Python 3.6+ (no external dependencies).

## Usage

```bash
# Download and extract everything
python download_dataset.py

# Download to a custom directory
python download_dataset.py --output /path/to/data

# Download without extracting
python download_dataset.py --no-unzip

# Interactively pick which files to download
python download_dataset.py --select

# Verify integrity of previously downloaded files
python download_dataset.py --verify-only

# Skip MD5 verification to speed up downloads
python download_dataset.py --skip-verify
```

Re-running the script is safe — it skips files that have already been downloaded.

## Dataset Overview

The extracted data is organized into **16 experiment folders**, **3 tool packages**, and **2 PDF documents**. Experiments share a consistent directory layout: `data/` for ECoG recordings, `brains/` for patient MRIs (`.nii`), `locs/` for electrode coordinates (`*_xslocs.mat`), `dc_files/` for data-cleaning scripts, `figs/` for figure-generation scripts, and `ref/` for reference papers. Most experiment folders also include a `README_*_dataset_notes.docx` with experiment-specific details.

### Experiments

| Experiment | Size | Patients | Description |
|---|---:|---:|---|
| `motor_basic` | 815 MB | 19 | Overt hand and tongue motor movements |
| `faces_noise` | 895 MB | ~14 | Face perception with noise-degraded stimuli |
| `gestures` | 803 MB | 5 | Hand gesture execution and observation |
| `faces_basic` | 585 MB | 14 | Face vs. house visual categorization |
| `speech_lists` | 554 MB | 3 | Extended spoken word lists (nouns/verbs, multiple runs per patient) |
| `fingerflex` | 554 MB | 9 | Individual finger flexion movements |
| `imagery_basic` | 544 MB | 7 | Motor imagery vs. overt movement (paired `*_im_t_h` / `*_mot_t_h` files) |
| `memory_nback` | 522 MB | 4 | N-back working memory task |
| `fixation_pwrlaw` | 370 MB | 20 | Resting-state power-law spectral analysis |
| `fixation_highfreq` | 333 MB | 4 | High-frequency (10 kHz) resting-state recordings |
| `imagery_feedback` | 254 MB | 4 | Motor imagery with real-time feedback |
| `fixation_PAC` | 205 MB | 11 | Resting-state phase-amplitude coupling |
| `speech_basic` | 196 MB | 11 | Spoken noun and verb production |
| `visual_search` | 154 MB | 5 | Visual search task |
| `joystick_track` | 135 MB | 4 | Continuous joystick tracking |
| `mouse_track` | 116 MB | 4 | Continuous mouse tracking |

### Tools and Support Files

| File | Size | Description |
|---|---:|---|
| `toolbox` | 0.1 MB | 98 MATLAB functions for signal processing, statistics, and visualization |
| `ctmr` | 97 MB | Cortical surface mapping toolkit (3,388 MATLAB functions, SPM12, DICOM-to-NIfTI converter) |
| `loc` | 67 MB | Electrode localization tools, brain templates (`halfbrains.mat`, `wholebrain.mat`), example patient X-rays, and tutorial |
| `xs_files` | 21 MB | Cross-experiment visualization tools (`xs_anat.m`, `xs_disp.m`, `xs_loc.m`, etc.) and full thesis reference |
| `BCI_Competion4_dataset4_data_fingerflexions` | 222 MB | BCI Competition IV dataset 4 — 3 subjects with train/test splits (derived from `fingerflex`) |

### Documents

| File | Description |
|---|---|
| `kjm_ECoGLibrary_PatientTaskTable.pdf` | Patient demographics and task assignment table |
| `MANUSCRIPT_kjm_ECoGLibrary_v14_FiguresIncluded.pdf` | Full manuscript (Nature Human Behaviour, 2019) |

**Total download: ~7.5 GB** (expands to ~15 GB when extracted)

## Directory Structure

After extraction, each experiment folder follows this pattern:

```
data/
├── experiment_name/
│   ├── data/                  # ECoG recordings (.mat)
│   │   ├── bp_mot_t_h.mat    # Direct .mat files, or...
│   │   └── bp/               # ...patient subdirectories
│   │       └── bp_task.mat
│   ├── brains/                # Patient MRI scans (.nii)
│   │   └── bp/
│   │       └── bp_mri.nii
│   ├── locs/                  # Electrode locations
│   │   └── bp_xslocs.mat
│   ├── dc_files/              # Data-cleaning functions (.m)
│   ├── figs/                  # Figure-generation scripts (.m)
│   ├── ref/                   # Reference papers (.pdf)
│   └── README_*_dataset_notes.docx
```

Not all subdirectories are present in every experiment — simpler experiments (e.g., fixation tasks) may only have `data/` and `ref/`.

## File Naming Conventions

Patients are identified by 2-letter lowercase codes (e.g., `bp`, `jc`, `wc`). Data files follow task-specific patterns:

| Pattern | Example | Used in |
|---|---|---|
| `ID_task_t_h.mat` | `bp_mot_t_h.mat` | motor_basic, imagery_basic |
| `ID_wordtype.mat` | `bp_verbs.mat` | speech_basic |
| `ID_wordtype_Ln_Rn.mat` | `jc_nouns_L1_R1.mat` | speech_lists |
| `ID_nback.mat` | `al_nback.mat` | memory_nback |
| `ID_base.mat` | `al_base.mat` | fixation_pwrlaw, fixation_PAC |
| `ID_10kbase.mat` | `s1_10kbase.mat` | fixation_highfreq |
| `ID_taskname.mat` | `jm_vissearch.mat` | visual_search, fingerflex |

Supporting files per patient:
- `ID_xslocs.mat` — electrode coordinates
- `ID_stim.mat` — stimulus timing/parameters
- `ID_mri.nii` — structural MRI

## Toolbox Highlights

The `toolbox/` directory provides 98 MATLAB functions:

- **Signal processing:** `car.m` (common average reference), `getButterFilter.m`, `computepsd.m`, `fftplot.m`, `welch.m`
- **Spectral analysis:** `spectraparse.m`, `logspectraparse.m`, `crossfreqCoh.m`, `cohere.m`
- **Statistics:** `calc_rsqu.m`, `dprime.m`, `roc_curve.m`, `bonf_holm.m`, `bootmean.m`, `information.m`
- **Visualization:** `kjm_errbar.m`, `barpic.m`, `histg.m`, `freezeColors.m`, `exportfig.m`
- **Data parsing:** `matparse.m`, `parseandmark.m`, `staparse.m`
- **Decomposition:** `runica.m`, `runpca.m`

The `ctmr/` directory adds cortical surface rendering, electrode overlay plotting (`ctmr_gauss_plot.m`, `el_add.m`), and a bundled SPM12 installation for neuroimaging analysis.

## Patient Coverage

34 unique patients across experiments. The table below shows how many patients contributed to each experiment:

```
fixation_pwrlaw ████████████████████ 20
motor_basic     ███████████████████  19
faces_basic     ██████████████       14
faces_noise     ██████████████       14
fixation_PAC    ███████████          11
speech_basic    ███████████          11
fingerflex      █████████             9
imagery_basic   ███████               7
gestures        █████                 5
visual_search   █████                 5
memory_nback    ████                  4
imagery_feedbk  ████                  4
joystick_track  ████                  4
mouse_track     ████                  4
fixation_highfq ████                  4
speech_lists    ███                   3
```

## Citation

```bibtex
@article{miller_library_2019,
  title     = {A library of human electrocorticographic data and analyses},
  author    = {Miller, Kai J.},
  journal   = {Nature Human Behaviour},
  volume    = {3},
  number    = {11},
  pages     = {1225--1235},
  year      = {2019},
  doi       = {10.1038/s41562-019-0678-3},
  url       = {https://purl.stanford.edu/zk881ps0522}
}
```

## License

- **Dataset:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (Stanford Digital Repository)
- **This download script:** [MIT](LICENSE)
