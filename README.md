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

## Dataset Contents

| File | Size | Description |
|------|------|-------------|
| `BCI_Competion4_dataset4_data_fingerflexions.zip` | 222 MB | BCI Competition IV finger flexion data |
| `ctmr.zip` | 97 MB | Cortical surface reconstructions |
| `faces_basic.zip` | 585 MB | Face perception experiment |
| `faces_noise.zip` | 895 MB | Face-in-noise perception experiment |
| `fingerflex.zip` | 554 MB | Finger flexion experiment |
| `fixation_PAC.zip` | 205 MB | Fixation phase-amplitude coupling |
| `fixation_highfreq.zip` | 333 MB | Fixation high-frequency analysis |
| `fixation_pwrlaw.zip` | 370 MB | Fixation power-law analysis |
| `gestures.zip` | 803 MB | Gesture experiment |
| `imagery_basic.zip` | 544 MB | Motor imagery experiment |
| `imagery_feedback.zip` | 254 MB | Motor imagery with feedback |
| `joystick_track.zip` | 135 MB | Joystick tracking experiment |
| `loc.zip` | 67 MB | Localizer experiment |
| `memory_nback.zip` | 522 MB | N-back memory experiment |
| `motor_basic.zip` | 815 MB | Basic motor experiment |
| `mouse_track.zip` | 116 MB | Mouse tracking experiment |
| `speech_basic.zip` | 196 MB | Basic speech experiment |
| `speech_lists.zip` | 554 MB | Speech lists experiment |
| `toolbox.zip` | 0.1 MB | MATLAB analysis toolbox |
| `visual_search.zip` | 154 MB | Visual search experiment |
| `xs_files.zip` | 21 MB | Cross-experiment summary files |
| `kjm_ECoGLibrary_PatientTaskTable.pdf` | 0.1 MB | Patient/task reference table |
| `MANUSCRIPT_kjm_ECoGLibrary_v14_FiguresIncluded.pdf` | 8.6 MB | Full manuscript with figures |

**Total: ~7.5 GB**

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
