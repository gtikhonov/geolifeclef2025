# GeoLifeCLEF 2025 - Core Analytical Pipeline

This directory contains the primary source code and modeling pipelines for the GeoLifeCLEF 2025 competition. The methodology integrates Deep Learning (DL) feature extraction with Joint Species Distribution Modeling (JSDM) using the HMSC framework.

## Directory Structure

- **`prithvi/`**: Implementation of deep learning models. Includes dataset loaders for multi-modal data (Sentinel-2 patches, Landsat/Bioclim cubes) and model architectures built on the Prithvi-EO-2.0 foundation model.
- **`hmsc/`**: R and Python/TensorFlow scripts for Hierarchical Modeling of Species Communities. This handles species co-occurrence patterns and spatial random effects.
- **`preprocess/`**: Notebooks and scripts for data cleaning, Earth Engine imports (WorldCover), and preparation of covariates.
- **`examples/`**: Reference notebooks providing baselines for different data modalities.
- **`misc/`**: Utility scripts for F1 optimization and miscellaneous experiments.
- **`config.py`**: Centralized configuration management script.

## Connectivity and Workflow

The project follows a modular two-stage pipeline:

1.  **Preprocessing**: Data is sourced from the competition datasets and auxiliary sources (like Google Earth Engine).
2.  **Feature Extraction & DL Modeling (`prithvi/`)**: Multimodal deep neural networks are trained to extract high-level features or predict species directly.
3.  **Joint Modeling (`hmsc/`)**: DL-extracted features are passed to HMSC models to incorporate inter-species dependencies and spatial structures, often using GPU-accelerated Gibbs sampling on HPC clusters.
4.  **Ensembling**: Final predictions are combined from multiple model variants to maximize the F1 score.

## Configuration

Paths are managed centrally to ensure portability across different environments (local desktop vs. HPC clusters).

### Setup

1.  **`config.json`**: Create a `config.json` file in this directory based on `config.json.template`. This file is ignored by Git to protect local environment details.
2.  **Required Variables**:
    - `GLC_DATA_PATH`: Root directory of the GeoLifeCLEF 2025 data.
    - `GLC_SCRATCH_PATH`: Directory for saving intermediate outputs and model weights.
    - `HMSC_HPC_PATH`: Path to the external `hmsc-hpc` library for TensorFlow acceleration.
    - `PRITHVI_WEIGHTS_PATH`: Directory where the Prithvi model weights (`.pt`) are stored.

### Usage in Code

Python scripts should import paths from `config.py`:
```python
from config import DATA_PATH, SCRATCH_PATH
```

R scripts retrieve the data path via environment variables:
```R
path_data = Sys.getenv("GLC_DATA_PATH")
```
