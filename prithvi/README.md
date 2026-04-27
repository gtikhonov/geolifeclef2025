# Prithvi Deep Learning Pipeline

This subdirectory implements a multi-modal deep learning pipeline for species distribution modeling. It leverages the **Prithvi-EO-2.0** (300M parameter) foundation model as a backbone for satellite imagery, integrated with time-series analysis and environmental covariates.

## Model Architecture: The Hybrid Approach

The core architecture, `ModifiedPrithviResNet18` (defined in `models.py`), is a late-fusion ensemble of three distinct branches:

1.  **Satellite Branch**: A Prithvi Vision Transformer (ViT) encoder processes 64x64 Sentinel-2 patches. Latent tokens are passed through a `SimpleDecoder` (MLP) to project the high-dimensional representation into a compressed feature vector.
2.  **Time-Series Branch**: A `ModifiedResNet18` processes temporal cubes. The input is an 8-channel tensor comprising 6 Landsat bands and 2 aggregated Bioclimatic variables (Precipitation and Mean Temperature), structured across quarters and years.
3.  **Tabular Branch**: Raw environmental covariates (Elevation, SoilGrids, WorldCover, LandCover, and Snow cover) are concatenated directly with the features extracted from the visual branches.

The fused feature vector is processed by a dropout-stabilized MLP tail to predict probabilities for **11,255 plant species**.

## Data Integration & Preprocessing

### Modalities
- **Sentinel-2**: 4 bands (RGB + NIR) @ 10m resolution. Includes an optional 5th mask channel to handle missing data areas.
- **Landsat**: 6 bands (Red, Green, Blue, NIR, SWIR1, SWIR2) seasonally aggregated into 4 quarters.
- **Bioclimatic Cubes**: Monthly CHELSA data, downsampled and aggregated to match Landsat's seasonal quarters for temporal alignment.
- **Environmental Covariates**: 24+ tabular variables extracted per survey location.

### Augmentations
- **Temporal**: `HorizontalCycleTransform` implements circular shifts along the year/quarter axis to make the model robust to inter-annual variation.
- **Spatial**: Standard random flips, rotations, and resized crops for Sentinel patches.

## Training Strategy
- **Loss**: Binary Cross Entropy with Logits (`BCEWithLogitsLoss`).
- **Optimizer**: AdamW with differential weight decay applied to the MLP tail.
- **Scheduling**: `CosineAnnealingLR` for smooth convergence over 75-100 epochs.
- **Filtering**: Species with very low presence counts (determined by `pa_presence_threshold`) can be filtered or mapped to a "rare" category.

## Inference: Expected F1-Score Maximization

Standard thresholding (e.g., $P > 0.5$) is insufficient for high-dimensional multi-label tasks like GeoLifeCLEF. This pipeline uses a Monte-Carlo approach (`utils.f1_score`) to optimize the output species list:

1.  **Probability Ranking**: Species are ranked by predicted probability.
2.  **Expectation via Sampling**: 400 Bernoulli trials are simulated based on the predicted probabilities.
3.  **Optimal Cutoff**: The algorithm calculates the expected F1-score for every possible list length $k$ (from 1 to 1000) and selects the length $k^*$ that maximizes the expectation.

## File Reference
- **`models.py`**: Multi-modal architectures and the Prithvi wrapper.
- **`glc_datasets.py`**: Complex data loaders that handle multi-modal alignment and normalization.
- **`utils.py`**: F1-optimization logic and general utilities.
- **`nb3.ipynb`**: Primary entry point for fitting models and generating optimized submissions.
