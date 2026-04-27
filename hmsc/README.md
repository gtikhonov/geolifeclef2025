# HMSC Joint Species Distribution Modeling

This module implements the **Joint Species Distribution Modeling (JSDM)** component of the pipeline using the **HMSC (Hierarchical Modeling of Species Communities)** framework. It is designed to capture species-to-species associations and spatial dependencies that independent deep learning models may miss.

## Pipeline Workflow

The modeling process is split into four distinct phases:

### 1. Covariate & Study Design Preparation (`hmsc_prepare.ipynb`)
-   **Feature Source**: HMSC can use either raw environmental variables (`orig`) or deep embeddings (`deep`) extracted from the `prithvi` module.
-   **Spatial Clustering**: To manage spatial random effects at scale, survey locations are clustered using K-means (typically $k=100, 200, 400$). These clusters serve as levels in the HMSC study design.
-   **PO Aggregation**: `hmsc_aggregate_po.ipynb` aggregates massive Presence-Only data into grid cells or environmental bins to make joint modeling computationally feasible.

### 2. Model Initialization (R scripts)
-   **`init.R`**: Sets up a standard PA model. It defines the `probit` distribution, study design (spatial clusters), and priors for latent factors.
-   **`init_po.R`**: Sets up a joint PA + PO model. It incorporates the aggregated PO data and defines a "type" factor to distinguish between the two data sources.
-   **Export**: Models are serialized to JSON-formatted RDS files using the `jsonify` package for compatibility with the Python sampler.

### 3. Accelerated Fitting (`fit_hmsc.ipynb` / `mahti_fit_hmsc.sh`)
-   The heavy lifting is performed by the **`hmsc-hpc`** library (TensorFlow-based).
-   **GPU Acceleration**: Uses a Gibbs sampler implemented in TensorFlow to generate posterior samples for fixed effects ($\beta$) and latent factors ($\lambda$).
-   **HPC Support**: `mahti_fit_hmsc.sh` provides a SLURM template for running long-chain samplers on high-performance clusters.

### 4. Prediction & Optimization
-   **`load_fit.R` / `load_fit_po.R`**: These R scripts import the TensorFlow posterior samples and generate species presence probabilities for the test set.
-   **`hmsc_load.ipynb`**: Converts raw HMSC probabilities into optimized species lists using the **Expected F1-Score Maximization** logic (sharing the same core algorithm as the `prithvi` module).

## Key Parameters
-   `ns`: Number of species modeled (often restricted to the most frequent $N$ species for stability).
-   `np`: Number of spatial units (K-means centroids).
-   `nf`: Number of latent factors (capturing species associations).
-   `samN` / `thinN`: Number of MCMC samples and the thinning interval.

## Requirements
-   **R**: `Hmsc`, `jsonify`, `abind`.
-   **Python**: `tensorflow`, `tensorflow-probability`.
-   **External**: The `hmsc-hpc` repository must be available and its path set in `config.json`.
