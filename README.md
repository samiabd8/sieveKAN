# sieveKAN: Kolmogorov-Arnold Sieve Networks for High-dimensional Estimation and Inference

## Overview

sieveKAN is a PyTorch implementation of Kolmogorov-Arnold Networks (KAN) designed for the sieve extremum estimation framework. It provides tools for high-dimensional estimation and nonlinear function approximation. The framework includes automated complexity control and (functional) sparsity routines.

## Key Features

* **B-Spline Basis**: Uses flexible B-spline functions for activations.
* **Sample-Size Scaling**: Automatically tunes network width, depth, and grid size based on the number of observations.
* **Functional Sparsity**: Includes edge pruning and projection to maintain model parsimony.
* **Regularization**: Centers around **Group Lasso** regularization.
* **CDF Transformation**: Includes an `EmpiricalCDFTransformer` for feature normalization.

## Requirements

* Python 3.8+
* PyTorch
* NumPy
* Pandas
* Scikit-Learn
* SciPy

## Repository Structure

* **`sieveKAN_sim.py`**: Contains the core `SieveKAN` and `SieveKANLayer` classes. It includes a simulation routine for Double Machine Learning (DML).
* **`sieveKAN_empirical.py`**: A dedicated script for applying the model to macro-finance factor datasets.

## Usage

### 1. Simulation

Run the simulation script to test the model on synthetic high-dimensional data:

```bash
python sieveKAN_sim.py

```

The script outputs metrics for MSE, R², and network sparsity.

### 2. Empirical Analysis

Run the empirical script to process real-world data:

```bash
python sieveKAN_empirical.py

```

Ensure `dataMacro.csv` and related factor files are in the working directory.


## Theoretical Foundation

This implementation satisfies the convergence rate required for valid causal inference in DML. It uses Group Lasso normalized pruning thresholds to manage metric entropy in high-dimensional settings. See the paper for the parameter ranges (e.g., Grid Size *G*) required to ensure consistency.

## Citation

Abdurahman, S. (2026). Kolmogorov-Arnold sieve networks for high-dimensional estimation and inference. *Working Paper.*
