# WIMOAD: Stacking Ensemble and Weighted Multi-Omics Integration for Alzheimer's Disease Diagnosis
**WIMOAD** a stacking ensemble and weighted integration of multi-omics data for AD diagnosis. WIMOAD synergistically leverages specialized classifiers for patients' paired gene expression and methylation data for multi-stage classification. The resulting scores of classifiers were then stacked for meta-learning performance improvement. The prediction results of two distinct meta-models were integrated with optimized weights for the final decision-making of the model, providing higher performance than using single omics only.

## Table of Contents (Under construction)
- [Overview](#overview)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
- [Using WIMOAD on Your Own Data](#using-wimoad-on-your-own-data)
- [Contact](#contact)
- [Publication](#publication)

## Overview
WIMOAD is organized around two stages, each usable independently:
- **Meta-learning (stacking)** — `runner.py`, `main.py`, `parallel.py`/`parallel_loo.py`, `model_config.py`, `data_loader.py`, `metrics.py`: for each omics branch, trains base classifiers, stacks their out-of-fold predictions with several meta-models under nested cross-validation, and reports evaluation metrics.
- **Integration (weighted fusion)** — `integration.py`: given each branch's best stacking model, searches the expression/methylation weight that maximizes the fused prediction's AUC.

This repository ships the modeling framework only; it does not include or require any specific dataset.

## Installation
1. Clone the WIMOAD git repository:
```bash
git clone https://github.com/wan-mlab/WIMOAD.git
cd WIMOAD
```
2. Create a new conda environment:
```bash
conda create -n wimoad python=3.9
conda activate wimoad
```
3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Data Format
WIMOAD expects one CSV per omics branch, each with:
- `RID`: a unique sample identifier (used as the index)
- `DX_bl`: the raw diagnosis label for that sample
- remaining columns: feature values for that branch (gene expression or methylation levels)

Raw `DX_bl` values are remapped to a binary label per task using the `label_map` defined for that task in `configs/tasks.yaml`; any samples with a value not covered by the mapping's binary classes are dropped. No other assumptions are made about the data — feature columns, sample counts, and diagnosis coding are all task/dataset-specific and configured in `configs/tasks.yaml`.

## Usage
Run the default `ca` task for both omics branches with leave-one-out CV:
```bash
python runner.py
```

Run a specific branch, CV strategy, and output directory:
```bash
python runner.py --group ca --omics expression --cv KFold --output-dir results
```

Core files:
- `configs/tasks.yaml`: task groups, label maps, and base-model hyperparameters
- `data_loader.py`: CSV loading, label remapping, and feature selection
- `model_config.py`: meta models and sklearn estimator factories built from `configs/tasks.yaml`
- `parallel.py` / `parallel_loo.py`: nested-CV stacking training (KFold and leave-one-out outer loops)
- `metrics.py`: classification metrics computed per stacking run
- `runner.py`: command-line stacking workflow
- `integration.py`: weighted multi-omics fusion — best-model selection per branch and fusion weight search

## Using WIMOAD on Your Own Data
1. Prepare one CSV per omics branch in the format described in [Data Format](#data-format).
2. Add a task entry to `configs/tasks.yaml`: point `expression_file`/`methylation_file` at your CSVs, define the `label_map` for your diagnosis coding, and set base-classifier hyperparameters for each branch.
3. Run `python runner.py --group <your_task>` to train and evaluate the stacking ensembles for that task.
4. Use `integration.py` to select each branch's best model and search the fusion weight over the resulting predictions.

## Contact
If you have any questions, comments, or would like to report a bug, please contact haxiao@unmc.edu

## Publication
Xiao, H.; Wang, J.; Wan, S. WIMOAD: Weighted Integration of Multi-Omics data for Alzheimer's Disease (AD) Diagnosis. bioRxiv 2024.09.25.614862, https://www.biorxiv.org/content/10.1101/2024.09.25.614862v1
