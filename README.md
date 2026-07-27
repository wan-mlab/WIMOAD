# WIMOAD: Stacking Ensemble and Weighted Multi-Omics Integration for Alzheimer’s Disease Diagnosis
**WIMOAD** a stacking ensemble and weighted integration of multi-omics data for AD diagnosis. WIMOAD synergistically leverages specialized classifiers for patients' paired gene expression and methylation data for multi-stage classification. The resulting scores of classifiers were then stacked for meta-learning performance improvement. The prediction results of two distinct meta-models were integrated with optimized weights for the final decision-making of the model, providing higher performance than using single omics only. In addition, WIMOAD also stands out as a biologically interpretable model by leveraging the SHapley Additive exPlanations (SHAP) to elucidate the contributions of each gene from each omics to the model output. 

## Table of Contents (Under construction)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
- [Publication](#publication)

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

## Data
The training scripts expect user-provided ADNI-derived feature matrices. ADNI
data are access-controlled and are not redistributed by this repository.

Input CSV files must include:
- `RID`: participant identifier
- `DX_bl`: baseline diagnosis label
- feature columns for the selected omics branch

Expression files may also include `DX_bl_nodia`, `DX_bl_dia`, and `RID_nodia`;
these helper columns are ignored by the loader. Task label mappings and model
hyperparameters are defined in `configs/tasks.yaml`.

## Usage
Run the default CA task for both omics branches with leave-one-out CV:
```bash
python runner.py
```

Run a specific branch, CV strategy, and output directory:
```bash
python runner.py --group ca --omics expression --cv KFold --output-dir results
```

Generate R3 supplemental McNemar tables from existing result CSV files:
```bash
python scripts/make_supplemental_tables.py --help
```

Core files:
- `configs/tasks.yaml`: task groups, label maps, and base-model hyperparameters
- `model_config.py`: meta models and sklearn estimator factories
- `runner.py`: command-line stacking workflow
- `wimoad/integration.py`: deterministic weighted omics fusion
- `wimoad/statistics.py`: exact McNemar utilities
- `scripts/make_supplemental_tables.py`: supplemental table generation

Run tests after installing development dependencies:
```bash
pytest -q
```

## Contact
If you have any questions, comments, or would like to report a bug, please contact haxiao@unmc.edu

## Publication
Xiao, H.; Wang, J.; Wan, S. WIMOAD: Weighted Integration of Multi-Omics data for Alzheimer’s Disease (AD) Diagnosis. bioRxiv 2024.09.25.614862, https://www.biorxiv.org/content/10.1101/2024.09.25.614862v1
