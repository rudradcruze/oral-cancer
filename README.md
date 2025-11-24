# Oral Cancer Histopathology Research Pipeline

This repository accompanies the Oral Cancer Histopathology research paper and implements an end-to-end workflow that spans tile preprocessing, transformer-based feature extraction, comparative modeling, feature-selection studies, architectural ablations, and embedding visualization.

## Web Application

A hosted demo of the inference pipeline is available at https://oralcancer.francisrudra.com for quick qualitative testing.

## Authors

-   **Author #1**: Francis Rudra D Cruze — francisrudra@gmail.com
-   **Author #2**: Jeba Wasima — wasima16-620@diu.edu.bd
-   **Author #3**: Md. Faruk Hosen — faruk.cis@diu.edu.bd
-   **Author #4**: Mohammad Badrul Alam Miah — badrul.ict@gmail.com
-   **Author #5**: Zia Muhammad — zia.muhammad@uj.edu
-   **Author #6**: Md Fuyad Al Masud — mdfuyadal.masud@ndsu.edu

-   **Submitting Author**: Md Fuyad Al Masud — mdfuyadal.masud@ndsu.edu
-   **Corresponding Authors**: Mohammad Badrul Alam Miah — badrul.ict@gmail.com, Md Fuyad Al Masud — mdfuyadal.masud@ndsu.edu

## Project Overview

1. **Pre-processing** – Generate 19 enhanced variants of each histopathology tile (denoising, contrast, stain normalization, morphological ops, geometric augmentation, noise injection, etc.) for both `normal` and `oscc` classes.
2. **Feature Extraction** – Use modified ConvNeXt, Swin Transformer, and BEiT backbones augmented with self-attention heads to export 1,024–2,048 dimensional embeddings as CSVs.
3. **Model Benchmarking** – Train bespoke PyTorch heads (Vision Transformer, Attention CNN, Focal-Loss SVM) and compare them with CatBoost and Random Forest baselines on ACC, AUC, SN, SP, F1, MCC, and runtime.
4. **Feature Selection** – Evaluate SHAP, mRMR, Lasso, Boruta, and Genetic Algorithm pipelines, each feeding a Multi-Scale CNN trained on the top 500 features.
5. **Feature Count Sensitivity** – Quantify the trade-off between accuracy and computational budget by varying the number of SHAP-ranked features (200–900).
6. **Ablation Analysis** – Remove attention, multi-scale branches, and global-context components to measure their contribution compared with a simple CNN baseline.
7. **Representation Analysis** – Project embeddings with UMAP and t-SNE to inspect class separability.

## Repository Layout

```
oral-cancer/
├── 1_Pre-Processing.ipynb        # Tile enhancement & visualization suite
├── 2_Feature_Extraction.ipynb    # ConvNeXt/Swin/BEiT embedding exporter
├── 3_Model_Comparison.py         # PyTorch + classical ML benchmarking
├── 4_Feature_Selection.py        # SHAP / mRMR / Lasso / Boruta / GA study
├── 5_Feature_Count.py            # SHAP feature-count sensitivity analysis
├── 6_Ablation_Study.py           # Multi-Scale CNN ablation pipeline
├── 7_UMAP.ipynb                  # UMAP & t-SNE visualizations
└── README.md
```

## Dataset Layout

The code expects tiles under `dataset/original/<class>`:

```
dataset/
└── original/
    ├── normal/
    └── oscc/
```

`1_Pre-Processing.ipynb` writes technique-specific outputs to `dataset/<technique>/<class>` and produces sample mosaics for the manuscript.

## Environment Setup

1. Create a Python ≥3.10 environment and activate it:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2. Install the required libraries (extend for CUDA if needed):

    ```bash
    pip install torch torchvision timm scikit-learn pandas numpy seaborn matplotlib tqdm \
        opencv-python shap mrmr-selection boruta deap catboost umap-learn
    ```

3. Hardware acceleration is auto-detected. Scripts try Apple Metal (MPS) first, then CUDA, and finally fall back to CPU.

## Stage-by-Stage Workflow

### 1. Pre-processing (`1_Pre-Processing.ipynb`)

-   Wraps all augmentation logic inside `HistopathologyPreprocessor`.
-   Supports Gaussian/bilateral filters, NLM denoising, CLAHE, gamma adjustment, color normalization, stain normalization, morphological opening/closing, adaptive thresholding, rotation, and noise injection.
-   Saves a visual report comparing selected techniques for both classes.

### 2. Feature Extraction (`2_Feature_Extraction.ipynb`)

-   Loads tiles, applies torchvision transforms, and extracts embeddings with:
    -   Modified ConvNeXt Large (attention-enhanced).
    -   Modified Swin Transformer Large (self-attention adapters).
    -   BEiT-based encoder for 1,024-D features.
-   Writes CSVs such as `2_Feature_Extraction/BEiT/feature_extract_BEiT_features.csv` (features + `label`, `class`, `filename`).

### 3. Model Benchmarking (`3_Model_Comparison.py`)

-   Performs a 70/15/15 train/validation/test split on BEiT embeddings.
-   PyTorch models:
    -   Vision Transformer head with positional encodings and transformer layers.
    -   Multi-Scale Attention CNN with four convolutional branches, adaptive attention, and deep classifier head.
    -   Focal-Loss SVM-style MLP.
-   Baselines:
    -   CatBoost (`CatBoostClassifier`, `*.cbm` saved).
    -   Random Forest.
-   Saves weights (`*.pt`), histories (`*_history.csv`), ROC figures (`all_models_roc.*`), and summary tables (`all_models_summary.csv`) in `3_Model_Comparison/BEiT/`.

Run:

```bash
python 3_Model_Comparison.py
```

### 4. Feature Selection Study (`4_Feature_Selection.py`)

-   Consumes Swin Transformer embeddings and evaluates five selectors:
    -   SHAP (TreeExplainer + bar/summary plots).
    -   mRMR.
    -   LassoCV.
    -   BorutaPy.
    -   Genetic Algorithm (DEAP).
-   Retrains the Multi-Scale CNN on each 500-feature subset and logs metrics, ROC curves, selection time, and selected feature indices (`*.npy`).
-   Outputs comparison plots, radar charts, overlap heatmaps, and distribution histograms inside `4_Feature_Selection_Comparison/SwinTransformer/`.

Run:

```bash
python 4_Feature_Selection.py
```

### 5. Feature Count Sensitivity (`5_Feature_Count.py`)

-   Reuses SHAP rankings and evaluates subsets of {200, 300, 500, 700, 900} features.
-   Records accuracy, AUC, precision, specificity, sensitivity, F1, MCC, training time, and efficiency (accuracy per feature / per second).
-   Saves metrics (`feature_count_comparison_results.csv`) and high-resolution plots (`feature_count_performance_comparison.png`, `training_time_vs_feature_count.png`, `efficiency_analysis.png`, etc.) in `5_Feature_Count/SwinTransformer/`.

Run:

```bash
python 5_Feature_Count.py
```

### 6. Ablation Study (`6_Ablation_Study.py`)

-   Fixes 500 SHAP-selected features and trains:
    -   `Full_Model` (complete Multi-Scale CNN).
    -   `No_Attention`, `Single_Scale`, `No_Global_Context`.
    -   `Simple_CNN` baseline.
-   Tracks parameter counts, loss/accuracy trajectories, ROC curves, and component contributions (accuracy drops when each module is removed).
-   Persists CSV summaries, parameter tables, and comparison plots into `6_Ablation_Study/SwinTransformer_Ablation/`.

Run:

```bash
python 6_Ablation_Study.py
```

### 7. Representation Analysis (`7_UMAP.ipynb`)

-   Loads BEiT embeddings and generates UMAP and t-SNE projections with publication-ready styling (Times New Roman, bordered axes, high-DPI PNG/PDF output).

## Generated Artifacts

-   **Metrics**: CSV files documenting every experiment (model comparisons, feature selectors, feature counts, ablations).
-   **Models**: PyTorch checkpoints (`*.pt`), CatBoost models (`*.cbm`), and sklearn estimators saved via joblib/pickle where applicable.
-   **Plots**: 1000 DPI PNG/PDF figures for ROC curves, loss curves, radar charts, efficiency plots, SHAP bars, overlap heatmaps, and manifold projections.
-   **Feature Sets**: NumPy arrays storing selected feature indices (`<method>_selected_features.npy`) for reproducibility.

## Troubleshooting & Tips

-   SHAP, Boruta, and GA routines can be compute-heavy. Reduce sample counts or feature targets if memory becomes a constraint.
-   Lower DataLoader batch sizes when running on CPUs or low-memory GPUs.
-   Ensure the relative paths inside each script (e.g., `2_Feature_Extraction/SwinTransformer/...`) match your directory layout.
-   Use `%matplotlib inline` or switch to a non-interactive backend when executing notebooks on headless servers.

## Citation

If you build upon this work, please cite the Oral Cancer Histopathology manuscript and acknowledge the authors above. Direct correspondence to the authors for collaboration opportunities or licensing questions.

## License

Provided for academic research. Contact the authors for commercial or alternative licensing arrangements.
