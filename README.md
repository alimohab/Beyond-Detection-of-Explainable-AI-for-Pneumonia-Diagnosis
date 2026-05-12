# Beyond Detection of Explainable AI for Pneumonia Diagnosis

This repository contains multiple notebook pipelines for pneumonia detection from chest X-ray images, plus explainability workflows (Grad-CAM, LIME, SHAP) and a standalone ViT feature extractor.

## Quickstart (2 minutes)

If you just want to run fast:

1. Create environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter tensorflow scikit-learn matplotlib seaborn opencv-python shap lime kagglehub
```

2. Download NIH Chest X-ray dataset and set your local dataset path inside notebooks (replace Kaggle hardcoded `BASE_PATH`).

3. Start Jupyter:

```bash
jupyter notebook
```

4. Run in this order:
- `VIT_features_extractor/vit.py` (optional)
- `convonext-mohamed-khaled.ipynb` or `vit-b16-mohamed-khaled (2).ipynb` or `team03_MobileNetV3_Mohamed_Hatem.ipynb`
- `knn-xai (3).ipynb`
- `team03-xai-icp-mohamed-hatem-model2.ipynb`

## 1. Project overview

Main assets:
- Training and evaluation notebooks for ConvNeXt, ViT, MobileNetV3, ResNet, and KNN-based baselines
- Explainability notebooks (XAI-ICP/Grad-CAM/LIME/SHAP)
- `VIT_features_extractor/vit.py` for extracting ViT embeddings to CSV/PKL

## 2. Requirements

You need:
- Python 3.10+ (3.11 recommended)
- pip
- Jupyter Notebook or JupyterLab

Install packages:

```bash
pip install -r requirements.txt
pip install jupyter tensorflow scikit-learn matplotlib seaborn opencv-python shap lime kagglehub
```

Notes:
- `requirements.txt` contains core packages used by the feature extractor.
- Several notebooks also use extra packages listed above.

## 3. Dataset setup

Most notebooks were originally developed in Kaggle and expect the NIH Chest X-ray dataset.

Dataset source:
- NIH Chest X-rays (Kaggle): https://www.kaggle.com/datasets/nih-chest-xrays/data

Expected structure (local equivalent):

```text
<DATA_ROOT>/
	Data_Entry_2017.csv
	images_001/
	images_002/
	...
```

Important:
- In several notebooks, `BASE_PATH` is hardcoded to:
	- `/kaggle/input/datasets/organizations/nih-chest-xrays/data`
- For local execution, update that variable to your local dataset path in each notebook.

## 4. How to run the full project

### Step A: Create and activate environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter tensorflow scikit-learn matplotlib seaborn opencv-python shap lime kagglehub
```

### Step B: Launch Jupyter

```bash
jupyter notebook
```

Open notebooks from the repository root.

### Step C: Recommended execution order

1. Optional feature extraction:
	 - `VIT_features_extractor/vit.py`
2. Model notebooks (choose one or run multiple for comparison):
	 - `convonext-mohamed-khaled.ipynb`
	 - `vit-b16-mohamed-khaled (2).ipynb`
	 - `team03_MobileNetV3_Mohamed_Hatem.ipynb`
	 - `team03_code_mohmed_hatem.ipynb`
3. Traditional ML + XAI:
	 - `knn-xai (3).ipynb`
4. Dedicated explainability workflow:
	 - `team03-xai-icp-mohamed-hatem-model2.ipynb`

## 5. Run the ViT feature extractor script

Command:

```bash
python VIT_features_extractor/vit.py --image-dir <PATH_TO_IMAGES> --output-dir outputs --batch-size 8
```

Example:

```bash
python VIT_features_extractor/vit.py --image-dir data/images --output-dir outputs --batch-size 8
```

Outputs:
- `outputs/vit_features_xai.pkl`
- `outputs/image_features.csv`

## 6. Notebook execution tips

- Always run cells top-to-bottom in a fresh kernel.
- If a notebook contains `!pip install ...` cells, you can keep them or remove them if already installed in your environment.
- If you see `FileNotFoundError`, first verify `BASE_PATH`, `CSV_PATH`, and image folder paths.
- For GPU acceleration, install CUDA-enabled frameworks and select the correct environment/kernel.

## 7. Troubleshooting

Common issues:
- `ModuleNotFoundError`:
	- Install missing package with `pip install <package_name>`.
- Dataset not found:
	- Confirm `Data_Entry_2017.csv` exists and path variables match your machine.
- Kernel mismatch:
	- In Jupyter, switch to the virtual environment kernel used for installation.

## 8. Reproducibility notes

- Different notebooks may use different preprocessing and train/validation/test splits.
- Keep run logs and output files per notebook for fair model comparison.
- If you need strict reproducibility, fix random seeds inside each notebook and keep package versions pinned.
