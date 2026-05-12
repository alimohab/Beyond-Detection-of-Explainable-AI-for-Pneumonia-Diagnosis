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
- `VIT_features_extractor/vit.py` (optional feature export)
- Start one deep model notebook (ViT/CNN family)
- Run KNN-XAI notebook
- Run final dedicated XAI notebook

5. For full details on all notebooks, follow sections 5 and 6 below.

## 1. Project overview

Main assets:
- Training and evaluation notebooks for ConvNeXt, ViT, MobileNetV3, ResNet, and KNN-based baselines
- Explainability notebooks (XAI-ICP/Grad-CAM/LIME/SHAP)
- `VIT_features_extractor/vit.py` for extracting ViT embeddings to CSV/PKL

## 2. Notebook-by-notebook guide

This repository contains multiple model variants and some iterative notebook versions. Use this map to understand each file.

- `convonext-mohamed-khaled.ipynb`
	- ConvNeXt-style deep learning workflow with EDA and explainability components.
- `knn-xai (3).ipynb`
	- KNN-based classification experiments with XAI analysis (LIME/SHAP style interpretation).
- `mohamed-khaled-vit (1).ipynb`
	- Vision Transformer training/evaluation workflow for pneumonia detection.
- `team03-ali-resnet152v2-model (2).ipynb`
	- ResNet152V2 model training and evaluation pipeline.
- `team03-code-abdelrhman-hisham (1).ipynb`
	- General training/experimentation notebook (Abdelrhman/Hisham version).
- `team03-code-ali (2).ipynb`
	- Ali experimentation/training workflow (older iteration).
- `team03-code-ali (4).ipynb`
	- Ali experimentation/training workflow (newer iteration, prefer this over `(2)`).
- `team03-efficientb0-ali-mohab.ipynb`
	- EfficientNet-B0 model training/evaluation.
- `team03-xai-icp-mohamed-hatem-model2.ipynb`
	- Dedicated explainability notebook for model interpretation (XAI-ICP-focused).
- `team03-xception-model-ali-mohab (3).ipynb`
	- Xception model training/validation notebook.
- `team03_code_abdelrhman.ipynb`
	- Abdelrhman training notebook variant; overlaps with `team03-code-abdelrhman-hisham (1).ipynb`.
- `team03_code_mohmed_hatem.ipynb`
	- Mohamed Hatem training/experimentation notebook.
- `team03_MobileNetV3_Mohamed_Hatem.ipynb`
	- MobileNetV3 model training/evaluation pipeline.
- `vit-b16-mohamed-khaled (2).ipynb`
	- ViT-B16 pipeline (earlier version).
- `vit-b16-mohamed-khaled (3) (1).ipynb`
	- ViT-B16 pipeline (later revision, generally prefer this over `(2)`).

Suggested rule for versioned files:
- If two notebooks look similar and only differ by suffix like `(2)`, `(3)`, `(4)`, start with the higher-numbered one.

## 3. Requirements

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

## 4. Dataset setup

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

## 5. How to run the full project

This section is the full project startup path (from zero to complete run).

### What "full project" means in this repository

Running the full project usually means:
- Setting up one consistent Python environment
- Pointing all notebooks to the same dataset path
- Running representative notebooks from each model family
- Running traditional ML + XAI notebooks
- Optionally exporting ViT features with the script pipeline

You do not need to run every duplicate notebook version. For similar files, prefer higher-numbered revisions.

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

### Step B.1: Before running any notebook (important)

Do this once before each notebook family run:
- Restart kernel and clear output (to avoid variable leakage from previous notebooks)
- Verify the active kernel is your project environment
- Update path cells (`BASE_PATH`, CSV/image paths) before running long training cells
- Run import/setup cells first and confirm no missing modules

### Step C: Recommended execution order

1. Optional feature extraction:
	 - `VIT_features_extractor/vit.py`
2. Pick deep-learning model notebooks to benchmark:
	 - ViT family:
		 - `vit-b16-mohamed-khaled (3) (1).ipynb` (preferred)
		 - `vit-b16-mohamed-khaled (2).ipynb`
		 - `mohamed-khaled-vit (1).ipynb`
	 - CNN family:
		 - `convonext-mohamed-khaled.ipynb`
		 - `team03_MobileNetV3_Mohamed_Hatem.ipynb`
		 - `team03-ali-resnet152v2-model (2).ipynb`
		 - `team03-efficientb0-ali-mohab.ipynb`
		 - `team03-xception-model-ali-mohab (3).ipynb`
	 - General experiment notebooks:
		 - `team03_code_mohmed_hatem.ipynb`
		 - `team03-code-abdelrhman-hisham (1).ipynb` or `team03_code_abdelrhman.ipynb`
		 - `team03-code-ali (4).ipynb` (preferred) or `team03-code-ali (2).ipynb`
3. Run traditional ML + local explainability:
	 - `knn-xai (3).ipynb`
4. Run dedicated explainability summary notebook:
	 - `team03-xai-icp-mohamed-hatem-model2.ipynb`

### Step D: How to start all notebooks (complete checklist)

Use this if you want broad coverage of the whole repository.

#### Group 1: ViT notebooks
- `vit-b16-mohamed-khaled (3) (1).ipynb` (start here)
- `vit-b16-mohamed-khaled (2).ipynb`
- `mohamed-khaled-vit (1).ipynb`

Run approach:
- Run environment/import cells
- Fix dataset path variables
- Run preprocessing and dataloader cells
- Run training/evaluation cells
- Save metrics/plots before moving to next notebook

#### Group 2: CNN architecture notebooks
- `convonext-mohamed-khaled.ipynb`
- `team03_MobileNetV3_Mohamed_Hatem.ipynb`
- `team03-ali-resnet152v2-model (2).ipynb`
- `team03-efficientb0-ali-mohab.ipynb`
- `team03-xception-model-ali-mohab (3).ipynb`

Run approach:
- Use same dataset split strategy where possible for fair comparison
- Record accuracy, precision, recall, F1, ROC-AUC per model
- Keep output folder names unique by notebook/model

#### Group 3: General experiment notebooks
- `team03_code_mohmed_hatem.ipynb`
- `team03-code-abdelrhman-hisham (1).ipynb`
- `team03_code_abdelrhman.ipynb`
- `team03-code-ali (4).ipynb` (preferred)
- `team03-code-ali (2).ipynb`

Run approach:
- Treat these as alternative/iterative experiment variants
- If two notebooks overlap heavily, run only one preferred version first

#### Group 4: Traditional ML + explainability notebooks
- `knn-xai (3).ipynb`
- `team03-xai-icp-mohamed-hatem-model2.ipynb`

Run approach:
- Run after at least one deep model notebook to align comparisons
- Export and keep explanation plots (LIME/SHAP/Grad-CAM/PDP/ICE) with clear names

### Step E: Suggested practical workflow (recommended)

If you need complete but efficient execution:
1. Run one preferred ViT notebook
2. Run two or three CNN notebooks (MobileNetV3 + ResNet/Xception + ConvNeXt/EfficientNet)
3. Run one general experiment notebook per author stream
4. Run KNN-XAI notebook
5. Run XAI-ICP notebook
6. Optionally run remaining duplicate versions only if needed

This gives full project coverage without wasting time on near-identical iterations.

## 6. Run the ViT feature extractor script

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

## 7. Notebook execution tips

- Always run cells top-to-bottom in a fresh kernel.
- If a notebook contains `!pip install ...` cells, you can keep them or remove them if already installed in your environment.
- If you see `FileNotFoundError`, first verify `BASE_PATH`, `CSV_PATH`, and image folder paths.
- For GPU acceleration, install CUDA-enabled frameworks and select the correct environment/kernel.
- After each notebook completes, immediately save:
	- model metrics
	- confusion matrix / ROC / PR plots
	- explainability outputs (if available)
	- model checkpoint path

## 8. Troubleshooting

Common issues:
- `ModuleNotFoundError`:
	- Install missing package with `pip install <package_name>`.
- Dataset not found:
	- Confirm `Data_Entry_2017.csv` exists and path variables match your machine.
- Kernel mismatch:
	- In Jupyter, switch to the virtual environment kernel used for installation.

## 9. Reproducibility notes

- Different notebooks may use different preprocessing and train/validation/test splits.
- Keep run logs and output files per notebook for fair model comparison.
- If you need strict reproducibility, fix random seeds inside each notebook and keep package versions pinned.

## 10. Additional project files

- `Team03_proposal (1).pdf`
	- Project proposal document.
- `Team03_phase1.pdf`
	- Phase/report document.
