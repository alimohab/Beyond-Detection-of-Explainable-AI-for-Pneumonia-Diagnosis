# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- CUDA-capable GPU (optional, for faster processing)

## Step-by-Step Installation

### 1. Clone or Download the Project

If you have the project files, navigate to the project directory:
```bash
cd deep_proj
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and TorchVision (for deep learning models)
- OpenCV (for video processing)
- scikit-learn (for clustering and utilities)
- NumPy, Matplotlib, and other dependencies

### 4. Verify Installation

Test if PyTorch is installed correctly:
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 5. Prepare Your Video

Place your 30-second video file in the project root or in `data/videos/` directory.

## Troubleshooting

### Issue: CUDA not available
- **Solution**: The code will automatically use CPU if CUDA is not available. Processing will be slower but still functional.

### Issue: OpenCV installation fails
- **Solution**: Try installing with:
  ```bash
  pip install opencv-python-headless
  ```

### Issue: PyTorch installation fails
- **Solution**: Visit https://pytorch.org/ and install PyTorch according to your system configuration.

### Issue: Out of memory errors
- **Solution**: 
  - Reduce `--max_frames` parameter
  - Increase `--frame_interval` to process fewer frames
  - Use a smaller video or resize it

## Quick Start

After installation, run:
```bash
python main.py --video your_video.mp4 --method both --compare
```

Or use the example script:
```bash
python run_example.py
```

## System Requirements

**Minimum:**
- RAM: 4GB
- Storage: 2GB free space
- CPU: Any modern processor

**Recommended:**
- RAM: 8GB or more
- GPU: NVIDIA GPU with CUDA support
- Storage: 5GB free space

