# Key Frame Detection using Deep Learning

## Project Overview
This project implements two deep learning approaches for key frame detection in videos:
1. **CNN-based Feature Extraction with Clustering** - Uses pre-trained CNN models to extract visual features and cluster frames
2. **LSTM-based Temporal Modeling** - Uses LSTM networks to model temporal dependencies and detect key frames

## Project Structure
```
deep_proj/
├── src/
│   ├── model1_cnn_clustering.py    # CNN-based key frame detection
│   ├── model2_lstm_temporal.py     # LSTM-based key frame detection
│   ├── video_processor.py          # Video processing utilities
│   ├── utils.py                    # Helper functions
│   └── visualize.py                # Visualization tools
├── data/
│   └── videos/                     # Input videos directory
├── output/
│   ├── keyframes/                  # Detected key frames
│   ├── models/                     # Saved models
│   └── results/                    # Comparison results
├── report/
│   └── report.md                   # Project report template
├── main.py                         # Main execution script
└── requirements.txt                # Python dependencies
```

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py --video path/to/video.mp4 --method both
```

### Using Specific Model
```bash
# CNN-based approach
python main.py --video path/to/video.mp4 --method cnn

# LSTM-based approach
python main.py --video path/to/video.mp4 --method lstm
```

### Compare Both Models
```bash
python main.py --video path/to/video.mp4 --method both --compare
```

## Models Description

### Model 1: CNN-based Feature Extraction with Clustering
- Uses pre-trained ResNet50 to extract features from each frame
- Applies K-means clustering on feature vectors
- Selects frames closest to cluster centers as key frames
- **Dataset**: Pre-trained on ImageNet

### Model 2: LSTM-based Temporal Modeling
- Extracts CNN features from frames
- Uses LSTM to model temporal sequences
- Detects key frames based on temporal changes
- **Dataset**: Can be trained on video datasets like UCF-101, Kinetics

## Output
- Detected key frames saved as images
- Comparison visualizations
- Performance metrics and analysis

## Authors
[Add your names, IDs, and supervisor information]

## Deadline
Friday 21 Dec Midnight

