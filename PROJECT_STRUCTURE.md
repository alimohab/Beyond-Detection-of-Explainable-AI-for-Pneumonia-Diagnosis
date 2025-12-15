# Project Structure

```
deep_proj/
│
├── main.py                          # Main execution script
├── run_example.py                   # Example script to run on provided video
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── PROJECT_STRUCTURE.md             # This file
├── .gitignore                       # Git ignore file
│
├── src/                             # Source code directory
│   ├── __init__.py
│   ├── model1_cnn_clustering.py    # Model 1: CNN-based key frame detection
│   ├── model2_lstm_temporal.py     # Model 2: LSTM-based key frame detection
│   ├── video_processor.py          # Video processing utilities
│   ├── utils.py                    # Helper functions
│   ├── visualize.py                # Visualization tools
│   └── comparison.py               # Model comparison utilities
│
├── data/                            # Data directory
│   └── videos/                      # Input videos (place your videos here)
│
├── output/                          # Output directory
│   ├── keyframes/                   # Detected key frames
│   │   ├── cnn/                     # Key frames from CNN model
│   │   └── lstm/                     # Key frames from LSTM model
│   ├── models/                      # Saved models (if any)
│   └── results/                     # Comparison results and visualizations
│       ├── keyframes_comparison.png
│       ├── keyframes_timeline.png
│       ├── feature_analysis.png
│       ├── model1_diagram.png
│       ├── model2_diagram.png
│       └── comparison_metrics.json
│
└── report/                          # Report directory
    └── report.md                    # Project report template (Markdown)
```

## Key Files Description

### Main Scripts
- **main.py**: Main entry point for key frame detection
- **run_example.py**: Example script to quickly run on a video

### Model Implementations
- **model1_cnn_clustering.py**: 
  - Uses ResNet50 for feature extraction
  - Applies K-means clustering
  - Dataset: ImageNet (pre-trained)

- **model2_lstm_temporal.py**:
  - Uses ResNet50 + LSTM for temporal modeling
  - Detects frames with significant changes
  - Dataset: ImageNet (CNN) + temporal modeling

### Utilities
- **video_processor.py**: Video loading and frame extraction
- **utils.py**: General utility functions
- **visualize.py**: Visualization and plotting functions
- **comparison.py**: Model comparison and evaluation

### Output Structure
After running the code, you'll find:
- Key frames saved as images in `output/keyframes/`
- Comparison visualizations in `output/results/`
- Model diagrams in `output/results/`

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run on a video:**
   ```bash
   python main.py --video path/to/video.mp4 --method both --compare
   ```

3. **Quick example:**
   ```bash
   python run_example.py
   ```

## Report Generation

The report template is in `report/report.md`. After running the code:
1. Add your personal information
2. Include sample frames from your video
3. Add the generated visualizations
4. Include quantitative results
5. Convert to PDF using a Markdown to PDF converter (e.g., Pandoc, Markdown PDF)

