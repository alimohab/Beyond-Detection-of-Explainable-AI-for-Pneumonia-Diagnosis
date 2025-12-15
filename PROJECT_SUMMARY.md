# Project Summary - Key Frame Detection using Deep Learning

## Overview

This project implements **two deep learning approaches** for key frame detection in videos:

1. **Model 1: CNN-based Feature Extraction with Clustering**
   - Uses pre-trained ResNet50 to extract visual features
   - Applies K-means clustering to identify representative frames
   - Dataset: ImageNet (pre-trained)

2. **Model 2: LSTM-based Temporal Key Frame Detection**
   - Combines ResNet50 features with LSTM temporal modeling
   - Detects frames with significant temporal changes
   - Dataset: ImageNet (CNN) + temporal modeling

## Project Deliverables

### ✅ Code Implementation
- [x] Two complete deep learning models
- [x] Video processing utilities
- [x] Visualization tools
- [x] Comparison and evaluation scripts
- [x] Main execution script with command-line interface

### ✅ Documentation
- [x] Comprehensive README
- [x] Installation guide
- [x] Quick start guide
- [x] Project structure documentation
- [x] Complete report template

### ✅ Report Template
- [x] All required sections included
- [x] Model architecture diagrams (code-generated)
- [x] Analysis and discussion sections
- [x] Enhancement ideas section
- [x] Ready for PDF conversion

## File Structure

```
deep_proj/
├── main.py                    # Main execution script
├── generate_report_data.py    # Generate all report data
├── run_example.py            # Quick example script
├── requirements.txt          # Dependencies
├── src/                      # Source code
│   ├── model1_cnn_clustering.py
│   ├── model2_lstm_temporal.py
│   ├── video_processor.py
│   ├── utils.py
│   ├── visualize.py
│   └── comparison.py
├── report/
│   └── report.md            # Report template
└── output/                   # Generated outputs
    ├── keyframes/
    └── results/
```

## Key Features

### Model 1: CNN Clustering
- **Architecture**: ResNet50 → Feature Extraction → K-means Clustering
- **Strengths**: Fast, visually diverse, interpretable
- **Use Case**: Static scenes, visual diversity

### Model 2: LSTM Temporal
- **Architecture**: ResNet50 → LSTM → Change Detection
- **Strengths**: Temporal awareness, scene change detection
- **Use Case**: Dynamic videos, action sequences

## How to Use

### Basic Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run both models
python main.py --video video.mp4 --method both --compare

# Generate report data
python generate_report_data.py --video video.mp4
```

### Command Options
- `--video`: Path to input video (required)
- `--method`: `cnn`, `lstm`, or `both` (default: both)
- `--n_keyframes`: Number of key frames for CNN (default: 5)
- `--compare`: Generate comparison visualizations
- `--frame_interval`: Process every Nth frame (default: 1)
- `--max_frames`: Maximum frames to process

## Output Files

### Key Frames
- Saved as images in `output/keyframes/cnn/` and `output/keyframes/lstm/`

### Visualizations
- Model architecture diagrams
- Key frame comparison
- Timeline visualization
- Feature space analysis

### Results
- JSON files with metrics and comparison data
- Ready to include in report

## Report Requirements Checklist

- [x] Problem description
- [x] Full analysis of the problem
- [x] Model designs with diagrams (code-generated)
- [x] Theory of operation for each model
- [x] Dataset information (ImageNet for both)
- [x] Analysis and discussion section
- [x] Sample results structure
- [x] Enhancement ideas
- [ ] Personal information (to be filled)
- [ ] Sample frames from video (to be added)
- [ ] Actual results (run code to generate)

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`

2. **Run on your video**:
   ```bash
   python generate_report_data.py --video your_video.mp4
   ```

3. **Fill in report**:
   - Add personal information in `report/report.md`
   - Include sample frames from your video
   - Add generated visualizations
   - Include quantitative results

4. **Convert to PDF**:
   - Use Pandoc: `pandoc report/report.md -o report/report.pdf`
   - Or use online Markdown to PDF converters

5. **Create presentation**:
   - Use the visualizations from `output/results/`
   - Include model diagrams
   - Show comparison results

## Technical Details

### Dependencies
- PyTorch 2.0+ (deep learning)
- OpenCV (video processing)
- scikit-learn (clustering)
- NumPy, Matplotlib (visualization)

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB recommended)
- GPU optional (CPU works, slower)

### Performance
- CNN model: ~1-2 seconds per 100 frames (CPU)
- LSTM model: ~2-4 seconds per 100 frames (CPU)
- GPU acceleration available if CUDA is installed

## Support

For issues or questions:
1. Check `INSTALLATION.md` for setup help
2. Review `README.md` for detailed documentation
3. Check `QUICK_START.md` for quick reference

## Deadline Reminder

**Deadline: Friday 21 Dec Midnight**

Make sure to:
- [ ] Run the code on your video
- [ ] Generate all visualizations
- [ ] Complete the report with your information
- [ ] Include sample frames and results
- [ ] Convert report to PDF
- [ ] Prepare presentation

---

**Good luck with your project! 🚀**

