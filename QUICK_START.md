# Quick Start Guide

## Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run on Your Video
```bash
python main.py --video your_video.mp4 --method both --compare
```

### Step 3: Generate Report Data
```bash
python generate_report_data.py --video your_video.mp4
```

## What You'll Get

After running the code, you'll have:

1. **Key Frames** (`output/keyframes/`)
   - `cnn/`: Key frames from CNN model
   - `lstm/`: Key frames from LSTM model

2. **Visualizations** (`output/results/`)
   - `model1_diagram.png`: CNN model architecture
   - `model2_diagram.png`: LSTM model architecture
   - `keyframes_comparison.png`: Side-by-side comparison
   - `keyframes_timeline.png`: Timeline visualization
   - `feature_analysis.png`: Feature space visualization

3. **Results** (`output/results/`)
   - `results_summary.json`: Complete results summary
   - `comparison_metrics.json`: Comparison metrics

## Example Commands

### Run both models with comparison
```bash
python main.py --video video.mp4 --method both --compare
```

### Run only CNN model
```bash
python main.py --video video.mp4 --method cnn --n_keyframes 5
```

### Run only LSTM model
```bash
python main.py --video video.mp4 --method lstm
```

### Process fewer frames (faster)
```bash
python main.py --video video.mp4 --method both --frame_interval 5 --max_frames 100
```

## For the Report

1. **Fill in personal information** in `report/report.md`
2. **Add sample frames** from your video
3. **Include generated visualizations** from `output/results/`
4. **Add quantitative results** from `output/results/results_summary.json`
5. **Convert to PDF** using Pandoc or Markdown PDF

## Troubleshooting

**Video too large?**
- Use `--frame_interval` to skip frames
- Use `--max_frames` to limit processing

**Out of memory?**
- Reduce `--n_keyframes`
- Increase `--frame_interval`
- Process shorter video segments

**Need help?**
- Check `INSTALLATION.md` for detailed setup
- Review `README.md` for full documentation

