"""
Script to generate data and visualizations for the report
This script runs both models and generates all necessary outputs for the report
"""
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from video_processor import VideoProcessor
from model1_cnn_clustering import CNNKeyFrameDetector
from model2_lstm_temporal import LSTMKeyFrameDetector
from utils import save_keyframes, get_video_info
from visualize import (plot_keyframes_comparison, plot_keyframe_timeline,
                      create_model_diagram, plot_feature_analysis)
from comparison import compare_models, print_comparison_report


def generate_report_data(video_path, n_keyframes=5, frame_interval=2):
    """
    Generate all data and visualizations needed for the report
    
    Args:
        video_path: Path to input video
        n_keyframes: Number of key frames for CNN model
        frame_interval: Process every Nth frame
    """
    print("=" * 70)
    print("GENERATING REPORT DATA AND VISUALIZATIONS")
    print("=" * 70)
    
    # Initialize video processor
    processor = VideoProcessor(video_path)
    video_info = processor.get_info()
    
    print(f"\nVideo Information:")
    print(f"  File: {video_path}")
    print(f"  Duration: {video_info['duration']:.2f} seconds")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  Total Frames: {video_info['frame_count']}")
    
    # Load frames
    frames = processor.load_frames(frame_interval=frame_interval)
    print(f"  Processed Frames: {len(frames)}")
    
    # Get video name
    video_name = Path(video_path).stem
    
    # Create output directories
    output_dir = Path('output/keyframes')
    results_dir = Path('output/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Model 1: CNN-based
    print("\n" + "=" * 70)
    print("MODEL 1: CNN-based Key Frame Detection")
    print("=" * 70)
    
    cnn_detector = CNNKeyFrameDetector(n_clusters=n_keyframes)
    model1_info = cnn_detector.get_model_info()
    
    print("\nModel Information:")
    for key, value in model1_info.items():
        print(f"  {key}: {value}")
    
    keyframes_cnn = cnn_detector.detect_keyframes(frames)
    print(f"\nDetected {len(keyframes_cnn)} key frames")
    print(f"Key frame indices: {keyframes_cnn}")
    
    # Save keyframes
    cnn_output_dir = output_dir / 'cnn'
    save_keyframes(keyframes_cnn, frames, str(cnn_output_dir), f"{video_name}_cnn")
    
    # Extract features for visualization
    features = cnn_detector.extract_features(frames)
    
    # Run Model 2: LSTM-based
    print("\n" + "=" * 70)
    print("MODEL 2: LSTM-based Temporal Key Frame Detection")
    print("=" * 70)
    
    lstm_detector = LSTMKeyFrameDetector()
    model2_info = lstm_detector.get_model_info()
    
    print("\nModel Information:")
    for key, value in model2_info.items():
        print(f"  {key}: {value}")
    
    keyframes_lstm = lstm_detector.detect_keyframes(frames)
    print(f"\nDetected {len(keyframes_lstm)} key frames")
    print(f"Key frame indices: {keyframes_lstm}")
    
    # Save keyframes
    lstm_output_dir = output_dir / 'lstm'
    save_keyframes(keyframes_lstm, frames, str(lstm_output_dir), f"{video_name}_lstm")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Model diagrams
    print("\n1. Creating model architecture diagrams...")
    create_model_diagram('cnn', str(results_dir / 'model1_diagram.png'))
    create_model_diagram('lstm', str(results_dir / 'model2_diagram.png'))
    
    # Comparison visualizations
    print("\n2. Creating comparison visualizations...")
    plot_keyframes_comparison(
        frames, keyframes_cnn, keyframes_lstm,
        model1_name="Model 1 (CNN Clustering)",
        model2_name="Model 2 (LSTM Temporal)",
        save_path=str(results_dir / 'keyframes_comparison.png')
    )
    
    plot_keyframe_timeline(
        frames, keyframes_cnn, keyframes_lstm,
        model1_name="Model 1 (CNN)",
        model2_name="Model 2 (LSTM)",
        save_path=str(results_dir / 'keyframes_timeline.png')
    )
    
    # Feature analysis
    print("\n3. Creating feature space visualization...")
    plot_feature_analysis(
        features, keyframes_cnn,
        save_path=str(results_dir / 'feature_analysis.png')
    )
    
    # Comparison metrics
    print("\n4. Computing comparison metrics...")
    comparison = compare_models(
        keyframes_cnn, keyframes_lstm, frames,
        model1_name="Model 1 (CNN Clustering)",
        model2_name="Model 2 (LSTM Temporal)"
    )
    print_comparison_report(comparison)
    
    # Save all results
    results_summary = {
        'video_info': {
            'path': str(video_path),
            'duration': video_info['duration'],
            'fps': video_info['fps'],
            'resolution': f"{video_info['width']}x{video_info['height']}",
            'total_frames': video_info['frame_count'],
            'processed_frames': len(frames)
        },
        'model1': {
            'info': model1_info,
            'keyframe_indices': keyframes_cnn,
            'n_keyframes': len(keyframes_cnn)
        },
        'model2': {
            'info': model2_info,
            'keyframe_indices': keyframes_lstm,
            'n_keyframes': len(keyframes_lstm)
        },
        'comparison': comparison
    }
    
    with open(results_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("REPORT DATA GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  Key frames: {output_dir}")
    print(f"  Visualizations: {results_dir}")
    print(f"  Summary: {results_dir / 'results_summary.json'}")
    print(f"\nNext steps:")
    print(f"  1. Review the generated visualizations in output/results/")
    print(f"  2. Add sample frames to the report")
    print(f"  3. Include the visualizations in the report")
    print(f"  4. Fill in the report template with your personal information")
    print(f"  5. Convert report.md to PDF")
    
    return results_summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate report data and visualizations')
    parser.add_argument('--video', type=str, default='3032568-uhd_3840_2160_25fps.mp4',
                       help='Path to input video')
    parser.add_argument('--n_keyframes', type=int, default=5,
                       help='Number of key frames for CNN model')
    parser.add_argument('--frame_interval', type=int, default=2,
                       help='Process every Nth frame')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        print("Please provide a valid video path using --video argument")
        sys.exit(1)
    
    generate_report_data(args.video, args.n_keyframes, args.frame_interval)

