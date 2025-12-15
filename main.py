"""
Main script for key frame detection using deep learning

Usage:
    python main.py --video path/to/video.mp4 --method both
    python main.py --video path/to/video.mp4 --method cnn
    python main.py --video path/to/video.mp4 --method lstm --compare
"""
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from video_processor import VideoProcessor
from model1_cnn_clustering import CNNKeyFrameDetector
from model2_lstm_temporal import LSTMKeyFrameDetector
from utils import save_keyframes
from visualize import (plot_keyframes_comparison, plot_keyframe_timeline,
                      create_model_diagram, plot_feature_analysis)
from comparison import compare_models, print_comparison_report


def main():
    parser = argparse.ArgumentParser(description='Key Frame Detection using Deep Learning')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--method', type=str, default='both', 
                       choices=['cnn', 'lstm', 'both'],
                       help='Detection method to use')
    parser.add_argument('--n_keyframes', type=int, default=5,
                       help='Number of key frames to detect (for CNN model)')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison visualizations')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='Process every Nth frame')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Initialize video processor
    print("=" * 60)
    print("Key Frame Detection using Deep Learning")
    print("=" * 60)
    
    processor = VideoProcessor(args.video)
    video_info = processor.get_info()
    
    print(f"\nVideo Information:")
    print(f"  Duration: {video_info['duration']:.2f} seconds")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  Total Frames: {video_info['frame_count']}")
    
    # Load frames
    frames = processor.load_frames(max_frames=args.max_frames, 
                                   frame_interval=args.frame_interval)
    
    # Get video name for output
    video_name = Path(args.video).stem
    
    # Create output directories
    output_dir = Path('output/keyframes')
    results_dir = Path('output/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    keyframes_cnn = None
    keyframes_lstm = None
    
    # Run CNN-based detection
    if args.method in ['cnn', 'both']:
        print("\n" + "=" * 60)
        print("Running Model 1: CNN-based Key Frame Detection")
        print("=" * 60)
        
        cnn_detector = CNNKeyFrameDetector(n_clusters=args.n_keyframes)
        model_info = cnn_detector.get_model_info()
        
        print(f"\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        keyframes_cnn = cnn_detector.detect_keyframes(frames)
        print(f"\nDetected {len(keyframes_cnn)} key frames: {keyframes_cnn}")
        
        # Save keyframes
        cnn_output_dir = output_dir / 'cnn'
        save_keyframes(keyframes_cnn, frames, str(cnn_output_dir), 
                      f"{video_name}_cnn")
        
        # Create model diagram
        create_model_diagram('cnn', str(results_dir / 'model1_diagram.png'))
    
    # Run LSTM-based detection
    if args.method in ['lstm', 'both']:
        print("\n" + "=" * 60)
        print("Running Model 2: LSTM-based Temporal Key Frame Detection")
        print("=" * 60)
        
        lstm_detector = LSTMKeyFrameDetector()
        model_info = lstm_detector.get_model_info()
        
        print(f"\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        keyframes_lstm = lstm_detector.detect_keyframes(frames)
        print(f"\nDetected {len(keyframes_lstm)} key frames: {keyframes_lstm}")
        
        # Save keyframes
        lstm_output_dir = output_dir / 'lstm'
        save_keyframes(keyframes_lstm, frames, str(lstm_output_dir), 
                      f"{video_name}_lstm")
        
        # Create model diagram
        create_model_diagram('lstm', str(results_dir / 'model2_diagram.png'))
    
    # Generate comparison visualizations and analysis
    if args.compare and keyframes_cnn is not None and keyframes_lstm is not None:
        print("\n" + "=" * 60)
        print("Generating Comparison Visualizations")
        print("=" * 60)
        
        # Comparison plot
        plot_keyframes_comparison(
            frames, keyframes_cnn, keyframes_lstm,
            model1_name="Model 1 (CNN Clustering)",
            model2_name="Model 2 (LSTM Temporal)",
            save_path=str(results_dir / 'keyframes_comparison.png')
        )
        
        # Timeline plot
        plot_keyframe_timeline(
            frames, keyframes_cnn, keyframes_lstm,
            model1_name="Model 1 (CNN)",
            model2_name="Model 2 (LSTM)",
            save_path=str(results_dir / 'keyframes_timeline.png')
        )
        
        # Feature analysis (if CNN model was used)
        if args.method in ['cnn', 'both']:
            features = cnn_detector.extract_features(frames)
            plot_feature_analysis(
                features, keyframes_cnn,
                save_path=str(results_dir / 'feature_analysis.png')
            )
        
        # Generate comparison report
        comparison = compare_models(
            keyframes_cnn, keyframes_lstm, frames,
            model1_name="Model 1 (CNN Clustering)",
            model2_name="Model 2 (LSTM Temporal)"
        )
        print_comparison_report(comparison)
        
        # Save comparison metrics
        import json
        with open(results_dir / 'comparison_metrics.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  Key frames: {output_dir}")
    print(f"  Visualizations: {results_dir}")


if __name__ == '__main__':
    main()

