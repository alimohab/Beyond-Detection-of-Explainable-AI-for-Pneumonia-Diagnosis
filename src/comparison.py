"""
Comparison and evaluation utilities for key frame detection models
"""
import numpy as np
from sklearn.metrics import silhouette_score
from utils import calculate_frame_differences


def compare_models(keyframes_model1, keyframes_model2, frames, 
                  model1_name="Model 1 (CNN)", 
                  model2_name="Model 2 (LSTM)"):
    """
    Compare two key frame detection models
    
    Args:
        keyframes_model1: Key frame indices from model 1
        keyframes_model2: Key frame indices from model 2
        frames: List of all frames
        model1_name: Name of model 1
        model2_name: Name of model 2
    
    Returns:
        Dictionary with comparison metrics
    """
    n_frames = len(frames)
    
    # Calculate overlap
    set1 = set(keyframes_model1)
    set2 = set(keyframes_model2)
    overlap = len(set1.intersection(set2))
    overlap_ratio = overlap / max(len(set1), len(set2)) if max(len(set1), len(set2)) > 0 else 0
    
    # Calculate coverage (how well keyframes are distributed)
    def calculate_coverage(keyframes, n_frames):
        if len(keyframes) == 0:
            return 0
        intervals = []
        for i in range(len(keyframes) - 1):
            intervals.append(keyframes[i+1] - keyframes[i])
        if len(intervals) == 0:
            return 1.0
        # Lower variance = better coverage
        coverage_score = 1.0 / (1.0 + np.std(intervals) / np.mean(intervals))
        return coverage_score
    
    coverage1 = calculate_coverage(keyframes_model1, n_frames)
    coverage2 = calculate_coverage(keyframes_model2, n_frames)
    
    # Calculate temporal distribution
    def temporal_distribution(keyframes, n_frames):
        if len(keyframes) == 0:
            return 0
        # Normalize to [0, 1]
        normalized = np.array(keyframes) / n_frames
        # Calculate spread
        spread = np.max(normalized) - np.min(normalized)
        return spread
    
    spread1 = temporal_distribution(keyframes_model1, n_frames)
    spread2 = temporal_distribution(keyframes_model2, n_frames)
    
    # Calculate frame difference scores at keyframes
    frame_diffs = calculate_frame_differences(frames)
    frame_diffs = np.concatenate([[0], frame_diffs])  # Add first frame
    
    def avg_change_at_keyframes(keyframes, frame_diffs):
        if len(keyframes) == 0:
            return 0
        changes = [frame_diffs[idx] for idx in keyframes if idx < len(frame_diffs)]
        return np.mean(changes) if len(changes) > 0 else 0
    
    avg_change1 = avg_change_at_keyframes(keyframes_model1, frame_diffs)
    avg_change2 = avg_change_at_keyframes(keyframes_model2, frame_diffs)
    
    comparison = {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'model1_keyframes': len(keyframes_model1),
        'model2_keyframes': len(keyframes_model2),
        'overlap_count': overlap,
        'overlap_ratio': overlap_ratio,
        'coverage_model1': coverage1,
        'coverage_model2': coverage2,
        'temporal_spread_model1': spread1,
        'temporal_spread_model2': spread2,
        'avg_change_model1': avg_change1,
        'avg_change_model2': avg_change2
    }
    
    return comparison


def print_comparison_report(comparison):
    """
    Print a formatted comparison report
    
    Args:
        comparison: Dictionary from compare_models()
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON REPORT")
    print("=" * 60)
    
    print(f"\n{comparison['model1_name']}:")
    print(f"  Number of key frames: {comparison['model1_keyframes']}")
    print(f"  Coverage score: {comparison['coverage_model1']:.3f}")
    print(f"  Temporal spread: {comparison['temporal_spread_model1']:.3f}")
    print(f"  Average change score: {comparison['avg_change_model1']:.3f}")
    
    print(f"\n{comparison['model2_name']}:")
    print(f"  Number of key frames: {comparison['model2_keyframes']}")
    print(f"  Coverage score: {comparison['coverage_model2']:.3f}")
    print(f"  Temporal spread: {comparison['temporal_spread_model2']:.3f}")
    print(f"  Average change score: {comparison['avg_change_model2']:.3f}")
    
    print(f"\nComparison:")
    print(f"  Overlap: {comparison['overlap_count']} frames ({comparison['overlap_ratio']:.1%})")
    
    # Determine which model performs better
    print(f"\nAnalysis:")
    if comparison['coverage_model1'] > comparison['coverage_model2']:
        print(f"  {comparison['model1_name']} has better temporal coverage")
    else:
        print(f"  {comparison['model2_name']} has better temporal coverage")
    
    if comparison['avg_change_model1'] > comparison['avg_change_model2']:
        print(f"  {comparison['model1_name']} captures frames with higher visual changes")
    else:
        print(f"  {comparison['model2_name']} captures frames with higher visual changes")
    
    print("=" * 60 + "\n")

