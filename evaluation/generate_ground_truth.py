"""
Generate ground truth labels for keyframe detection based on frame differences.
"""
import numpy as np
try:
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
except ImportError:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from KeyFrameDetection.models.cnn_lstm import build_cnn_lstm

def generate_ground_truth_labels(frames_data, percentile_threshold=75):
    """
    Generate ground truth labels for keyframe detection based on frame differences.
    Frames with significant changes are considered keyframes.
    Uses the same logic as the training label generation.
    
    Args:
        frames_data: numpy array of shape (num_frames, H, W, 3)
        percentile_threshold: Percentile threshold for determining keyframes (0-100)
    
    Returns:
        Binary array of shape (num_frames,) where 1 indicates keyframe, 0 otherwise
    """
    print(f"[INFO] Generating ground truth labels using {percentile_threshold}th percentile threshold...")
    
    # Extract features using CNN (same as in training)
    _, feature_extractor = build_cnn_lstm(input_shape=frames_data.shape[1:], sequence_length=10)
    frame_features = feature_extractor.predict(frames_data, verbose=0)
    
    # Calculate all frame differences
    frame_diffs = np.abs(np.diff(frame_features, axis=0))
    
    # Calculate percentile threshold
    threshold = np.percentile(np.mean(frame_diffs, axis=1), percentile_threshold)
    
    # Create labels: 1 for keyframe (significant change), 0 otherwise
    labels = np.zeros(len(frames_data), dtype=int)
    labels[0] = 1  # First frame is always a keyframe
    
    # For each frame after the first, check if change from previous frame exceeds threshold
    for i in range(1, len(frames_data)):
        change = np.mean(np.abs(frame_features[i] - frame_features[i-1]))
        if change > threshold:
            labels[i] = 1
    
    print(f"[INFO] Generated {np.sum(labels)} keyframes out of {len(frames_data)} frames")
    
    return labels

