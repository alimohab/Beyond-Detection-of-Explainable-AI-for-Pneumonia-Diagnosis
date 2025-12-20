import numpy as np
import os
import sys

# Add parent directory to path to import from models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KeyFrameDetection.models.cnn_lstm import build_cnn_lstm

def detect_keyframes_cnn_lstm(model, frames_data, sequence_length=10, threshold=0.5, return_binary=False):
    """
    Detect keyframes using a CNN-LSTM model.
    
    Args:
        model: Trained CNN-LSTM model
        frames_data: numpy array of shape (num_frames, H, W, 3)
        sequence_length: Length of sequences used during training
        threshold: Probability threshold for keyframe classification
        return_binary: If True, also return binary predictions array
    
    Returns:
        List of keyframe indices (and binary array if return_binary=True)
    """
    print("[INFO] Detecting keyframes using CNN-LSTM...")
    
    # Extract CNN features using the same architecture as in build_cnn_lstm
    _, feature_extractor = build_cnn_lstm(input_shape=frames_data.shape[1:], sequence_length=sequence_length)
    
    # Extract features for all frames
    frame_features = feature_extractor.predict(frames_data, verbose=0)
    
    # Create sequences and predict
    sequences = []
    for i in range(sequence_length, len(frame_features)):
        seq = frame_features[i-sequence_length:i]
        sequences.append(seq)
    
    sequences = np.array(sequences)
    predictions = model.predict(sequences, verbose=0)
    
    # Create binary predictions array for all frames
    binary_predictions = np.zeros(len(frames_data), dtype=int)
    # Set predictions for frames after sequence_length
    sequence_predictions = (predictions.flatten() > threshold).astype(int)
    binary_predictions[sequence_length:] = sequence_predictions
    # First frame is always considered a keyframe
    binary_predictions[0] = 1
    
    # Get keyframe indices
    keyframe_indices = np.where(binary_predictions)[0].tolist()
    
    print(f"[INFO] Detected {len(keyframe_indices)} keyframes out of {len(frames_data)} frames")
    print(f"       Keyframe indices: {keyframe_indices[:10]}..." if len(keyframe_indices) > 10 else f"       Keyframe indices: {keyframe_indices}")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'cnn_lstm_keyframes')
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, 'keyframe_indices.npy'), keyframe_indices)
    np.save(os.path.join(results_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(results_dir, 'binary_predictions.npy'), binary_predictions)
    
    print("[INFO] Results saved to results/cnn_lstm_keyframes/")
    
    if return_binary:
        return keyframe_indices, binary_predictions
    return keyframe_indices

