import numpy as np
import os
import sys

# Add parent directory to path to import from models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KeyFrameDetection.models.cnn_lstm import build_cnn_lstm

def find_optimal_threshold(model, sequences, y_true, thresholds=None):
    """
    Find the optimal threshold that maximizes F1-score.
    
    Args:
        model: Trained CNN-LSTM model
        sequences: Input sequences for prediction
        y_true: Ground truth labels
        thresholds: List of thresholds to try. If None, uses a range from 0.1 to 0.9
    
    Returns:
        Optimal threshold value and corresponding F1-score
    """
    from sklearn.metrics import f1_score
    
    predictions = model.predict(sequences, verbose=0).flatten()
    
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    best_threshold = 0.5
    best_f1 = 0
    
    for thresh in thresholds:
        y_pred = (predictions > thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1

def detect_keyframes_cnn_lstm(model, frames_data, sequence_length=10, threshold=0.5, return_binary=False, feature_extractor_model_path=None):
    """
    Detect keyframes using a CNN-LSTM model.
    
    Args:
        model: Trained CNN-LSTM model
        frames_data: numpy array of shape (num_frames, H, W, 3)
        sequence_length: Length of sequences used during training
        threshold: Probability threshold for keyframe classification
        return_binary: If True, also return binary predictions array
        feature_extractor_model_path: Path to saved feature extractor model. If None, builds a new one (NOT RECOMMENDED)
    
    Returns:
        List of keyframe indices (and binary array if return_binary=True)
    """
    print("[INFO] Detecting keyframes using CNN-LSTM...")
    
    # Create sequences directly from frames (model expects frame sequences, not features)
    print("[INFO] Creating frame sequences for prediction...")
    sequences = []
    for i in range(sequence_length, len(frames_data)):
        seq = frames_data[i-sequence_length:i]
        sequences.append(seq)
    
    sequences = np.array(sequences)
    print(f"[INFO] Created {len(sequences)} sequences of shape {sequences.shape}")
    
    # Use the end-to-end model to predict (it will extract features and apply LSTM internally)
    predictions = model.predict(sequences, verbose=0)
    prediction_probs = predictions.flatten()
    
    # Use provided threshold (can be optimal threshold found via find_optimal_threshold)
    print(f"[INFO] Using threshold: {threshold:.4f}")
    print(f"[INFO] Prediction probabilities - min: {np.min(prediction_probs):.4f}, max: {np.max(prediction_probs):.4f}, mean: {np.mean(prediction_probs):.4f}, median: {np.median(prediction_probs):.4f}, std: {np.std(prediction_probs):.4f}")
    
    # Diagnostic: Check prediction distribution
    unique_vals, counts = np.unique(prediction_probs.round(decimals=2), return_counts=True)
    print(f"[INFO] Prediction distribution: {len(unique_vals)} unique values (rounded to 2 decimals)")
    if len(unique_vals) < 5:
        print(f"[WARNING] Very few unique prediction values! This suggests predictions are too similar.")
        print(f"         Unique values: {unique_vals[:10]}")
        print(f"         Counts: {counts[:10]}")
    
    # Create binary predictions array for all frames
    binary_predictions = np.zeros(len(frames_data), dtype=int)
    # Set predictions for frames after sequence_length
    sequence_predictions = (prediction_probs > threshold).astype(int)
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

