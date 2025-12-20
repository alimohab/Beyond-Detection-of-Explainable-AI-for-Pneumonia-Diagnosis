import numpy as np
import os

def detect_keyframes_autoencoder(model, frames_data, threshold_percentile=75, return_binary=False):
    """
    Detect keyframes using an autoencoder model.
    Frames with high reconstruction error are considered keyframes.
    
    Args:
        model: Trained autoencoder model
        frames_data: numpy array of shape (num_frames, H, W, 3)
        threshold_percentile: Percentile to use as threshold for keyframe detection
        return_binary: If True, also return binary predictions array
    
    Returns:
        List of keyframe indices (and binary array if return_binary=True)
    """
    print("[INFO] Detecting keyframes using autoencoder...")
    
    # Predict reconstructed frames
    reconstructed = model.predict(frames_data, verbose=0)
    
    # Calculate reconstruction error (MSE) for each frame
    reconstruction_errors = np.mean((frames_data - reconstructed) ** 2, axis=(1, 2, 3))
    
    # Threshold based on percentile
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    # Keyframes are frames with reconstruction error above threshold
    binary_predictions = (reconstruction_errors > threshold).astype(int)
    keyframe_indices = np.where(binary_predictions)[0].tolist()
    
    print(f"[INFO] Detected {len(keyframe_indices)} keyframes out of {len(frames_data)} frames")
    print(f"       Threshold: {threshold:.4f}")
    print(f"       Keyframe indices: {keyframe_indices[:10]}..." if len(keyframe_indices) > 10 else f"       Keyframe indices: {keyframe_indices}")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'autoencoder_keyframes')
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, 'keyframe_indices.npy'), keyframe_indices)
    np.save(os.path.join(results_dir, 'reconstruction_errors.npy'), reconstruction_errors)
    np.save(os.path.join(results_dir, 'binary_predictions.npy'), binary_predictions)
    
    print("[INFO] Results saved to results/autoencoder_keyframes/")
    
    if return_binary:
        return keyframe_indices, binary_predictions
    return keyframe_indices

