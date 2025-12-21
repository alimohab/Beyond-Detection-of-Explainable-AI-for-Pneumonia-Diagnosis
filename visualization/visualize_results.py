"""
Visualization script for keyframe detection results.
Creates plots for loss/accuracy, reconstruction errors, truth vs predictions, and confusion matrix.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Add KeyFrameDetection to path
keyframe_detection_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, keyframe_detection_root)

try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import from KeyFrameDetection
from evaluation.metrics import evaluate
from evaluation.generate_ground_truth import generate_ground_truth_labels
from preprocessing.dataset_loader import load_frames_dataset
from inference.detect_keyframes_ae import detect_keyframes_autoencoder

def plot_loss_acc(history):
    """Plot training loss and accuracy curves."""
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    if 'loss' in history:
        plt.plot(history['loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.title("Autoencoder Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy/Metrics (for autoencoder, we might have MAE)
    plt.subplot(1, 2, 2)
    if 'accuracy' in history:
        plt.plot(history['accuracy'], label='Train Acc', marker='o')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Val Acc', marker='s')
    if 'mae' in history:
        plt.plot(history['mae'], label='Train MAE', marker='o')
    if 'val_mae' in history:
        plt.plot(history['val_mae'], label='Val MAE', marker='s')
    plt.title("Autoencoder Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_autoencoder_results(model_path=None, frames_data=None, y_true=None, 
                                  results_dir=None, model=None):
    """
    Visualize autoencoder keyframe detection results.
    
    Args:
        model_path: Path to saved model (if model not provided)
        frames_data: Frames data array (will load if not provided)
        y_true: Ground truth labels (will generate if not provided)
        results_dir: Directory with saved results
        model: Loaded model (will load if not provided)
    """
    print("[INFO] Starting visualization...")
    
    # Load or use provided model
    if model is None:
        if model_path is None:
            # Try default path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'autoencoder_best.h5')
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, 'models', 'autoencoder_final.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print(f"[INFO] Loading model from {model_path}")
        try:
            model = load_model(model_path, compile=False)
        except Exception as e:
            print(f"[WARNING] Error loading model with compile=False: {e}")
            print("[INFO] Trying to load with custom objects...")
            try:
                model = load_model(model_path)
            except Exception as e2:
                print(f"[ERROR] Could not load model: {e2}")
                raise
    
    # Load or use provided frames data
    if frames_data is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        frames_dir = os.path.join(base_dir, 'data', 'Highway_Traffic', 'frames')
        print(f"[INFO] Loading frames from {frames_dir}")
        frames_data = load_frames_dataset(frames_dir, img_size=(224, 224))
    
    # Load or generate ground truth
    if y_true is None:
        print("[INFO] Generating ground truth labels...")
        y_true = generate_ground_truth_labels(frames_data, percentile_threshold=75)
    
    # Get predictions (reconstruction errors or binary predictions)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if results_dir is None:
        results_dir = os.path.join(base_dir, 'results', 'autoencoder_keyframes')
    
    # Check if results already exist
    reconstruction_errors_path = os.path.join(results_dir, 'reconstruction_errors.npy')
    binary_predictions_path = os.path.join(results_dir, 'binary_predictions.npy')
    
    if os.path.exists(reconstruction_errors_path) and os.path.exists(binary_predictions_path):
        print("[INFO] Loading existing results...")
        reconstruction_errors = np.load(reconstruction_errors_path)
        y_pred = np.load(binary_predictions_path)
    else:
        print("[INFO] Generating predictions...")
        _, y_pred = detect_keyframes_autoencoder(model, frames_data, return_binary=True)
        reconstruction_errors = np.load(os.path.join(results_dir, 'reconstruction_errors.npy'))
    
    # Normalize reconstruction errors to 0-1 range for visualization
    if reconstruction_errors.max() > 1:
        reconstruction_errors_norm = (reconstruction_errors - reconstruction_errors.min()) / (reconstruction_errors.max() - reconstruction_errors.min())
    else:
        reconstruction_errors_norm = reconstruction_errors
    
    # Plot 1: Reconstruction Errors (as probability-like scores)
    plt.figure(figsize=(12, 4))
    plt.plot(reconstruction_errors_norm, marker='o', markersize=2, linewidth=0.5, alpha=0.7)
    plt.title("Reconstruction Error (Normalized) - Higher values indicate keyframes", fontsize=12)
    plt.xlabel("Frame Index")
    plt.ylabel("Normalized Reconstruction Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Truth vs Prediction
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="Ground Truth", linewidth=2, alpha=0.8, color='green')
    plt.plot(y_pred, label="Prediction", linestyle='--', linewidth=2, alpha=0.8, color='red')
    plt.title("Ground Truth vs Prediction", fontsize=12)
    plt.xlabel("Frame Index")
    plt.ylabel("Key-frame (1=Keyframe, 0=Not)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Keyframe', 'Keyframe'])
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Autoencoder Confusion Matrix", fontsize=12, pad=20)
    plt.tight_layout()
    plt.show()
    
    # Print evaluation metrics
    precision, recall, f1 = evaluate(y_true, y_pred)
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60)
    
    # Plot 4: Reconstruction Errors with Threshold
    threshold = np.percentile(reconstruction_errors, 75)
    plt.figure(figsize=(12, 5))
    plt.plot(reconstruction_errors, label='Reconstruction Error', linewidth=1, alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.fill_between(range(len(reconstruction_errors)), 0, reconstruction_errors, 
                     where=(reconstruction_errors > threshold), alpha=0.3, color='red', label='Keyframes')
    plt.title("Reconstruction Errors with Threshold", fontsize=12)
    plt.xlabel("Frame Index")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # You can specify custom paths here or let it use defaults
    visualize_autoencoder_results()

