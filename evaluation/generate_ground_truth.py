"""
All-in-one evaluation function for keyframe detection using frame-level scores
(e.g., autoencoder reconstruction error or any frame difference measure).
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_keyframe_detection_all_in_one(
    frames_data,
    frame_scores,
    gt_percentile=75,
    pred_percentile=75
):
    """
    Generate ground truth labels, predict keyframes, and evaluate performance.
    
    Args:
        frames_data: numpy array (N, H, W, 3), normalized to [0,1]
        frame_scores: numpy array (N,), frame-level scores
                      (e.g., autoencoder reconstruction error)
        gt_percentile: percentile threshold for ground truth generation
        pred_percentile: percentile threshold for prediction
    
    Returns:
        Dictionary containing labels, predictions, thresholds, and metrics
    """
    print("[INFO] Starting keyframe detection evaluation...")
    print(f"       Frames shape: {frames_data.shape}")
    print(f"       GT percentile: {gt_percentile}")
    print(f"       Prediction percentile: {pred_percentile}")

    # -------------------------------------------------
    # 1. Generate Ground Truth (Pixel Differences)
    # -------------------------------------------------
    frame_differences = []
    for i in range(1, len(frames_data)):
        diff = np.mean(np.abs(frames_data[i] - frames_data[i - 1]))
        frame_differences.append(diff)

    frame_differences = np.array(frame_differences)
    gt_threshold = np.percentile(frame_differences, gt_percentile)

    y_true = np.zeros(len(frames_data), dtype=int)
    y_true[0] = 1  # first frame always keyframe

    for i in range(1, len(frames_data)):
        if frame_differences[i - 1] > gt_threshold:
            y_true[i] = 1

    print(f"[INFO] Ground truth keyframes: {np.sum(y_true)} / {len(y_true)}")
    print(f"       GT threshold: {gt_threshold:.6f}")

    # -------------------------------------------------
    # 2. Generate Predictions (from scores)
    # -------------------------------------------------
    pred_threshold = np.percentile(frame_scores, pred_percentile)

    y_pred = (frame_scores > pred_threshold).astype(int)
    y_pred[0] = 1  # first frame always keyframe

    print(f"[INFO] Predicted keyframes: {np.sum(y_pred)} / {len(y_pred)}")
    print(f"       Prediction threshold: {pred_threshold:.6f}")

    # -------------------------------------------------
    # 3. Evaluation Metrics
    # -------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n[RESULTS]")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "gt_threshold": gt_threshold,
        "pred_threshold": pred_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }
