"""
All-in-one keyframe detection evaluation with temporal smoothing
and score normalization for higher accuracy.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_keyframe_detection_all_in_one(
    frames_data,
    frame_scores,
    gt_percentile=80,
    pred_percentile=85,
    min_gap=5
):
    """
    Args:
        frames_data: (N, H, W, 3) normalized to [0,1]
        frame_scores: (N,) frame-level scores (e.g., AE reconstruction error)
        gt_percentile: percentile for GT generation
        pred_percentile: percentile for prediction
        min_gap: minimum frames between keyframes (temporal smoothing)
    """
    print("[INFO] Starting keyframe detection evaluation...")
    print(f"       Frames shape: {frames_data.shape}")
    print(f"       GT percentile: {gt_percentile}")
    print(f"       Prediction percentile: {pred_percentile}")
    print(f"       Temporal min_gap: {min_gap}")

    # -------------------------------------------------
    # 1. Ground Truth Generation (pixel differences)
    # -------------------------------------------------
    frame_diffs = np.mean(
        np.abs(frames_data[1:] - frames_data[:-1]),
        axis=(1, 2, 3)
    )

    gt_threshold = np.percentile(frame_diffs, gt_percentile)

    y_true = np.zeros(len(frames_data), dtype=int)
    y_true[0] = 1
    y_true[1:] = (frame_diffs > gt_threshold).astype(int)

    print(f"[INFO] Ground truth keyframes: {y_true.sum()} / {len(y_true)}")
    print(f"       GT threshold: {gt_threshold:.6f}")

    # -------------------------------------------------
    # 2. Normalize Frame Scores
    # -------------------------------------------------
    frame_scores = (frame_scores - frame_scores.mean()) / (frame_scores.std() + 1e-8)

    # -------------------------------------------------
    # 3. Prediction (percentile threshold)
    # -------------------------------------------------
    pred_threshold = np.percentile(frame_scores, pred_percentile)

    y_pred = (frame_scores > pred_threshold).astype(int)
    y_pred[0] = 1

    # -------------------------------------------------
    # 4. Temporal Smoothing
    # -------------------------------------------------
    last_kf = -min_gap
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if i - last_kf < min_gap:
                y_pred[i] = 0
            else:
                last_kf = i

    print(f"[INFO] Predicted keyframes: {y_pred.sum()} / {len(y_pred)}")
    print(f"       Prediction threshold: {pred_threshold:.6f}")

    # -------------------------------------------------
    # 5. Evaluation Metrics
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
