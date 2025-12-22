"""
Main script to run the full key frame detection training pipeline.
1. Extract frames from SumMe videos
2. Load frames dataset
3. Train Autoencoder
4. Train CNN + LSTM
"""

import argparse
import os
from tensorflow.keras.models import load_model

from KeyFrameDetection.preprocessing.extract_frames import extract_cars_dataset
from KeyFrameDetection.preprocessing.dataset_loader import load_frames_dataset
from KeyFrameDetection.inference.detect_keyframes_ae import detect_keyframes_autoencoder
from KeyFrameDetection.inference.detect_keyframes_lstm import detect_keyframes_cnn_lstm
from KeyFrameDetection.evaluation.generate_ground_truth import generate_ground_truth_labels
from KeyFrameDetection.evaluation.metrics import evaluate
from KeyFrameDetection.training.train_autoencoder import train_autoencoder
from KeyFrameDetection.training.train_cnn_lstm import train_cnn_lstm

DEFAULT_TRAIN_VIDEO_DIR = "KeyFrameDetection\data\Highway_Traffic\video"
DEFAULT_TRAIN_FRAMES_DIR = "KeyFrameDetection\data\Highway_Traffic\frames"
DEFAULT_TEST_DIR = "KeyFrameDetection/data/test_video/frames"
DEFAULT_AE_MODEL = "KeyFrameDetection/models/autoencoder_best.h5"
DEFAULT_LSTM_MODEL = "KeyFrameDetection/models/cnn_lstm_best.h5"
FPS = 2

IMG_SIZE = (224, 224)
BATCH_SIZE_AE = 16
EPOCHS_AE = 15

SEQ_LEN = 20
BATCH_SIZE_LSTM = 2
EPOCHS_LSTM = 30


# -----------------------------
# Step 1: Extract frames
# -----------------------------
# print("[STEP 1] Extracting frames from SumMe videos...")
extract_cars_dataset(DEFAULT_TRAIN_VIDEO_DIR, DEFAULT_TRAIN_FRAMES_DIR, fps=FPS)

# -----------------------------
# Step 2: Load frames dataset
# -----------------------------
print("[STEP 2] Loading frames dataset for training...")
frames_dataset = load_frames_dataset(DEFAULT_TRAIN_VIDEO_DIR, img_size=IMG_SIZE)
print(f"[INFO] Total frames loaded: {frames_dataset.shape[0]}")

# # -----------------------------
# # Step 3: Train Autoencoder
# # -----------------------------
# print("[STEP 3] Training Autoencoder (auto-tuned)...")

ae_model, ae_history = train_autoencoder(
    frames=frames_dataset,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE_AE,
    max_epochs=50,          # ← NOT fixed epochs anymore
    save_path="KeyFrameDetection/models/autoencoder.h5"
)

print("[STEP 5] Detecting keyframes using Autoencoder...")

keyframes_ae, errors_ae = detect_keyframes_autoencoder(
    frames_dir=DEFAULT_TEST_DIR,
    model_path="KeyFrameDetection/models/autoencoder.h5",
    img_size=IMG_SIZE,
    diff_percentile=90,
    min_scene_len=3,
    visualize=True,
    save_keyframes=True,
    output_dir="KeyFrameDetection/results/autoencoder_keyframes"
)


print("[DONE] Autoencoder keyframe detection completed.")

# -----------------------------
# Step 4: Train CNN + LSTM
# -----------------------------
print("[STEP 4] Training CNN + LSTM...")
train_cnn_lstm(
    frames_dataset,   # ← Changed from DEFAULT_TRAIN_FRAMES_DIR to frames_dataset
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE_LSTM,
    max_epochs=EPOCHS_LSTM,
    save_path="models/cnn_lstm.h5"
)
print("[DONE] CNN+LSTM training completed.")


print("[STEP 5] Detecting keyframes using CNN+LSTM...")

keyframes, scores = detect_keyframes_cnn_lstm(
    frames_dir=DEFAULT_TEST_DIR,
    model_path="models/cnn_lstm.h5",
    img_size=(224, 224),
    seq_len=20,
    diff_percentile=85,   # adjust sensitivity
    min_scene_len=3,
    visualize=True,
    save_keyframes=True,
    output_dir="results/cnn_lstm_keyframes"
)

print("Detected keyframes indices:", keyframes)

print("[DONE] CNN+LSTM keyframe detection completed.")

print("[DONE] Training pipeline completed successfully!")




"""
Main entry point for keyframe detection.

This script lets you run inference with the trained models (autoencoder and/or
CNN-LSTM) on a given frames directory.

Usage examples:
  python main.py --mode detect_lstm
  python main.py --mode detect_ae
  python main.py --mode detect_both --test_dir KeyFrameDetection/data/test_video/frames
"""






def run_detect_ae(test_dir, ae_model_path, img_size):
    if not os.path.exists(ae_model_path):
        raise FileNotFoundError(f"Autoencoder model not found: {ae_model_path}")
    print(f"[INFO] Loading autoencoder model from {ae_model_path}")
    ae_model = load_model(ae_model_path, compile=False)

    print(f"[INFO] Loading test frames from {test_dir}")
    test_data = load_frames_dataset(test_dir, img_size=img_size)
    print(f"[INFO] Loaded {len(test_data)} frames")

    _, ae_predictions = detect_keyframes_autoencoder(ae_model, test_data, return_binary=True)
    test_ground_truth = generate_ground_truth_labels(test_data, percentile_threshold=75)
    ae_precision, ae_recall, ae_f1 = evaluate(test_ground_truth, ae_predictions)
    print("\n[RESULT] Autoencoder Evaluation")
    print(f"  Precision: {ae_precision:.4f}")
    print(f"  Recall:    {ae_recall:.4f}")
    print(f"  F1-Score:  {ae_f1:.4f}")


def run_detect_lstm(test_dir, lstm_model_path, img_size, seq_len, threshold=0.5):
    if not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"CNN-LSTM model not found: {lstm_model_path}")
    print(f"[INFO] Loading CNN-LSTM model from {lstm_model_path}")
    lstm_model = load_model(lstm_model_path, compile=False)

    print(f"[INFO] Loading test frames from {test_dir}")
    test_data = load_frames_dataset(test_dir, img_size=img_size)
    print(f"[INFO] Loaded {len(test_data)} frames")

    test_ground_truth = generate_ground_truth_labels(test_data, percentile_threshold=75)
    _, lstm_predictions = detect_keyframes_cnn_lstm(
        lstm_model,
        test_data,
        sequence_length=seq_len,
        threshold=threshold,
        return_binary=True
    )
    lstm_precision, lstm_recall, lstm_f1 = evaluate(test_ground_truth, lstm_predictions)
    print("\n[RESULT] CNN-LSTM Evaluation")
    print(f"  Precision: {lstm_precision:.4f}")
    print(f"  Recall:    {lstm_recall:.4f}")
    print(f"  F1-Score:  {lstm_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Keyframe detection runner")
    parser.add_argument("--mode", choices=["detect_ae", "detect_lstm", "detect_both"],
                        default="detect_lstm", help="Which model(s) to run")
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR,
                        help="Directory of frames to run inference on")
    parser.add_argument("--ae_model", type=str, default=DEFAULT_AE_MODEL,
                        help="Path to autoencoder model")
    parser.add_argument("--lstm_model", type=str, default=DEFAULT_LSTM_MODEL,
                        help="Path to CNN-LSTM model")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224],
                        help="Image size for loading frames")
    parser.add_argument("--seq_len", type=int, default=10,
                        help="Sequence length for CNN-LSTM")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for CNN-LSTM keyframe prediction")
    args = parser.parse_args()

    img_size = tuple(args.img_size)

    if args.mode in ("detect_ae", "detect_both"):
        run_detect_ae(args.test_dir, args.ae_model, img_size)

    if args.mode in ("detect_lstm", "detect_both"):
        run_detect_lstm(args.test_dir, args.lstm_model, img_size, args.seq_len, args.threshold)


if __name__ == "__main__":
    main()