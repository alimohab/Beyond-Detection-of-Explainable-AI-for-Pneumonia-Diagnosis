from KeyFrameDetection.preprocessing.dataset_loader import load_frames_dataset
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def detect_keyframes_autoencoder(
    frames_dir,
    model_path,
    img_size=(224,224),
    diff_percentile=90,
    min_scene_len=5,          # prevents micro-scenes
    visualize=True,
    save_keyframes=True,
    output_dir="results/autoencoder_keyframes"
):
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
    train_data = load_frames_dataset(frames_dir, img_size=img_size)
    
    # Calculate reconstruction error (MSE) for each frame
    print("[INFO] Loading AE...")
    model = load_model(model_path, compile=False)    
    # Threshold based on percentile
    print("[INFO] Reconstructing...")
    recon = model.predict(train_data, verbose=0)
    errors = np.mean((train_data - recon) ** 2, axis=(1, 2, 3))
    
    diffs = np.abs(np.diff(errors))
    threshold = np.percentile(diffs, diff_percentile)
    scene_boundaries = np.where(diffs > threshold)[0] + 1

    print(f"[INFO] Detected {len(scene_boundaries)} keyframes out of {len(train_data)} frames")
    print(f"       Threshold: {threshold:.4f}")
    
    # Save results
    scenes = []
    start = 0

    for cut in scene_boundaries:
        if cut - start >= min_scene_len:
            scenes.append((start, cut))
        start = cut

    if len(train_data) - start >= min_scene_len:
        scenes.append((start, len(train_data)))

    print(f"[INFO] Detected {len(scenes)} scenes")

    # -----------------------------
    # 3. One keyframe per scene
    # -----------------------------
    keyframes = []
    for (s, e) in scenes:
        idx = s + np.argmax(errors[s:e])   # index ONLY
        keyframes.append(idx)

    keyframes = np.array(keyframes)
    print(f"[RESULT] Returned {len(keyframes)} keyframes (1 per scene)")

    # -----------------------------
    # Save ORIGINAL keyframe images
    # -----------------------------
    if save_keyframes:
        os.makedirs(output_dir, exist_ok=True)

        for idx in keyframes:
            # Convert normalized frame back to uint8
            original_frame = (train_data[idx] * 255).astype(np.uint8)

            # Save directly, no color conversion needed
            cv2.imwrite(
                os.path.join(output_dir, f"scene_{idx:05d}.jpg"),
                original_frame
            )
    print(f"[INFO] Saved keyframes to {output_dir}")

    # -----------------------------
    # Visualization (SAVE + SHOW)
    # -----------------------------
    if visualize:
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 4))
        plt.plot(errors, label="Reconstruction Error")
        plt.scatter(keyframes, errors[keyframes], c="red", label="Scene Keyframes")
        plt.xlabel("Frame Index")
        plt.ylabel("Reconstruction Error")
        plt.title("Scene-based AE Keyframe Detection")
        plt.legend()
        plt.tight_layout()

        # SAVE visualization
        vis_path = os.path.join(output_dir, "reconstruction_error_plot.png")
        plt.savefig(vis_path, dpi=200)
        print(f"[INFO] Visualization saved to {vis_path}")

        plt.show()

    return keyframes, errors

