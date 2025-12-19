import os
import cv2
import numpy as np

def load_frames_dataset(frames_root, img_size=(224,224)):
    """
    Load all frames from the frames_root folder.
    Returns a NumPy array of shape (num_frames, H, W, 3)
    """
    if not os.path.exists(frames_root):
        raise FileNotFoundError(f"Frames root directory does not exist: {frames_root}")
    
    frames_list = []
    for video_folder in os.listdir(frames_root):
        video_path = os.path.join(frames_root, video_folder)
        if not os.path.isdir(video_path):
            continue
        for frame_file in sorted(os.listdir(video_path)):
            if frame_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                frame_path = os.path.join(video_path, frame_file)
                img = cv2.imread(frame_path)
                if img is None:
                    print(f"[WARNING] Failed to read image: {frame_path}")
                    continue
                img = cv2.resize(img, img_size)
                img = img / 255.0
                frames_list.append(img)
    
    if not frames_list:
        raise ValueError(f"No frames found in: {frames_root}")
    
    return np.array(frames_list)