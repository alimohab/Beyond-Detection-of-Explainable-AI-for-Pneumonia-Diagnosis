"""
Utility functions for video processing and key frame detection
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os


def extract_frames(video_path, max_frames=None, frame_interval=1):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None for all)
        frame_interval: Extract every Nth frame
    
    Returns:
        List of frames (numpy arrays)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            if max_frames and len(frames) >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return frames


def get_video_info(video_path):
    """
    Get video information
    
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return info


def preprocess_frame(frame, size=(224, 224)):
    """
    Preprocess frame for CNN input
    
    Args:
        frame: RGB frame (numpy array)
        size: Target size (width, height)
    
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    frame_tensor = transform(frame)
    return frame_tensor.unsqueeze(0)  # Add batch dimension


def save_keyframes(keyframe_indices, frames, output_dir, video_name):
    """
    Save detected key frames as images
    
    Args:
        keyframe_indices: List of frame indices
        frames: List of all frames
        output_dir: Output directory
        video_name: Name of video (for naming files)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, frame_idx in enumerate(keyframe_indices):
        frame = frames[frame_idx]
        frame_pil = Image.fromarray(frame)
        output_path = os.path.join(output_dir, f"{video_name}_keyframe_{idx+1}_frame_{frame_idx}.jpg")
        frame_pil.save(output_path)
    
    print(f"Saved {len(keyframe_indices)} key frames to {output_dir}")


def calculate_frame_differences(frames):
    """
    Calculate differences between consecutive frames
    
    Args:
        frames: List of frames
    
    Returns:
        Array of differences
    """
    differences = []
    for i in range(1, len(frames)):
        # Convert to grayscale and calculate difference
        frame1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(frame1, frame2)
        differences.append(np.sum(diff))
    
    return np.array(differences)


def get_device():
    """Get available device (CUDA or CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

