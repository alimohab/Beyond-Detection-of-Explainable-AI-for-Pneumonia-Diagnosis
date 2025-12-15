"""
Video processing module for key frame detection
"""
import cv2
import numpy as np
from utils import extract_frames, get_video_info, preprocess_frame


class VideoProcessor:
    """Class for processing videos and extracting frames"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.info = get_video_info(video_path)
        self.frames = None
        
    def load_frames(self, max_frames=None, frame_interval=1):
        """Load frames from video"""
        print(f"Loading frames from {self.video_path}...")
        self.frames = extract_frames(self.video_path, max_frames, frame_interval)
        print(f"Loaded {len(self.frames)} frames")
        return self.frames
    
    def get_info(self):
        """Get video information"""
        return self.info
    
    def get_frames(self):
        """Get loaded frames"""
        if self.frames is None:
            raise ValueError("Frames not loaded. Call load_frames() first.")
        return self.frames

