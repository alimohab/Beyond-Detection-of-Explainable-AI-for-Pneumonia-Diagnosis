"""
Example script to run key frame detection on the provided video
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from main import main
import argparse

if __name__ == '__main__':
    # Example: Run on the video in the root directory
    video_path = '3032568-uhd_3840_2160_25fps.mp4'
    
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        print("Please update the video_path in this script or provide a video file.")
        sys.exit(1)
    
    # Create argument list
    sys.argv = [
        'run_example.py',
        '--video', video_path,
        '--method', 'both',
        '--compare',
        '--n_keyframes', '5',
        '--frame_interval', '2'  # Process every 2nd frame for faster processing
    ]
    
    main()

