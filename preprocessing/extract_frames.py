import cv2
import os

def extract_frames(video_path, output_dir, fps=2):
    """
    Extract frames from a single video at a given FPS.
    Saves frames to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps else 0
    print(f"[INFO] Processing video: {video_path}")
    print(f"       FPS: {video_fps:.2f}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

    frame_interval = max(int(video_fps // fps), 1)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            if frame is None:
                print(f"[WARNING] Skipping invalid frame at index {count}")
                continue
            frame_resized = cv2.resize(frame, (224, 224))
            frame_file = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            success = cv2.imwrite(frame_file, frame_resized)
            if not success:
                print(f"[WARNING] Failed to save frame {saved} to {frame_file}")
                continue
            print(f"[FRAME] Saved frame {saved} (Original Index: {count})")
            saved += 1

        count += 1

    cap.release()
    print(f"[DONE] Finished extracting {saved} frames from {video_path}\n")
    return saved


def extract_cars_dataset(video_dir, output_root, fps=2):
    """
    Extract frames from all videos in a dataset directory.
    Saves frames in separate folders for each video.
    """
    if not os.path.exists(video_dir):
        print(f"[ERROR] Video directory does not exist: {video_dir}")
        return
    
    os.makedirs(output_root, exist_ok=True)
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"[WARNING] No video files found in: {video_dir}")
        return

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_root, video_name)

        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"[SKIP] Frames already exist for video: {video_file}")
            continue

        extract_frames(video_path, output_dir, fps=fps)

    print("[INFO] All videos processed.")

if __name__ == "__main__":
    # Extract frames from Highway_Traffic dataset
    HIGHWAY_VIDEO_DIR = "KeyFrameDetection/data/Highway_Traffic/video"
    HIGHWAY_OUTPUT_FRAME_DIR = "KeyFrameDetection/data/Highway_Traffic/frames"
    extract_cars_dataset(HIGHWAY_VIDEO_DIR, HIGHWAY_OUTPUT_FRAME_DIR, fps=2)
    
