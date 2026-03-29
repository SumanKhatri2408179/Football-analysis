
Copy

import cv2
import threading
import time
from trackers import Tracker
 
# ── Single shared Tracker instance (reuses your existing model) ───────────
_tracker      = None
stream_active = False
latest_frame  = None
frame_lock    = threading.Lock()
tracking_data = {}
 
 
def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = Tracker("models/best1.pt")
        print("Live tracker: model loaded")
    return _tracker
 
 
def run_tracking(ip_url: str):
    global stream_active, latest_frame, tracking_data
 
    tracker = _get_tracker()
    tracker.live_ball_trail.clear()   # reset trail for new session
 
    print(f"Connecting to IPWebcam: {ip_url}")
    cap = cv2.VideoCapture(ip_url)
 
    if not cap.isOpened():
        print("ERROR: Cannot connect to IPWebcam stream")
        stream_active = False
        return
 
    frame_count = 0
 
    while stream_active:
        ret, frame = cap.read()
        if not ret:
            print("Connection lost, retrying in 2 seconds...")
            time.sleep(2)
            cap = cv2.VideoCapture(ip_url)
            continue
 
        frame_count += 1
 
        try:
            # ── This calls your Tracker.live_track_frame() ───────────────
            annotated_frame, stats = tracker.live_track_frame(frame)
 
            with frame_lock:
                latest_frame  = annotated_frame.copy()
                tracking_data = stats
 
        except Exception as e:
            print(f"Error on frame {frame_count}: {e}")
            with frame_lock:
                latest_frame = frame.copy()
 
    cap.release()
    print("Live tracking stopped.")
 
 
def generate_frames():
    """MJPEG generator for FastAPI StreamingResponse."""
    while stream_active:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            _, buffer = cv2.imencode(
                '.jpg', latest_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
 
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes + b'\r\n'
        )
        time.sleep(0.033)   # ~30 FPS