# import cv2
# import threading
# import time
# from trackers import Tracker
 
# # Single shared Tracker instance (reuses your existing model) 
# _tracker      = None
# stream_active = False
# latest_frame  = None
# frame_lock    = threading.Lock()
# tracking_data = {}
 
 
# def _get_tracker():
#     global _tracker
#     if _tracker is None:
#         _tracker = Tracker("models/best1.pt")
#         print("Live tracker: model loaded")
#     return _tracker
 
 
# def run_tracking(ip_url: str):
#     global stream_active, latest_frame, tracking_data
 
#     tracker = _get_tracker()
#     tracker.live_ball_trail.clear()   # reset trail for new session
 
#     print(f"Connecting to IPWebcam: {ip_url}")
#     cap = cv2.VideoCapture(ip_url)
 
#     if not cap.isOpened():
#         print("ERROR: Cannot connect to IPWebcam stream")
#         stream_active = False
#         return
 
#     frame_count = 0
 
#     while stream_active:
#         ret, frame = cap.read()
#         if not ret:
#             print("Connection lost, retrying in 2 seconds...")
#             time.sleep(2)
#             cap = cv2.VideoCapture(ip_url)
#             continue
 
#         frame_count += 1
 
#         try:
#             # This calls your Tracker.live_track_frame()
#             annotated_frame, stats = tracker.live_track_frame(frame)
 
#             with frame_lock:
#                 latest_frame  = annotated_frame.copy()
#                 tracking_data = stats
 
#         except Exception as e:
#             print(f"Error on frame {frame_count}: {e}")
#             with frame_lock:
#                 latest_frame = frame.copy()
 
#     cap.release()
#     print("Live tracking stopped.")
 
 
# def generate_frames():
#     """MJPEG generator for FastAPI StreamingResponse."""
#     while stream_active:
#         with frame_lock:
#             if latest_frame is None:
#                 time.sleep(0.05)
#                 continue
#             _, buffer = cv2.imencode(
#                 '.jpg', latest_frame,
#                 [cv2.IMWRITE_JPEG_QUALITY, 85])
#             frame_bytes = buffer.tobytes()
 
#         yield (
#             b'--frame\r\n'
#             b'Content-Type: image/jpeg\r\n\r\n'
#             + frame_bytes + b'\r\n'
#         )
#         time.sleep(0.033)   # ~30 FPS
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

# ── Speed settings ────────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 2    # process every 2nd frame — halves CPU load
RESIZE_WIDTH           = 640  # resize frame before YOLO — much faster
JPEG_QUALITY           = 75   # lower = faster streaming (was 85)
STREAM_FPS             = 15   # stream at 15fps instead of 30fps


def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = Tracker("models/best1.pt")
        print("Live tracker: model loaded")
    return _tracker


def resize_frame(frame):
    """Resize frame to RESIZE_WIDTH before YOLO processing — faster detection."""
    h, w = frame.shape[:2]
    if w > RESIZE_WIDTH:
        scale  = RESIZE_WIDTH / w
        frame  = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
    return frame


def run_tracking(ip_url: str):
    global stream_active, latest_frame, tracking_data

    tracker = _get_tracker()
    tracker.live_ball_trail.clear()  # reset trail for new session

    print(f"Connecting to IPWebcam: {ip_url}")
    cap = cv2.VideoCapture(ip_url)

    # ── OpenCV buffer settings — reduces latency ──────────────────────────
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # only keep 1 frame in buffer
    cap.set(cv2.CAP_PROP_FPS, 15)         # request 15fps from stream

    if not cap.isOpened():
        print("ERROR: Cannot connect to IPWebcam stream")
        stream_active = False
        return

    frame_count    = 0
    last_annotated = None  # reuse last annotated frame on skipped frames

    while stream_active:
        ret, frame = cap.read()
        if not ret:
            print("Connection lost, retrying in 2 seconds...")
            time.sleep(2)
            cap = cv2.VideoCapture(ip_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            continue

        frame_count += 1

        # ── Skip every other frame to reduce CPU load ─────────────────────
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            # On skipped frames, reuse last annotated frame
            if last_annotated is not None:
                with frame_lock:
                    latest_frame = last_annotated
            continue

        try:
            # ── Resize before YOLO — significantly faster ─────────────────
            small_frame = resize_frame(frame)

            # ── Run YOLOv8 + ByteTrack ────────────────────────────────────
            annotated_frame, stats = tracker.live_track_frame(small_frame)

            last_annotated = annotated_frame.copy()

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
    frame_interval = 1.0 / STREAM_FPS  # 0.066s at 15fps

    while stream_active:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            # ── Lower JPEG quality = smaller payload = faster streaming ───
            _, buffer = cv2.imencode(
                '.jpg', latest_frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes + b'\r\n'
        )
        time.sleep(frame_interval)  # 15fps instead of 30fps