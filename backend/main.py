import os
import cv2
import uvicorn
import numpy as np
import torch
import threading
import time
import json
import subprocess
from collections import deque
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from ultralytics import YOLO
import logging
import aiofiles
import traceback
import shutil
from database import Base, engine
from auth import router as auth_router
import models
from player_rating import PlayerRatingSystem

# FastAPI App
app = FastAPI(debug=True)

# Create database tables
Base.metadata.create_all(bind=engine)

# Include auth routes
app.include_router(auth_router, prefix="/auth",  tags=["auth"])
app.include_router(auth_router, prefix="/users", tags=["users"])

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
input_videos_dir  = "input_videos"
output_videos_dir = "output_videos"
stubs_dir         = "stubs"
os.makedirs(input_videos_dir,  exist_ok=True)
os.makedirs(output_videos_dir, exist_ok=True)
os.makedirs(stubs_dir,         exist_ok=True)

# Chunk size for streaming
CHUNK_SIZE = 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print GPU info at startup
print("=" * 50)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("GPU: Not available - using CPU")
print("=" * 50)

# Live Tracking — Global State
live_model     = YOLO("models/best1.pt")
stream_active  = False
latest_frame   = None
frame_lock     = threading.Lock()
tracking_data  = []
ball_trail     = deque(maxlen=25)

# Colours (BGR)
COLOR_PLAYER  = (0,   255,   0)
COLOR_BALL    = (0,   255, 255)
COLOR_REFEREE = (255, 165,   0)
COLOR_TRAIL   = (0,   200, 255)


def draw_ball_trail(frame):
    for i in range(1, len(ball_trail)):
        if ball_trail[i - 1] is None or ball_trail[i] is None:
            continue
        thickness = max(1, int(np.sqrt(25 / float(i + 1)) * 2))
        cv2.line(frame, ball_trail[i - 1], ball_trail[i], COLOR_TRAIL, thickness)


def draw_player(frame, x1, y1, x2, y2, conf):
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PLAYER, 2)
    label = f"Player {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLOR_PLAYER, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 3, COLOR_PLAYER, -1)


def draw_ball(frame, x1, y1, x2, y2, conf):
    cx     = (x1 + x2) // 2
    cy     = (y1 + y2) // 2
    radius = max((x2 - x1), (y2 - y1)) // 2 + 4
    ball_trail.append((cx, cy))
    draw_ball_trail(frame)
    cv2.circle(frame, (cx, cy), radius + 4, COLOR_BALL, 1)
    cv2.circle(frame, (cx, cy), radius, COLOR_BALL, 3)
    cv2.line(frame, (cx - radius - 8, cy), (cx + radius + 8, cy), COLOR_BALL, 1)
    cv2.line(frame, (cx, cy - radius - 8), (cx, cy + radius + 8), COLOR_BALL, 1)
    label = f"Ball {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame,
                  (cx - tw // 2 - 4, y1 - th - 12),
                  (cx + tw // 2 + 4, y1 - 4),
                  COLOR_BALL, -1)
    cv2.putText(frame, label,
                (cx - tw // 2, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_referee(frame, x1, y1, x2, y2, conf):
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_REFEREE, 2)
    label = f"Referee {conf:.0%}"
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_REFEREE, 2)


def run_tracking(ip_url: str):
    global stream_active, latest_frame, tracking_data
    logger.info(f"Connecting to IPWebcam: {ip_url}")
    cap = cv2.VideoCapture(ip_url)
    if not cap.isOpened():
        logger.error("ERROR: Cannot connect to IPWebcam stream")
        stream_active = False
        return
    frame_count  = 0
    ball_missing = 0
    while stream_active:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Connection lost, retrying in 2 seconds...")
            time.sleep(2)
            cap = cv2.VideoCapture(ip_url)
            continue
        frame_count  += 1
        frame_results = []
        ball_found    = False
        try:
            results = live_model.predict(frame, conf=0.35, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf  = float(box.conf[0])
                    cls   = int(box.cls[0])
                    label = live_model.names[cls]
                    cx    = (x1 + x2) // 2
                    cy    = (y1 + y2) // 2
                    if label == "player":
                        draw_player(frame, x1, y1, x2, y2, conf)
                    elif label == "ball":
                        draw_ball(frame, x1, y1, x2, y2, conf)
                        ball_found   = True
                        ball_missing = 0
                    elif label == "referee":
                        draw_referee(frame, x1, y1, x2, y2, conf)
                    frame_results.append({
                        "label":      label,
                        "confidence": round(conf, 2),
                        "bbox":       [x1, y1, x2, y2],
                        "center":     [cx, cy]
                    })
            if not ball_found:
                ball_missing += 1
                if ball_missing <= 10:
                    draw_ball_trail(frame)
                else:
                    ball_trail.clear()
            player_count = sum(1 for r in frame_results if r["label"] == "player")
            h, w         = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (280, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (280, 80), (50, 50, 50),  1)
            cv2.putText(frame, f"Players: {player_count}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, COLOR_PLAYER, 2)
            ball_text  = "Ball: Detected" if ball_found else "Ball: Not found"
            ball_color = COLOR_BALL if ball_found else (100, 100, 100)
            cv2.putText(frame, ball_text,
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, ball_color, 2)
            cv2.circle(frame, (w - 30, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, "LIVE",
                        (w - 72, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 150, 150), 1)
        except Exception as e:
            logger.error(f"Detection error on frame {frame_count}: {e}")
        with frame_lock:
            latest_frame  = frame.copy()
            tracking_data = frame_results
    cap.release()
    ball_trail.clear()
    logger.info("Live tracking stopped.")


def generate_frames():
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
        time.sleep(0.033)


def convert_to_mp4(avi_path: str, mp4_path: str) -> bool:
    """
    Convert AVI to MP4 using a temp file to avoid Windows file-lock conflicts.
    Returns True on success, False on failure.
    """
    dir_name = os.path.dirname(mp4_path)
    tmp_path = os.path.join(dir_name, f"_tmp_{os.path.basename(mp4_path)}")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", avi_path,
                "-c:v", "libx264",
                "-crf", "23",
                "-movflags", "+faststart",  # enables streaming before full download
                tmp_path
            ],
            capture_output=True
        )

        if result.returncode == 0:
            # Atomic swap — avoids the Windows file-lock problem
            os.replace(tmp_path, mp4_path)
            return True
        else:
            print(f"FFmpeg error:\n{result.stderr.decode()}")
            return False

    except Exception as e:
        print(f"FFmpeg exception: {e}")
        return False
    finally:
        # Always clean up the temp file if it still exists
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def process_video(input_video_path):
    try:
        video_name            = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path_avi = os.path.join(output_videos_dir, f"{video_name}.avi")
        output_video_path_mp4 = os.path.join(output_videos_dir, f"{video_name}.mp4")
        stub_path             = f'stubs/{video_name}_track_stubs.pkl'

        print(f"Processing video: {input_video_path}")
        video_frames = read_video(input_video_path)

        tracker = Tracker('models/best1.pt')
        tracks  = tracker.get_object_tracks(
            video_frames, read_from_stub=True, stub_path=stub_path)
        tracker.add_position_to_tracks(tracks)

        camera_estimator          = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_estimator.get_camera_movement(
            video_frames, read_from_stub=True,
            stub_path='stubs/camera_movement_stub.pkl')
        camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

        frame_height, frame_width, _ = video_frames[0].shape
        view_transformer = ViewTransformer((frame_height, frame_width))
        view_transformer.add_transformed_position_to_tracks(tracks)

        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        speed_distance_estimator = SpeedAndDistance_Estimator()
        speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        team_assigner = TeamAssigner(device=device, video_path=input_video_path)
        team_assigner.load_team_assignments()

        for frame_num, player_track in enumerate(tracks['players']):
            player_ids    = list(player_track.keys())
            player_bboxes = [player_track[pid]["bbox"] for pid in player_ids]

            # FIX: extract_player_crops now returns (crops, valid_ids) so they are always in sync
            player_crops, valid_ids = team_assigner.extract_player_crops(
                video_frames[frame_num],
                player_bboxes,
                player_ids,
                [1.0] * len(player_ids)
            )

            features         = team_assigner.extract_features(valid_ids, player_crops)
            reduced_features = team_assigner.reduce_dimensionality(features)
            labels           = team_assigner.assign_teams_by_track_id(
                valid_ids, reduced_features,
                reassign=(frame_num % 30 == 0)
            )

            # FIX: use valid_ids (not all player_ids) when writing back team labels
            for pid, label in zip(valid_ids, labels):
                tracks['players'][frame_num][pid]['team'] = int(label)

        team_assigner.save_team_assignments()

        player_assigner   = PlayerBallAssigner()
        team_ball_control = []

        for frame_num, player_track in enumerate(tracks['players']):
            ball_info = (tracks['ball'][frame_num]
                         if frame_num < len(tracks['ball']) else {})
            ball_bbox = (ball_info.get(1, {}).get("bbox", None)
                         if isinstance(ball_info, dict) else None)

            if not ball_bbox:
                last_team = team_ball_control[-1] if team_ball_control else "Unknown"
                team_ball_control.append(last_team)
                continue

            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1 and assigned_player in player_track:
                player_data = player_track[assigned_player]
                if 'team' not in player_data:
                    player_data['team'] = 0
                player_data['has_ball'] = True
                team_ball_control.append(player_data['team'])
            else:
                last_team = team_ball_control[-1] if team_ball_control else "Unknown"
                team_ball_control.append(last_team)

        team_ball_control   = np.array(team_ball_control)
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        output_video_frames = camera_estimator.draw_camera_movement(
            output_video_frames, camera_movement_per_frame)
        speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

        # Player Rating System
        try:
            rating_system = PlayerRatingSystem()
            rating_system.update_stats(tracks)
            ratings      = rating_system.get_all_ratings()
            rating_path  = os.path.join(output_videos_dir, f"{video_name}_ratings.json")
            with open(rating_path, "w") as f:
                json.dump(ratings, f, indent=2)
            print(f"Ratings saved: {rating_path}")
        except Exception as e:
            print(f"Rating error: {e}")
            traceback.print_exc()

        save_video(output_video_frames, output_video_path_avi)

        # FIX: use atomic temp-file conversion to avoid Windows file-lock on second run
        if shutil.which("ffmpeg"):
            success = convert_to_mp4(output_video_path_avi, output_video_path_mp4)
            if success:
                try:
                    os.remove(output_video_path_avi)
                except Exception:
                    pass
                print(f"Saved MP4: {output_video_path_mp4}")
                return output_video_path_mp4
            else:
                print("FFmpeg conversion failed, keeping AVI")
                return output_video_path_avi
        else:
            print("FFmpeg not found, keeping AVI")
            return output_video_path_avi

    except Exception as e:
        print(f"Error in processing: {e}")
        traceback.print_exc()
        return None


def generate_video_chunks(video_filename, start_byte=0, end_byte=None):
    with open(video_filename, "rb") as f:
        f.seek(start_byte)
        remaining = (end_byte - start_byte + 1) if end_byte else None
        while remaining is None or remaining > 0:
            chunk_size = min(CHUNK_SIZE, remaining) if remaining else CHUNK_SIZE
            chunk = f.read(chunk_size)
            if not chunk:
                break
            if remaining:
                remaining -= len(chunk)
            yield chunk


# API Endpoints
@app.get("/")
def root():
    return {"message": "GamePlan server running. Use /docs for API."}


@app.post("/upload-video/")
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    input_video_path = os.path.join(input_videos_dir, file.filename)
    async with aiofiles.open(input_video_path, "wb") as f:
        await f.write(await file.read())
    if background_tasks:
        background_tasks.add_task(process_video, input_video_path)
    return JSONResponse(
        content={
            "message":   "Video uploaded. Processing in background.",
            "video_url": f"/output/{file.filename}"
        },
        status_code=202
    )


@app.get("/output/{video_filename}")
async def stream_video(video_filename: str, request: Request):
    video_path_mp4 = os.path.join(
        output_videos_dir,
        f"{os.path.splitext(video_filename)[0]}.mp4")
    video_path_avi = os.path.join(
        output_videos_dir,
        f"{os.path.splitext(video_filename)[0]}.avi")
    video_path = (video_path_mp4
                  if os.path.exists(video_path_mp4)
                  else video_path_avi)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not processed yet.")
    file_size    = os.stat(video_path).st_size
    range_header = request.headers.get("Range")
    start, end   = 0, file_size - 1
    if range_header:
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0])
        end   = int(byte_range[1]) if byte_range[1] else file_size - 1
    headers = {
        "Content-Type":        "video/mp4",
        "Content-Length":      str(end - start + 1),
        "Content-Disposition": "inline",
        "Accept-Ranges":       "bytes",
        "Content-Range":       f"bytes {start}-{end}/{file_size}",
        "Vary":                "Range",
        "Cache-Control":       "no-cache, no-store, must-revalidate",
    }
    return StreamingResponse(
        content=generate_video_chunks(video_path, start, end),
        headers=headers,
        status_code=206 if range_header else 200
    )


@app.get("/download/{video_filename}")
async def download_video(video_filename: str):
    video_path_mp4 = os.path.join(
        output_videos_dir,
        f"{os.path.splitext(video_filename)[0]}.mp4")
    video_path_avi = os.path.join(
        output_videos_dir,
        f"{os.path.splitext(video_filename)[0]}.avi")
    video_path = (video_path_mp4
                  if os.path.exists(video_path_mp4)
                  else video_path_avi)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    async def iterfile():
        async with aiofiles.open(video_path, "rb") as f:
            while chunk := await f.read(10 * 1024 * 1024):
                yield chunk

    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Content-Disposition":
                f"attachment; filename={os.path.basename(video_path)}"
        }
    )


@app.get("/ratings/{video_filename}")
def get_ratings(video_filename: str):
    rating_path = os.path.join(
        output_videos_dir,
        f"{os.path.splitext(video_filename)[0]}_ratings.json"
    )
    if not os.path.exists(rating_path):
        raise HTTPException(
            status_code=404,
            detail="Ratings not found. Process video first."
        )
    with open(rating_path, "r") as f:
        return json.load(f)


# API Endpoints — Live Tracking
@app.post("/live/start")
def start_live(ip_url: str = "http://192.168.1.5:8080/video"):
    global stream_active
    if stream_active:
        return {"status": "already running", "ip_url": ip_url}
    stream_active = True
    thread = threading.Thread(target=run_tracking, args=(ip_url,), daemon=True)
    thread.start()
    logger.info(f"Live tracking started: {ip_url}")
    return {"status": "started", "ip_url": ip_url}


@app.post("/live/stop")
def stop_live():
    global stream_active
    stream_active = False
    logger.info("Live tracking stopped by user.")
    return {"status": "stopped"}


@app.get("/live/feed")
def live_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/live/status")
def live_status():
    return {"active": stream_active}


@app.get("/live/data")
def live_data():
    players  = [d for d in tracking_data if d["label"] == "player"]
    ball     = next((d for d in tracking_data if d["label"] == "ball"), None)
    referees = [d for d in tracking_data if d["label"] == "referee"]
    return {
        "active":        stream_active,
        "player_count":  len(players),
        "players":       players,
        "ball_detected": ball is not None,
        "ball":          ball,
        "ball_position": ball["center"] if ball else None,
        "ball_trail":    list(ball_trail),
        "referees":      referees,
        "total_objects": len(tracking_data)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)