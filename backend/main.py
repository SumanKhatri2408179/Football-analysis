# import os
# import cv2
# import uvicorn
# import numpy as np
# import torch
# from fastapi import FastAPI, UploadFile, File, HTTPException, Request
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import  JSONResponse
# from fastapi import FastAPI
# from utils import read_video, save_video
# from trackers import Tracker
# from team_assigner import TeamAssigner
# from player_ball_assigner import PlayerBallAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator
# import logging
# import aiofiles
# import ffmpeg 

# app = FastAPI(debug=True)

# # CORS settings
# origins = [
#     "*",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Or specify the origins you want to allow
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Ensure directories exist
# input_videos_dir = "input_videos"
# output_videos_dir = "output_videos"

# #chunk size desc 1 mb
# CHUNK_SIZE = 1024 * 1024 

# #base Directory
# base_dir = os.getcwd()

# os.makedirs(input_videos_dir, exist_ok=True)
# os.makedirs(output_videos_dir, exist_ok=True)

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def process_video(input_video_path, output_video_path):
#     video_path = input_video_path
#     video_frames = read_video(video_path)

#     # Generate a unique stub path
#     video_name = os.path.splitext(os.path.basename(video_path))[0]  
#     stub_path = f'stubs/{video_name}_track_stubs.pkl'

#     # Initialize Tracker
#     tracker = Tracker('models/best1.pt')
    
#     # Get Tracks (validate or regenerate as needed)
#     tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_path)
    
#     # Get object positions 
#     tracker.add_position_to_tracks(tracks)
#     # camera movement estimator
#     camera_movement_estimator = CameraMovementEstimator(video_frames[0])
#     camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
#                                                                                 read_from_stub=True,
#                                                                                 stub_path='stubs/camera_movement_stub.pkl')
#     camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    
    
#     # View Trasnformer
#     frame_height, frame_width, _ = video_frames[0].shape
#     view_transformer = ViewTransformer((frame_height, frame_width))
#     view_transformer.add_transformed_position_to_tracks(tracks)


    
#     # Interpolate Ball Positions
#     tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    
#     # Speed and distance estimator
#     speed_and_distance_estimator = SpeedAndDistance_Estimator()
#     speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

#     # Assign Player Teams using TeamAssigner
#     team_assigner = TeamAssigner(device="cuda" if torch.cuda.is_available() else "cpu", video_path=video_path)

#     # Ensure saved team assignments are loaded (avoid recomputation)
#     team_assigner.load_team_assignments()

#     for frame_num, player_track in enumerate(tracks['players']):
#         player_ids = list(player_track.keys())
#         player_bboxes = [player_track[pid]["bbox"] for pid in player_ids]

#         # Extract and reduce features
#         player_crops = team_assigner.extract_player_crops(video_frames[frame_num], player_bboxes, [1.0] * len(player_ids))
#         features = team_assigner.extract_features(player_ids, player_crops)
#         reduced_features = team_assigner.reduce_dimensionality(features)

#         # Assign teams
#         labels = team_assigner.assign_teams_by_track_id(player_ids, reduced_features, reassign=(frame_num % 30 == 0))

#         for pid, label in zip(player_ids, labels):
#             tracks['players'][frame_num][pid]['team'] = label  # ✅ Assign team normally

#         # 🔹 Ensure every player has a valid 'team' entry
#         if 'team' not in tracks['players'][frame_num][pid]:
#             tracks['players'][frame_num][pid]['team'] = "Unknown"  # Default team assignment


#     # Save assigned teams for future runs
#     team_assigner.save_team_assignments()


#     # Assign Ball to Players
#     player_assigner = PlayerBallAssigner()
#     team_ball_control = []

#     for frame_num, player_track in enumerate(tracks['players']):
#         # ✅ Safely get ball information
#         ball_info = tracks['ball'][frame_num] if frame_num < len(tracks['ball']) else {}
#         ball_bbox = ball_info.get(1, {}).get("bbox", None) if isinstance(ball_info, dict) else None

#         if not ball_bbox:
#             print(f"⚠️ Frame {frame_num}: Ball not detected, using last known team control.")
#             last_team = team_ball_control[-1] if team_ball_control else "Unknown"
#             team_ball_control.append(last_team)
#             continue

#         # Assign the ball to the closest player
#         assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

#         if assigned_player != -1:
#             if assigned_player in tracks['players'][frame_num]:
#                 player_data = tracks['players'][frame_num][assigned_player]

#                 # ✅ Ensure a team is always assigned
#                 if 'team' not in player_data:
#                     print(f"⚠️ Frame {frame_num}: Assigned player {assigned_player} has no team! Assigning default team.")
#                     player_data['team'] = 0  # Default team to avoid UI errors

#                 # ✅ Assign ball possession
#                 player_data['has_ball'] = True
#                 team_ball_control.append(player_data['team'])
#                 print(f"✅ Frame {frame_num}: Player {assigned_player} has ball. Team: {player_data['team']}")
#             else:
#                 print(f"⚠️ Frame {frame_num}: Assigned player {assigned_player} not found in tracking data!")
#                 last_team = team_ball_control[-1] if team_ball_control else "Unknown"
#                 team_ball_control.append(last_team)
#         else:
#             # Maintain previous team possession if no assignment is found
#             last_team = team_ball_control[-1] if team_ball_control else "Unknown"
#             team_ball_control.append(last_team)

#     team_ball_control = np.array(team_ball_control)  # Convert to NumPy array



    
#     # Draw Annotations
#     output_video_frames = tracker.draw_annotations(video_frames, tracks,  team_ball_control)
    
#     ## Draw Camera movement
#     output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
#     ## Draw Speed and Distance
#     speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)



#     # Debugging Information
#     print(f"Input video frames: {len(video_frames)}")
#     print(f"Output video frames: {len(output_video_frames)}")

#     # Save Annotated Video
#     output_video_path_avi = f"output_videos/outputVideonew.avi"
#     save_video(output_video_frames,output_video_path_avi)
    
#     #output to fix browser format
#     output_video_path_mp4 = f"output_videos/{video_name}.mp4"
#     ffmpeg.input(output_video_path_avi).output(output_video_path_mp4,vcodec="libx264").run()
#     print(f"saved video on dir:", output_video_path_mp4)

# def generate_video_chunks(video_filename, start_byte=0, end_byte=None):
#     counter = 0  

#     with open(video_filename, "rb") as file_object:
#         file_object.seek(start_byte)
#         remaining_bytes = (end_byte - start_byte + 1) if end_byte else None

#         while remaining_bytes is None or remaining_bytes > 0:
#             chunk_size = min(CHUNK_SIZE, remaining_bytes) if remaining_bytes else CHUNK_SIZE
#             chunk = file_object.read(chunk_size)

#             if not chunk:
#                 print("End of file reached.")
#                 break  # Ensure we exit when file ends

#             if remaining_bytes:
#                 remaining_bytes -= len(chunk)

#             counter += 1
#             print(f"Sending chunk #{counter}, size: {len(chunk)} bytes")

#             yield chunk
        
#     print("✅ Finished sending all chunks.") 
    
    

# @app.post("/upload-video/")
# async def upload_video(file: UploadFile = File(...)):
#     try:
#         input_video_path = os.path.join(input_videos_dir, file.filename)
#         with open(input_video_path, "wb") as f:
#             f.write(await file.read())
#         print(f"Video uploaded to: {input_video_path}")

#         output_video_path = os.path.join(output_videos_dir, f"processed_{file.filename}")
#         output_video_url = input_video_path
        
#         try:
#             process_video(input_video_path, output_video_path)
#         except Exception as e:
#             print(f"Error in background processing: {e}")

#         return JSONResponse(
#             content={
#                 "message": "Video uploaded and processed successfully",
#                 "video_url": f"{os.path.basename(output_video_url)}"
#             },
#             status_code=202
#         )
#     except Exception as e:
#         print(f"Exception occurred: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

   
# @app.get("/output/{video_filename}")
# async def stream_video(video_filename: str, request: Request):
#     VIDEO_PATH = os.path.join(base_dir, "output_videos", video_filename)

#     if not os.path.exists(VIDEO_PATH):
#         raise HTTPException(status_code=404, detail="File not found")

#     file_size = os.stat(VIDEO_PATH).st_size
#     range_header = request.headers.get("Range")

#     if range_header:
#         byte_range = range_header.replace("bytes=", "").split("-")
#         start_byte = int(byte_range[0])
#         end_byte = int(byte_range[1]) if byte_range[1] else file_size - 1

#         if start_byte >= file_size or end_byte >= file_size:
#             raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

#         headers = {
#             "Content-Type": "video/mp4",
#             "Accept-Ranges": "bytes",
#             "Content-Range": f"bytes {start_byte}-{end_byte}/{file_size}",
#             "Content-Length": str(end_byte - start_byte + 1),
#             "Content-Disposition": "inline",
#             "Vary": "Range",
#             "Cache-Control": "no-cache, no-store, must-revalidate",
#         }

#         print(f"📡 Streaming partial content: {start_byte} - {end_byte}")

#         return StreamingResponse(
#             content=generate_video_chunks(VIDEO_PATH, start_byte, end_byte),
#             headers=headers,
#             status_code=206
#         )

#     headers = {
#         "Content-Type": "video/mp4",
#         "Accept-Ranges": "bytes",
#         "Content-Length": str(file_size),
#         "Content-Disposition": "inline",
#         "Vary": "Range",
#     }

#     print("📡 Streaming full video")

#     return StreamingResponse(
#         content=generate_video_chunks(VIDEO_PATH),
#         headers=headers,
#         status_code=200
#     )

    
# @app.get("/download/{video_filename}")
# async def download_video(video_filename: str):
#     video_path = os.path.join(output_videos_dir, video_filename)
#     CHUNK_SIZE= 10 * 1024 * 1024
    
#     if os.path.exists(video_path):
#         async def iterfile():
#             async with aiofiles.open(video_path, "rb") as file_like:
#                 while chunk := await file_like.read(CHUNK_SIZE):
#                     yield chunk
        
#         return StreamingResponse(iterfile(), 
#                                  media_type="video/avi", 
#                                  headers={"Content-Disposition": f"attachment; filename={video_filename}"})
#     else:
#         raise HTTPException(status_code=404, detail="Video not found")




# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000,timeout_keep_alive=300, )
# import os
# import cv2
# import uvicorn
# import numpy as np
# import torch
# from fastapi import FastAPI, UploadFile, File, HTTPException, Request
# from fastapi.responses import StreamingResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from utils import read_video, save_video
# from trackers import Tracker
# from team_assigner import TeamAssigner
# from player_ball_assigner import PlayerBallAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator
# import logging
# import aiofiles
# import ffmpeg

# # FastAPI app initialization
# app = FastAPI(debug=True)

# # CORS settings
# origins = ["*"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Directories for input/output videos
# input_videos_dir = "input_videos"
# output_videos_dir = "output_videos"
# os.makedirs(input_videos_dir, exist_ok=True)
# os.makedirs(output_videos_dir, exist_ok=True)

# # Chunk size for streaming (1 MB)
# CHUNK_SIZE = 1024 * 1024
# base_dir = os.getcwd()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --------------------------
# # Helper Functions
# # --------------------------

# def process_video(input_video_path, output_video_path):
#     """Processes the video and saves the output annotated video."""
#     video_frames = read_video(input_video_path)

#     video_name = os.path.splitext(os.path.basename(input_video_path))[0]
#     stub_path = f'stubs/{video_name}_track_stubs.pkl'

#     # Initialize Tracker
#     tracker = Tracker('models/best1.pt')
#     tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_path)
#     tracker.add_position_to_tracks(tracks)

#     # Camera movement estimation
#     camera_estimator = CameraMovementEstimator(video_frames[0])
#     camera_movement_per_frame = camera_estimator.get_camera_movement(
#         video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl'
#     )
#     camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

#     # View transformation
#     frame_height, frame_width, _ = video_frames[0].shape
#     view_transformer = ViewTransformer((frame_height, frame_width))
#     view_transformer.add_transformed_position_to_tracks(tracks)

#     # Interpolate ball positions
#     tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

#     # Speed and distance estimation
#     speed_distance_estimator = SpeedAndDistance_Estimator()
#     speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

#     # Assign teams
#     team_assigner = TeamAssigner(device="cuda" if torch.cuda.is_available() else "cpu", video_path=input_video_path)
#     team_assigner.load_team_assignments()
#     for frame_num, player_track in enumerate(tracks['players']):
#         player_ids = list(player_track.keys())
#         player_bboxes = [player_track[pid]["bbox"] for pid in player_ids]
#         player_crops = team_assigner.extract_player_crops(video_frames[frame_num], player_bboxes, [1.0] * len(player_ids))
#         features = team_assigner.extract_features(player_ids, player_crops)
#         reduced_features = team_assigner.reduce_dimensionality(features)
#         labels = team_assigner.assign_teams_by_track_id(player_ids, reduced_features, reassign=(frame_num % 30 == 0))
#         for pid, label in zip(player_ids, labels):
#             tracks['players'][frame_num][pid]['team'] = label
#             if 'team' not in tracks['players'][frame_num][pid]:
#                 tracks['players'][frame_num][pid]['team'] = "Unknown"
#     team_assigner.save_team_assignments()

#     # Assign ball to players
#     player_assigner = PlayerBallAssigner()
#     team_ball_control = []
#     for frame_num, player_track in enumerate(tracks['players']):
#         ball_info = tracks['ball'][frame_num] if frame_num < len(tracks['ball']) else {}
#         ball_bbox = ball_info.get(1, {}).get("bbox", None) if isinstance(ball_info, dict) else None
#         if not ball_bbox:
#             last_team = team_ball_control[-1] if team_ball_control else "Unknown"
#             team_ball_control.append(last_team)
#             continue
#         assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
#         if assigned_player != -1 and assigned_player in player_track:
#             player_data = player_track[assigned_player]
#             if 'team' not in player_data:
#                 player_data['team'] = 0
#             player_data['has_ball'] = True
#             team_ball_control.append(player_data['team'])
#         else:
#             last_team = team_ball_control[-1] if team_ball_control else "Unknown"
#             team_ball_control.append(last_team)
#     team_ball_control = np.array(team_ball_control)

#     # Draw annotations
#     output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
#     output_video_frames = camera_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
#     speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

#     # Save video
#     output_video_path_avi = os.path.join(output_videos_dir, f"{video_name}.avi")
#     save_video(output_video_frames, output_video_path_avi)
#     output_video_path_mp4 = os.path.join(output_videos_dir, f"{video_name}.mp4")
#     ffmpeg.input(output_video_path_avi).output(output_video_path_mp4, vcodec="libx264").run()
#     print(f"Saved video to: {output_video_path_mp4}")


# def generate_video_chunks(video_filename, start_byte=0, end_byte=None):
#     """Yields video chunks for streaming."""
#     with open(video_filename, "rb") as f:
#         f.seek(start_byte)
#         remaining = (end_byte - start_byte + 1) if end_byte else None
#         while remaining is None or remaining > 0:
#             chunk_size = min(CHUNK_SIZE, remaining) if remaining else CHUNK_SIZE
#             chunk = f.read(chunk_size)
#             if not chunk:
#                 break
#             if remaining:
#                 remaining -= len(chunk)
#             yield chunk


# # --------------------------
# # API Endpoints
# # --------------------------

# @app.get("/")
# def root():
#     return {"message": "Server is running! Visit /docs for API documentation."}


# @app.post("/upload-video/")
# async def upload_video(file: UploadFile = File(...)):
#     try:
#         input_video_path = os.path.join(input_videos_dir, file.filename)
#         async with aiofiles.open(input_video_path, "wb") as f:
#             await f.write(await file.read())
#         print(f"Video uploaded to: {input_video_path}")

#         output_video_path = os.path.join(output_videos_dir, f"processed_{file.filename}")
#         try:
#             process_video(input_video_path, output_video_path)
#         except Exception as e:
#             print(f"Error in video processing: {e}")

#         return JSONResponse(
#             content={
#                 "message": "Video uploaded and processed successfully",
#                 "video_url": f"/output/{os.path.basename(output_video_path)}"
#             },
#             status_code=202
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/output/{video_filename}")
# async def stream_video(video_filename: str, request: Request):
#     video_path = os.path.join(output_videos_dir, video_filename)
#     if not os.path.exists(video_path):
#         raise HTTPException(status_code=404, detail="File not found")

#     file_size = os.stat(video_path).st_size
#     range_header = request.headers.get("Range")
#     start, end = 0, file_size - 1

#     if range_header:
#         byte_range = range_header.replace("bytes=", "").split("-")
#         start = int(byte_range[0])
#         end = int(byte_range[1]) if byte_range[1] else file_size - 1

#     headers = {
#         "Content-Type": "video/mp4",
#         "Content-Length": str(end - start + 1),
#         "Content-Disposition": "inline",
#         "Accept-Ranges": "bytes",
#         "Content-Range": f"bytes {start}-{end}/{file_size}",
#         "Vary": "Range",
#         "Cache-Control": "no-cache, no-store, must-revalidate",
#     }

#     return StreamingResponse(
#         content=generate_video_chunks(video_path, start, end),
#         headers=headers,
#         status_code=206 if range_header else 200
#     )


# @app.get("/download/{video_filename}")
# async def download_video(video_filename: str):
#     video_path = os.path.join(output_videos_dir, video_filename)
#     if not os.path.exists(video_path):
#         raise HTTPException(status_code=404, detail="Video not found")

#     async def iterfile():
#         async with aiofiles.open(video_path, "rb") as f:
#             while chunk := await f.read(10 * 1024 * 1024):
#                 yield chunk

#     return StreamingResponse(
#         iterfile(),
#         media_type="video/mp4",
#         headers={"Content-Disposition": f"attachment; filename={video_filename}"}
#     )


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)
# import os
# import cv2
# import uvicorn
# import numpy as np
# import torch
# from fastapi import FastAPI, UploadFile, File, HTTPException, Request
# from fastapi.responses import StreamingResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from utils import read_video, save_video
# from trackers import Tracker
# from team_assigner import TeamAssigner
# from player_ball_assigner import PlayerBallAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator
# import logging
# import aiofiles
# import ffmpeg
# import traceback
# import shutil

# # FastAPI app initialization
# app = FastAPI(debug=True)

# # CORS settings
# origins = ["*"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Directories for input/output videos
# input_videos_dir = "input_videos"
# output_videos_dir = "output_videos"
# stubs_dir = "stubs"
# os.makedirs(input_videos_dir, exist_ok=True)
# os.makedirs(output_videos_dir, exist_ok=True)
# os.makedirs(stubs_dir, exist_ok=True)

# # Chunk size for streaming (1 MB)
# CHUNK_SIZE = 1024 * 1024
# base_dir = os.getcwd()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --------------------------
# # Helper Functions
# # --------------------------

# def process_video(input_video_path, output_video_path=None):
#     """Processes the video and saves the output annotated video."""
#     try:
#         print(f"Starting video processing for: {input_video_path}")
#         video_frames = read_video(input_video_path)
#         print(f"Read {len(video_frames)} frames from video")

#         video_name = os.path.splitext(os.path.basename(input_video_path))[0]
#         stub_path = f'stubs/{video_name}_track_stubs.pkl'

#         # Initialize Tracker
#         tracker = Tracker('models/best1.pt')
#         tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_path)
#         tracker.add_position_to_tracks(tracks)

#         # Camera movement estimation
#         camera_estimator = CameraMovementEstimator(video_frames[0])
#         camera_movement_per_frame = camera_estimator.get_camera_movement(
#             video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl'
#         )
#         camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

#         # View transformation
#         frame_height, frame_width, _ = video_frames[0].shape
#         view_transformer = ViewTransformer((frame_height, frame_width))
#         view_transformer.add_transformed_position_to_tracks(tracks)

#         # Interpolate ball positions
#         tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

#         # Speed and distance estimation
#         speed_distance_estimator = SpeedAndDistance_Estimator()
#         speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

#         # Assign teams
#         team_assigner = TeamAssigner(device="cuda" if torch.cuda.is_available() else "cpu", video_path=input_video_path)
#         team_assigner.load_team_assignments()
#         for frame_num, player_track in enumerate(tracks['players']):
#             player_ids = list(player_track.keys())
#             player_bboxes = [player_track[pid]["bbox"] for pid in player_ids]
#             player_crops = team_assigner.extract_player_crops(video_frames[frame_num], player_bboxes, [1.0] * len(player_ids))
#             features = team_assigner.extract_features(player_ids, player_crops)
#             reduced_features = team_assigner.reduce_dimensionality(features)
#             labels = team_assigner.assign_teams_by_track_id(player_ids, reduced_features, reassign=(frame_num % 30 == 0))
#             for pid, label in zip(player_ids, labels):
#                 tracks['players'][frame_num][pid]['team'] = label
#                 if 'team' not in tracks['players'][frame_num][pid]:
#                     tracks['players'][frame_num][pid]['team'] = "Unknown"
#         team_assigner.save_team_assignments()

#         # Assign ball to players
#         player_assigner = PlayerBallAssigner()
#         team_ball_control = []
#         for frame_num, player_track in enumerate(tracks['players']):
#             ball_info = tracks['ball'][frame_num] if frame_num < len(tracks['ball']) else {}
#             ball_bbox = ball_info.get(1, {}).get("bbox", None) if isinstance(ball_info, dict) else None
#             if not ball_bbox:
#                 last_team = team_ball_control[-1] if team_ball_control else "Unknown"
#                 team_ball_control.append(last_team)
#                 continue
#             assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
#             if assigned_player != -1 and assigned_player in player_track:
#                 player_data = player_track[assigned_player]
#                 if 'team' not in player_data:
#                     player_data['team'] = 0
#                 player_data['has_ball'] = True
#                 team_ball_control.append(player_data['team'])
#             else:
#                 last_team = team_ball_control[-1] if team_ball_control else "Unknown"
#                 team_ball_control.append(last_team)
#         team_ball_control = np.array(team_ball_control)

#         # Draw annotations
#         print("Drawing annotations on video frames...")
#         output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
#         output_video_frames = camera_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
#         speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

#         # Save video
#         print("Saving processed video...")
#         video_name = os.path.splitext(os.path.basename(input_video_path))[0]
#         output_video_path_avi = os.path.join(output_videos_dir, f"{video_name}.avi")
#         save_video(output_video_frames, output_video_path_avi)
#         print(f"✅ Saved AVI to: {output_video_path_avi}")
        
#         # Try converting to MP4 with FFmpeg
#         output_video_path_mp4 = os.path.join(output_videos_dir, f"{video_name}.mp4")
        
#         try:
#             # Check if FFmpeg is available
#             if shutil.which("ffmpeg") is None:
#                 raise FileNotFoundError("FFmpeg not found in PATH")
                
#             print(f"Converting to MP4: {output_video_path_mp4}")
#             (
#                 ffmpeg
#                 .input(output_video_path_avi)
#                 .output(output_video_path_mp4, vcodec="libx264", acodec="aac")
#                 .overwrite_output()
#                 .run(capture_stdout=True, capture_stderr=True)
#             )
#             print(f"✅ Successfully saved MP4 to: {output_video_path_mp4}")
            
#             # Delete AVI file
#             try:
#                 os.remove(output_video_path_avi)
#                 print(f"Deleted temporary AVI file")
#             except:
#                 pass
                
#             return output_video_path_mp4
            
#         except (FileNotFoundError, ffmpeg.Error) as e:
#             print(f"⚠️ FFmpeg not available or conversion failed: {e}")
#             print(f"💡 Serving AVI file instead: {output_video_path_avi}")
#             print(f"📥 Install FFmpeg from: https://www.gyan.dev/ffmpeg/builds/")
            
#             # Just return the AVI file - browsers can play it
#             return output_video_path_avi
            
#     except Exception as e:
#         print(f"❌ Error in video processing: {e}")
#         traceback.print_exc()
#         return None


# def generate_video_chunks(video_filename, start_byte=0, end_byte=None):
#     """Yields video chunks for streaming."""
#     with open(video_filename, "rb") as f:
#         f.seek(start_byte)
#         remaining = (end_byte - start_byte + 1) if end_byte else None
#         while remaining is None or remaining > 0:
#             chunk_size = min(CHUNK_SIZE, remaining) if remaining else CHUNK_SIZE
#             chunk = f.read(chunk_size)
#             if not chunk:
#                 break
#             if remaining:
#                 remaining -= len(chunk)
#             yield chunk


# # --------------------------
# # API Endpoints
# # --------------------------

# @app.get("/")
# def root():
#     return {"message": "Server is running! Visit /docs for API documentation."}


# @app.post("/upload-video/")
# async def upload_video(file: UploadFile = File(...)):
#     try:
#         # Save uploaded file
#         input_video_path = os.path.join(input_videos_dir, file.filename)
#         async with aiofiles.open(input_video_path, "wb") as f:
#             await f.write(await file.read())
#         print(f"✅ Video uploaded to: {input_video_path}")

#         # Get base filename without extension
#         video_name = os.path.splitext(file.filename)[0]
        
#         # Process video
#         print(f"📹 Starting video processing...")
#         result_path = process_video(input_video_path, None)
        
#         if result_path is None or not os.path.exists(result_path):
#             print(f"❌ Video processing failed - no output file created")
#             raise HTTPException(
#                 status_code=500, 
#                 detail="Video processing failed. Check backend logs for details."
#             )
        
#         # Get the output filename
#         output_filename = os.path.basename(result_path)
#         print(f"✅ Processing complete! Output file: {output_filename}")
        
#         return JSONResponse(
#             content={
#                 "message": "Video uploaded and processed successfully",
#                 "video_url": f"/output/{output_filename}"
#             },
#             status_code=200
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"❌ Upload error: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/output/{video_filename}")
# async def stream_video(video_filename: str, request: Request):
#     video_path = os.path.join(output_videos_dir, video_filename)
#     print(f"Attempting to stream video: {video_path}")
    
#     if not os.path.exists(video_path):
#         print(f"❌ Video file not found: {video_path}")
#         print(f"Files in output directory: {os.listdir(output_videos_dir)}")
#         raise HTTPException(status_code=404, detail=f"Video file not found: {video_filename}")

#     file_size = os.stat(video_path).st_size
#     range_header = request.headers.get("Range")
#     start, end = 0, file_size - 1

#     if range_header:
#         byte_range = range_header.replace("bytes=", "").split("-")
#         start = int(byte_range[0])
#         end = int(byte_range[1]) if byte_range[1] else file_size - 1

#     # Detect file type
#     file_ext = os.path.splitext(video_filename)[1].lower()
#     content_type = "video/mp4" if file_ext == ".mp4" else "video/x-msvideo" if file_ext == ".avi" else "video/mp4"

#     headers = {
#         "Content-Type": content_type,
#         "Content-Length": str(end - start + 1),
#         "Content-Disposition": "inline",
#         "Accept-Ranges": "bytes",
#         "Content-Range": f"bytes {start}-{end}/{file_size}",
#         "Vary": "Range",
#         "Cache-Control": "no-cache, no-store, must-revalidate",
#     }

#     print(f"✅ Streaming video: {video_filename} (bytes {start}-{end}/{file_size})")

#     return StreamingResponse(
#         content=generate_video_chunks(video_path, start, end),
#         headers=headers,
#         status_code=206 if range_header else 200
#     )


# @app.get("/download/{video_filename}")
# async def download_video(video_filename: str):
#     video_path = os.path.join(output_videos_dir, video_filename)
    
#     if not os.path.exists(video_path):
#         print(f"❌ Download failed - file not found: {video_path}")
#         raise HTTPException(status_code=404, detail="Video not found")

#     async def iterfile():
#         async with aiofiles.open(video_path, "rb") as f:
#             while chunk := await f.read(10 * 1024 * 1024):
#                 yield chunk

#     # Detect file type
#     file_ext = os.path.splitext(video_filename)[1].lower()
#     media_type = "video/mp4" if file_ext == ".mp4" else "video/x-msvideo" if file_ext == ".avi" else "video/mp4"

#     print(f"✅ Starting download: {video_filename}")
#     return StreamingResponse(
#         iterfile(),
#         media_type=media_type,
#         headers={"Content-Disposition": f"attachment; filename={video_filename}"}
#     )


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)
# import os
# import cv2
# import uvicorn
# import numpy as np
# import torch
# from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
# from fastapi.responses import StreamingResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from utils import read_video, save_video
# from trackers import Tracker
# from team_assigner import TeamAssigner
# from player_ball_assigner import PlayerBallAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator
# import logging
# import aiofiles
# import ffmpeg
# import traceback
# import shutil
# import time

# # --------------------------
# # FastAPI App
# # --------------------------
# app = FastAPI(debug=True)

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Directories
# input_videos_dir = "input_videos"
# output_videos_dir = "output_videos"
# stubs_dir = "stubs"
# os.makedirs(input_videos_dir, exist_ok=True)
# os.makedirs(output_videos_dir, exist_ok=True)
# os.makedirs(stubs_dir, exist_ok=True)

# # Chunk size for streaming
# CHUNK_SIZE = 1024 * 1024

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --------------------------
# # Video Processing
# # --------------------------

# def process_video(input_video_path):
#     """Processes the video and saves annotated output."""
#     try:
#         video_name = os.path.splitext(os.path.basename(input_video_path))[0]
#         output_video_path_avi = os.path.join(output_videos_dir, f"{video_name}.avi")
#         output_video_path_mp4 = os.path.join(output_videos_dir, f"{video_name}.mp4")
#         stub_path = f'stubs/{video_name}_track_stubs.pkl'

#         print(f"Processing video: {input_video_path}")
#         video_frames = read_video(input_video_path)

#         # Tracker
#         tracker = Tracker('models/best1.pt')
#         tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_path)
#         tracker.add_position_to_tracks(tracks)

#         # Camera
#         camera_estimator = CameraMovementEstimator(video_frames[0])
#         camera_movement_per_frame = camera_estimator.get_camera_movement(
#             video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl'
#         )
#         camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

#         # View transformation
#         frame_height, frame_width, _ = video_frames[0].shape
#         view_transformer = ViewTransformer((frame_height, frame_width))
#         view_transformer.add_transformed_position_to_tracks(tracks)

#         # Ball interpolation
#         tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

#         # Speed & distance
#         speed_distance_estimator = SpeedAndDistance_Estimator()
#         speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

#         # Teams
#         team_assigner = TeamAssigner(device="cuda" if torch.cuda.is_available() else "cpu", video_path=input_video_path)
#         team_assigner.load_team_assignments()
#         for frame_num, player_track in enumerate(tracks['players']):
#             player_ids = list(player_track.keys())
#             player_bboxes = [player_track[pid]["bbox"] for pid in player_ids]
#             player_crops = team_assigner.extract_player_crops(video_frames[frame_num], player_bboxes, [1.0]*len(player_ids))
#             features = team_assigner.extract_features(player_ids, player_crops)
#             reduced_features = team_assigner.reduce_dimensionality(features)
#             labels = team_assigner.assign_teams_by_track_id(player_ids, reduced_features, reassign=(frame_num % 30 == 0))
#             for pid, label in zip(player_ids, labels):
#                 tracks['players'][frame_num][pid]['team'] = label
#                 if 'team' not in tracks['players'][frame_num][pid]:
#                     tracks['players'][frame_num][pid]['team'] = "Unknown"
#         team_assigner.save_team_assignments()

#         # Ball assignment
#         player_assigner = PlayerBallAssigner()
#         team_ball_control = []
#         for frame_num, player_track in enumerate(tracks['players']):
#             ball_info = tracks['ball'][frame_num] if frame_num < len(tracks['ball']) else {}
#             ball_bbox = ball_info.get(1, {}).get("bbox", None) if isinstance(ball_info, dict) else None
#             if not ball_bbox:
#                 last_team = team_ball_control[-1] if team_ball_control else "Unknown"
#                 team_ball_control.append(last_team)
#                 continue
#             assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
#             if assigned_player != -1 and assigned_player in player_track:
#                 player_data = player_track[assigned_player]
#                 if 'team' not in player_data:
#                     player_data['team'] = 0
#                 player_data['has_ball'] = True
#                 team_ball_control.append(player_data['team'])
#             else:
#                 last_team = team_ball_control[-1] if team_ball_control else "Unknown"
#                 team_ball_control.append(last_team)
#         team_ball_control = np.array(team_ball_control)

#         # Draw
#         output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
#         output_video_frames = camera_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
#         speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

#         # Save AVI
#         save_video(output_video_frames, output_video_path_avi)

#         # Convert to MP4
#         if shutil.which("ffmpeg"):
#             try:
#                 ffmpeg.input(output_video_path_avi).output(output_video_path_mp4, vcodec="libx264", acodec="aac").overwrite_output().run()
#                 os.remove(output_video_path_avi)
#                 print(f"Saved MP4: {output_video_path_mp4}")
#             except:
#                 print(f"FFmpeg failed, keeping AVI")
#                 return output_video_path_avi
#             return output_video_path_mp4
#         else:
#             print("FFmpeg not found, keeping AVI")
#             return output_video_path_avi

#     except Exception as e:
#         print(f"❌ Error in processing: {e}")
#         traceback.print_exc()
#         return None

# def generate_video_chunks(video_filename, start_byte=0, end_byte=None):
#     with open(video_filename, "rb") as f:
#         f.seek(start_byte)
#         remaining = (end_byte - start_byte + 1) if end_byte else None
#         while remaining is None or remaining > 0:
#             chunk_size = min(CHUNK_SIZE, remaining) if remaining else CHUNK_SIZE
#             chunk = f.read(chunk_size)
#             if not chunk:
#                 break
#             if remaining:
#                 remaining -= len(chunk)
#             yield chunk

# # --------------------------
# # API Endpoints
# # --------------------------

# @app.get("/")
# def root():
#     return {"message": "Server running. Use /docs for API."}

# @app.post("/upload-video/")
# async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
#     input_video_path = os.path.join(input_videos_dir, file.filename)
#     output_placeholder_path = os.path.join(output_videos_dir, f"{file.filename}.processing")

#     async with aiofiles.open(input_video_path, "wb") as f:
#         await f.write(await file.read())

#     # Start background processing
#     if background_tasks:
#         background_tasks.add_task(process_video, input_video_path)

#     # Immediately respond to frontend
#     return JSONResponse(
#         content={
#             "message": "Video uploaded. Processing in background.",
#             "video_url": f"/output/{file.filename}"  # frontend can poll later
#         },
#         status_code=202
#     )

# @app.get("/output/{video_filename}")
# async def stream_video(video_filename: str, request: Request):
#     video_path_mp4 = os.path.join(output_videos_dir, f"{os.path.splitext(video_filename)[0]}.mp4")
#     video_path_avi = os.path.join(output_videos_dir, f"{os.path.splitext(video_filename)[0]}.avi")
#     video_path = video_path_mp4 if os.path.exists(video_path_mp4) else video_path_avi

#     if not os.path.exists(video_path):
#         raise HTTPException(status_code=404, detail="Video not processed yet.")

#     file_size = os.stat(video_path).st_size
#     range_header = request.headers.get("Range")
#     start, end = 0, file_size - 1
#     if range_header:
#         byte_range = range_header.replace("bytes=", "").split("-")
#         start = int(byte_range[0])
#         end = int(byte_range[1]) if byte_range[1] else file_size - 1

#     headers = {
#         "Content-Type": "video/mp4",
#         "Content-Length": str(end - start + 1),
#         "Content-Disposition": "inline",
#         "Accept-Ranges": "bytes",
#         "Content-Range": f"bytes {start}-{end}/{file_size}",
#         "Vary": "Range",
#         "Cache-Control": "no-cache, no-store, must-revalidate",
#     }

#     return StreamingResponse(
#         content=generate_video_chunks(video_path, start, end),
#         headers=headers,
#         status_code=206 if range_header else 200
#     )

# # --------------------------
# # Download endpoint
# # --------------------------

# @app.get("/download/{video_filename}")
# async def download_video(video_filename: str):
#     video_path_mp4 = os.path.join(output_videos_dir, f"{os.path.splitext(video_filename)[0]}.mp4")
#     video_path_avi = os.path.join(output_videos_dir, f"{os.path.splitext(video_filename)[0]}.avi")
#     video_path = video_path_mp4 if os.path.exists(video_path_mp4) else video_path_avi

#     if not os.path.exists(video_path):
#         raise HTTPException(status_code=404, detail="Video not found")

#     async def iterfile():
#         async with aiofiles.open(video_path, "rb") as f:
#             while chunk := await f.read(10*1024*1024):
#                 yield chunk

#     return StreamingResponse(
#         iterfile(),
#         media_type="video/mp4",
#         headers={"Content-Disposition": f"attachment; filename={os.path.basename(video_path)}"}
#     )


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)
import os
import cv2
import uvicorn
import numpy as np
import torch
import threading
import time
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
import ffmpeg
import traceback
import shutil

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(debug=True)

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

# --------------------------
# Live Tracking — Global State
# --------------------------
live_model     = YOLO("models/best1.pt")   # reuse your existing model
stream_active  = False
latest_frame   = None
frame_lock     = threading.Lock()
tracking_data  = []
ball_trail     = deque(maxlen=25)          # stores last 25 ball positions

# Colours (BGR)
COLOR_PLAYER  = (0,   255,   0)   # Green
COLOR_BALL    = (0,   255, 255)   # Yellow
COLOR_REFEREE = (255, 165,   0)   # Orange
COLOR_TRAIL   = (0,   200, 255)   # Light yellow trail


# --------------------------
# Live Tracking — Draw Helpers
# --------------------------

def draw_ball_trail(frame):
    """Draw a fading trail behind the ball showing its recent path."""
    for i in range(1, len(ball_trail)):
        if ball_trail[i - 1] is None or ball_trail[i] is None:
            continue
        thickness = max(1, int(np.sqrt(25 / float(i + 1)) * 2))
        cv2.line(frame, ball_trail[i - 1], ball_trail[i], COLOR_TRAIL, thickness)


def draw_player(frame, x1, y1, x2, y2, conf):
    """Draw player bounding box with confidence label."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PLAYER, 2)
    label = f"Player {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLOR_PLAYER, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 3, COLOR_PLAYER, -1)


def draw_ball(frame, x1, y1, x2, y2, conf):
    """Draw ball with circle, crosshair and trail."""
    cx     = (x1 + x2) // 2
    cy     = (y1 + y2) // 2
    radius = max((x2 - x1), (y2 - y1)) // 2 + 4

    # Add position to trail
    ball_trail.append((cx, cy))

    # Draw trail first (behind the ball)
    draw_ball_trail(frame)

    # Outer glow
    cv2.circle(frame, (cx, cy), radius + 4, COLOR_BALL, 1)
    # Main circle
    cv2.circle(frame, (cx, cy), radius, COLOR_BALL, 3)
    # Crosshair
    cv2.line(frame, (cx - radius - 8, cy), (cx + radius + 8, cy), COLOR_BALL, 1)
    cv2.line(frame, (cx, cy - radius - 8), (cx, cy + radius + 8), COLOR_BALL, 1)

    # Label above ball
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
    """Draw referee bounding box."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_REFEREE, 2)
    label = f"Referee {conf:.0%}"
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_REFEREE, 2)


# --------------------------
# Live Tracking — Core Loop
# --------------------------

def run_tracking(ip_url: str):
    """Runs in a background thread. Reads frames from IPWebcam,
       runs YOLOv8, draws annotations, stores latest frame."""
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

            # Keep trail visible for 10 frames after ball disappears
            if not ball_found:
                ball_missing += 1
                if ball_missing <= 10:
                    draw_ball_trail(frame)
                else:
                    ball_trail.clear()

            # ── Stats overlay (top-left dark strip) ───────────────────
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

            # ── LIVE badge (top-right) ─────────────────────────────────
            cv2.circle(frame, (w - 30, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, "LIVE",
                        (w - 72, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255), 2)

            # ── Frame counter (bottom-left) ────────────────────────────
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


# --------------------------
# Live Tracking — MJPEG Generator
# --------------------------

def generate_frames():
    """Yields MJPEG frames for the /live/feed endpoint."""
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
        time.sleep(0.033)  # ~30 FPS


# --------------------------
# Video Processing (existing)
# --------------------------

def process_video(input_video_path):
    """Processes the uploaded video and saves annotated output."""
    try:
        video_name            = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path_avi = os.path.join(output_videos_dir, f"{video_name}.avi")
        output_video_path_mp4 = os.path.join(output_videos_dir, f"{video_name}.mp4")
        stub_path             = f'stubs/{video_name}_track_stubs.pkl'

        print(f"Processing video: {input_video_path}")
        video_frames = read_video(input_video_path)

        # Tracker
        tracker = Tracker('models/best1.pt')
        tracks  = tracker.get_object_tracks(
            video_frames, read_from_stub=True, stub_path=stub_path)
        tracker.add_position_to_tracks(tracks)

        # Camera
        camera_estimator          = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_estimator.get_camera_movement(
            video_frames, read_from_stub=True,
            stub_path='stubs/camera_movement_stub.pkl')
        camera_estimator.add_adjust_positions_to_tracks(
            tracks, camera_movement_per_frame)

        # View transformation
        frame_height, frame_width, _ = video_frames[0].shape
        view_transformer = ViewTransformer((frame_height, frame_width))
        view_transformer.add_transformed_position_to_tracks(tracks)

        # Ball interpolation
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # Speed & distance
        speed_distance_estimator = SpeedAndDistance_Estimator()
        speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        # Teams
        team_assigner = TeamAssigner(
            device="cuda" if torch.cuda.is_available() else "cpu",
            video_path=input_video_path)
        team_assigner.load_team_assignments()

        for frame_num, player_track in enumerate(tracks['players']):
            player_ids   = list(player_track.keys())
            player_bboxes = [player_track[pid]["bbox"] for pid in player_ids]
            player_crops  = team_assigner.extract_player_crops(
                video_frames[frame_num], player_bboxes,
                [1.0] * len(player_ids))
            features         = team_assigner.extract_features(player_ids, player_crops)
            reduced_features = team_assigner.reduce_dimensionality(features)
            labels           = team_assigner.assign_teams_by_track_id(
                player_ids, reduced_features,
                reassign=(frame_num % 30 == 0))
            for pid, label in zip(player_ids, labels):
                tracks['players'][frame_num][pid]['team'] = label
                if 'team' not in tracks['players'][frame_num][pid]:
                    tracks['players'][frame_num][pid]['team'] = "Unknown"

        team_assigner.save_team_assignments()

        # Ball assignment
        player_assigner  = PlayerBallAssigner()
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

            assigned_player = player_assigner.assign_ball_to_player(
                player_track, ball_bbox)

            if assigned_player != -1 and assigned_player in player_track:
                player_data = player_track[assigned_player]
                if 'team' not in player_data:
                    player_data['team'] = 0
                player_data['has_ball'] = True
                team_ball_control.append(player_data['team'])
            else:
                last_team = team_ball_control[-1] if team_ball_control else "Unknown"
                team_ball_control.append(last_team)

        team_ball_control = np.array(team_ball_control)

        # Draw annotations
        output_video_frames = tracker.draw_annotations(
            video_frames, tracks, team_ball_control)
        output_video_frames = camera_estimator.draw_camera_movement(
            output_video_frames, camera_movement_per_frame)
        speed_distance_estimator.draw_speed_and_distance(
            output_video_frames, tracks)

        # Save AVI
        save_video(output_video_frames, output_video_path_avi)

        # Convert to MP4 if ffmpeg available
        if shutil.which("ffmpeg"):
            try:
                ffmpeg.input(output_video_path_avi).output(
                    output_video_path_mp4,
                    vcodec="libx264", acodec="aac"
                ).overwrite_output().run()
                os.remove(output_video_path_avi)
                print(f"Saved MP4: {output_video_path_mp4}")
                return output_video_path_mp4
            except Exception:
                print("FFmpeg failed, keeping AVI")
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


# ==========================
# API Endpoints — Existing
# ==========================

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


# ==========================
# API Endpoints — Live Tracking (NEW)
# ==========================

@app.post("/live/start")
def start_live(ip_url: str = "http://192.168.1.5:8080/video"):
    """Start live player and ball tracking from IPWebcam stream."""
    global stream_active
    if stream_active:
        return {"status": "already running", "ip_url": ip_url}

    stream_active = True
    thread = threading.Thread(
        target=run_tracking,
        args=(ip_url,),
        daemon=True
    )
    thread.start()
    logger.info(f"Live tracking started: {ip_url}")
    return {"status": "started", "ip_url": ip_url}


@app.post("/live/stop")
def stop_live():
    """Stop the live tracking stream."""
    global stream_active
    stream_active = False
    logger.info("Live tracking stopped by user.")
    return {"status": "stopped"}


@app.get("/live/feed")
def live_feed():
    """MJPEG stream of the annotated live video feed."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/live/status")
def live_status():
    """Check whether the live stream is currently active."""
    return {"active": stream_active}


@app.get("/live/data")
def live_data():
    """Return current frame tracking data: players, ball, referee."""
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


# ==========================
# Run
# ==========================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)

