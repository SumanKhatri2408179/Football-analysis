
import os
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, SiglipVisionModel
from trackers.tracker import Tracker
from team_assigner.team_classifier import TeamAssigner
from development_and_analysis.k_means_custom import CustomKMeans

def main():
    # Initialize components
    tracker = Tracker('models/best2.pt')  # Your model path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    team_assigner = TeamAssigner(device=device)  # Initialize with device parameter
    kmeans = CustomKMeans(n_clusters=2)  # For team classification

    video_path = "input_videos/sample1.mp4"  # Update with your video path
    
    # Create output directory if it doesn't exist
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output path based on input video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections and tracks using tracker
            tracks = tracker.update(frame)

            # Extract player crops and assign teams
            if 'players' in tracks:
                player_bboxes = [track_info["bbox"] for track_info in tracks['players'].values()]
                if player_bboxes:
                    # Extract and classify team
                    player_crops = team_assigner.extract_player_crops(frame, player_bboxes)
                    features = team_assigner.extract_features(list(tracks['players'].keys()), player_crops)
                    reduced_features = team_assigner.reduce_dimensionality(features)
                    team_labels = team_assigner.assign_teams(reduced_features)
            
            # Draw annotations on the frame
            annotated_frame = tracker.draw_annotations(frame, tracks, [])  # Empty list for team_ball_control for now
            
            # Write the frame to output video
            out.write(annotated_frame)
            
            # Optional: Display progress
            frame_count += 1
            if frame_count % 30 == 0:  # Show progress every 30 frames
                print(f"Processed {frame_count} frames")
                
    except Exception as e:
        print(f"Error processing video: {e}")
        
    finally:
        # Release resources
        cap.release()
        out.release()
        print(f"\nVideo processing complete. Output saved to: {output_path}")
        
    cap.release()

if __name__ == '__main__':
    main()