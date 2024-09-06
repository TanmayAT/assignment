import os
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')  # Pretrained YOLOv8 model

# Initialize the tracker
tracker = DeepSort()  # Initialize DeepSORT tracker

# Define paths
input_folder = r'C:\Users\Vaidik\Desktop\test\videos'
output_folder = r'C:\Users\Vaidik\Desktop\test\output_videos'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each video in the input folder
for video_file in os.listdir(input_folder):
    video_path = os.path.join(input_folder, video_file)
    
    # Check if the file is a video
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print(f"Skipping non-video file: {video_file}")
        continue

    print(f"Processing video: {video_file}")

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        continue

    # Define output video path
    output_video_path = os.path.join(output_folder, f'output_{video_file}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Loop through the video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video: {video_file}")
            break  # End of video

        # Run inference using YOLO
        results = model(frame)

        # Prepare detections for DeepSORT (x1, y1, x2, y2, score)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
                conf = box.conf[0].item()            # Confidence score
                cls = box.cls[0].item()              # Class index (for YOLO, class 0 is person)

                # Append detection only for "person" class
                if cls == 0:  # Person class
                    # Ensure detection is in the correct format
                    detection = [x1, y1, x2, y2, conf]
                    detections.append(detection)

        # Debugging information
        print(f"Number of detections: {len(detections)}")
        print(f"Detections: {detections}")

        # Convert detections to the format expected by DeepSORT
        if detections:
            detections = [detections]  # DeepSORT expects a list of detections in a list
        else:
            detections = [[]]  # DeepSORT expects a list, even if empty

        try:
            # Update the tracker with the new detections
            tracked_objects = tracker.update_tracks(detections, frame=frame)  # Provide the frame here
        except Exception as e:
            print(f"Error during tracking: {e}")
            continue

        # Draw bounding boxes and IDs on the frame
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3], obj['track_id']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame with detections
        cv2.imshow('Child and Therapist Detection', frame)

        # Quit with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("Processing complete.")
