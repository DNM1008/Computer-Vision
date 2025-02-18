import cv2
import pandas as pd
import torch
from openpyxl import Workbook

def process_video(input_video, output_video, output_excel, model_name):
    # Load YOLO model
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    # Open the video file
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Initialize an empty list to store frame-wise person counts
    data = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO on the current frame
        results = model(frame)

        # Extract detections and filter for people (class ID 0 in COCO dataset)
        detections = results.pred[0].cpu().numpy() if hasattr(results, 'pred') else results[0].cpu().numpy()
        person_detections = [det for det in detections if int(det[5]) == 0]

        # Count the number of people in the frame
        person_count = len(person_detections)
        data.append({'Frame': frame_id, 'PersonCount': person_count})

        # Draw bounding boxes on the frame
        for det in person_detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Person {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

        frame_id += 1

    cap.release()
    out.release()

    # Save the results to an Excel file
    df = pd.DataFrame(data)
    df.to_excel(output_excel, index=False)

    print(f"Processing with {model_name} complete. Results saved in {output_excel} and {output_video}")

# Input and output configurations
videos = [
    {"input": "video.mp4", "output_video": "outputv8.mp4", "output_excel": "people_count_v8.xlsx", "model_name": "yolov8l"},
    {"input": "video.mp4", "output_video": "outputv11.mp4", "output_excel": "people_count_v11.xlsx", "model_name": "yolov5lu"} # Updated for YOLOv5u
]

for video_config in videos:
    process_video(video_config["input"], video_config["output_video"], video_config["output_excel"], video_config["model_name"])
