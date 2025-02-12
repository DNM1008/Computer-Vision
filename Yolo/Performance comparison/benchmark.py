from ultralytics import YOLO
import cv2
import time
import pandas as pd

# List of models to compare
models_to_compare = [
    {"name": "yolov5n", "path": "models/yolov5n.pt"},
    {"name": "yolov5m", "path": "models/yolov5m.pt"},
    {"name": "yolov5l", "path": "models/yolov5l.pt"},
    {"name": "yolov5x", "path": "models/yolov5x.pt"},
    {"name": "yolov8n", "path": "models/yolov8n.pt"},
    {"name": "yolov8m", "path": "models/yolov8m.pt"},
    {"name": "yolov8l", "path": "models/yolov8l.pt"},
    {"name": "yolov8x", "path": "models/yolov8x.pt"},
    {"name": "yolo11n", "path": "models/yolo11n.pt"},
    {"name": "yolo11m", "path": "models/yolo11m.pt"},
    {"name": "yolo11l", "path": "models/yolo11l.pt"},
    {"name": "yolo11x", "path": "models/yolo11x.pt"}
]

# Video path
video_path = "video (11).mp4"

# Output Excel file for results
output_excel = "model_comparison_results1.xlsx"

def process_video(model_name, model_path, video_path):
    print(f"Processing video with {model_name}...")
    
    # Load model using YOLO class (Ultralytics)
    model = YOLO(model_path).to('cuda')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_time = 0
    total_people_count = 0
    processing_times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Start processing
        start_time = time.time()
        results = model(frame)
        end_time = time.time()

        # Count people (class id 0 for "person")
        detections = results[0].boxes  # For YOLOv8, results[0].boxes gives a Box object
        people_count = 0
        for det in detections:
            # Access class id (det.cls) and check if it's a person (usually class 0)
            if det.cls == 0:
                people_count += 1

        total_people_count += people_count

        # Track performance metrics
        processing_times.append(end_time - start_time)
        total_time += (end_time - start_time)
        frame_count += 1

    cap.release()

    # Calculate metrics
    median_time = sum(processing_times) / len(processing_times)
    total_params = sum(p.numel() for p in model.parameters())
    gflops = "N/A"  # Optional: Calculate GFLOPs if desired

    print(f"Completed processing with {model_name}.")
    return {
        "Model": model_name,
        "Median Time per Frame (ms)": median_time * 1000,  # Convert to ms
        "Total Parameters": total_params,
        "Total Processing Time (s)": total_time,
        "Total Frames": frame_count,
        "Total People Count": total_people_count
    }

def main():
    # Results storage
    results = []

    for model in models_to_compare:
        result = process_video(model["name"], model["path"], video_path)
        results.append(result)

    # Save results to Excel and drop the Median Time per Frame (s) column
    df = pd.DataFrame(results)
    # Saving only the converted time column and the other required metrics
    df.drop(columns=["Median Time per Frame (s)"], errors='ignore', inplace=True)
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")

if __name__ == "__main__":
    main()
