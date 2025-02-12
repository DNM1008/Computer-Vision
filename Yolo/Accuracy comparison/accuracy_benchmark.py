from ultralytics import YOLO
import cv2
import time
import pandas as pd

# List of models to compare
models_to_compare = [
    {"name": "yolov9e", "path": "models/yolov9e.pt"},
    {"name": "yolov8l", "path": "models/yolov8l.pt"},
    {"name": "yolo11l", "path": "models/yolo11l.pt"},
]

# Video path
video_path = "video.mp4"

# Output Excel file for results
output_excel = "model_comparison_results.xlsx"


def process_video(model_name, model_path, video_path):
    print(f"Processing video with {model_name}...")

    # Load model using YOLO class (Ultralytics)
    model = YOLO(model_path).to("cuda")

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_time = 0
    total_people_count = 0
    processing_times = []

    # Record per-frame data
    frame_results = {}

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

        # Record frame-specific data
        if frame_count not in frame_results:
            frame_results[frame_count] = {}
        frame_results[frame_count][model_name] = people_count

    cap.release()

    # Calculate metrics
    median_time = sum(processing_times) / len(processing_times)
    total_params = sum(p.numel() for p in model.parameters())
    gflops = "N/A"  # Optional: Calculate GFLOPs if desired

    print(f"Completed processing with {model_name}.")
    
    summary = {
        "Model": model_name,
        "Median Time per Frame (ms)": median_time * 1000,  # Convert to ms
        "Total Parameters": total_params,
        "Total Processing Time (s)": total_time,
        "Total Frames": frame_count,
        "Total People Count": total_people_count
    }

    return summary, frame_results


def main():
    # Results storage
    summary_results = []
    all_frame_results = {}

    for model in models_to_compare:
        summary, frame_results = process_video(model["name"], model["path"], video_path)
        summary_results.append(summary)

        # Combine frame results from all models
        for frame, counts in frame_results.items():
            if frame not in all_frame_results:
                all_frame_results[frame] = {}
            all_frame_results[frame].update(counts)

    # Save results to Excel
    with pd.ExcelWriter(output_excel) as writer:
        # Save summary results
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Save per-frame results
        frame_results_df = pd.DataFrame.from_dict(all_frame_results, orient="index")
        frame_results_df.index.name = "Frame"
        frame_results_df.reset_index(inplace=True)
        frame_results_df.to_excel(writer, sheet_name="Frame Results", index=False)

    print(f"Results saved to {output_excel}")


if __name__ == "__main__":
    main()
