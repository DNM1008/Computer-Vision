import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model (Large version)
model = YOLO("../app/data/best_casset_1.pt")


def process_video(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Ensure FPS is valid
    if fps == 0 or fps is None:
        fps = 30  # Set default FPS

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' codec for MP4 output
    output_video_path = output_video_path.replace(".mkv", ".mp4")  # Force MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Check if VideoWriter initialized correctly
    if not out.isOpened():
        print("Error: VideoWriter failed to initialize. Check codec and output path.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 model on the frame
        results = model(frame)

        # Draw results on frame
        annotated_frame = results[0].plot()

        # Write the frame to the output video
        out.write(annotated_frame)
        print("Frame written successfully.")  # Debug message

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved as {output_video_path}")


# Example usage
input_video = "../../videos/01_22_2025 2_59_59 PM (UTC+07_00).mkv"  # Replace with your input video file path
output_video = (
    "../../videos/01_22_2025 2_59_59 PM (UTC+07_00)_test_1.mkv"  # Output file path
)
process_video(input_video, output_video)
