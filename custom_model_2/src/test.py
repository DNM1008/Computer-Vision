"""
This program tests the custom models by inferring it to a video and save the
result.

"""

import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model (Large version)
big_model = "../data/results/big_model/weights/best.pt"
small_model_cassette = "../data/results/small_cassette/weights/best.pt"


def process_video(input_video_path, output_video_path, model_path):
    """
    Process the video using custom YOLO model

    Args:
        input_video_path (str): path to the input video
        output_video_path (str): path to the output video
        model_path (str): path the the trained model
    """
    model = YOLO(model_path)
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
input_video = "../../videos/raw/01_16_2025 4_05_59 PM (UTC+07_00).mkv"  # Replace with your input video file path
output_video_big = "../../videos/results/01_16_2025 4_05_59 PM (UTC+07_00)_full.mkv"  # Output file path
output_video_small_cassette = "../../videos/results/01_16_2025 4_05_59 PM (UTC+07_00)_cassette.mkv"  # Output file path
process_video(input_video, output_video_big, big_model)

print("Tested big model")
process_video(input_video, output_video_small_cassette, small_model_cassette)

print("Tested small model cassette")
