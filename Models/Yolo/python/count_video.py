import os

import cv2
from ultralytics import RTDETR

# Specify the model and directories
model_name = "rtdetr-l"
model_path = "../models/rtdetr-l.pt"  # Path to your model
input_folder = "../videos"  # Folder containing video files
output_folder = "../output_videos"  # Folder to save the output videos


def count_people_in_videos(model_name, model_path, input_folder, output_folder):
    print(
        f"Counting people in all AVI video files in folder '{input_folder}' using {model_name}..."
    )

    # Load the model
    model = RTDETR(model_path).to(
        "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    )

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all AVI video files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".avi"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            print(f"Processing {input_path}...")

            # Open video file
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {input_path}.")
                continue

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform inference
                results = model(frame)

                # Count people (class id 0 for "person") and draw bounding boxes
                people_count = 0
                detections = results[
                    0
                ].boxes  # For YOLOv8, results[0].boxes gives a Box object
                for det in detections:
                    if det.cls == 0:  # Check if the detected object is a person
                        people_count += 1
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, det.xyxy.cpu().numpy().flatten())
                        # Draw the bounding box on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Add a label
                        cv2.putText(
                            frame,
                            "Person",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                # Display people count on the frame
                cv2.putText(
                    frame,
                    f"People Count: {people_count}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Write the processed frame to the output video
                out.write(frame)

            # Release resources
            cap.release()
            out.release()
            print(f"Processed {file_name}: Video saved to {output_path}")


if __name__ == "__main__":
    count_people_in_videos(model_name, model_path, input_folder, output_folder)
