import os

import cv2
from ultralytics import YOLO

# Specify the model and directories
model_name = "yolov8l"
model_path = "../models/yolov8l.pt"  # Path to your model
input_folder = "../images"  # Folder containing PNG images
output_folder = "../output_images"  # Folder to save the output images


def count_people_in_folder(model_name, model_path, input_folder, output_folder):
    print(
        f"Counting people in all PNG images in folder '{input_folder}' using {model_name}..."
    )

    # Load the model
    model = YOLO(model_path).to(
        "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    )

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all PNG files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".png"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            print(f"Processing {input_path}...")

            # Load the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not load image from {input_path}.")
                continue

            # Perform inference
            results = model(image)

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
                    # Draw the bounding box on the image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add a label
                    cv2.putText(
                        image,
                        "Person",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # Save the result image
            cv2.imwrite(output_path, image)
            print(f"{file_name}: Total number of people detected: {people_count}")


if __name__ == "__main__":
    count_people_in_folder(model_name, model_path, input_folder, output_folder)
