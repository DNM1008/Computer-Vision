import math
import os

import cv2
import torch
from ultralytics import YOLO

# Specify model and directories
model_name = "yolov8l"
model_path = "../models/yolov8l.pt"  # Path to your model
input_folder = "../images"  # Folder containing image files
output_folder = "../output_near_images_dist"  # Updated folder to save output images


def count_people_near_objects(model_name, model_path, input_folder, output_folder):
    print(
        f"Detecting objects and counting people near TVs, chairs, or trucks in PNG images in '{input_folder}' using {model_name}..."
    )

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Object class IDs for YOLOv8 (COCO dataset)
    OBJECT_CLASSES = {63: "TV", 56: "Chair", 7: "Truck", 0: "Person"}

    # Iterate through PNG images in the input folder
    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".png"):
            continue

        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        print(f"Processing {input_path}...")

        # Read image
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Could not open image file {input_path}.")
            continue

        # Get the image dimensions
        height, width, _ = frame.shape

        # Perform inference
        results = model(frame)
        detections = results[0].boxes

        object_boxes = []
        people_boxes = []

        # Identify objects and people
        for det in detections:
            cls_id = int(det.cls)
            if cls_id in OBJECT_CLASSES:
                x1, y1, x2, y2 = map(int, det.xyxy.cpu().numpy().astype(int).flatten())
                if cls_id == 0:  # Person
                    people_boxes.append((x1, y1, x2, y2))
                else:
                    object_boxes.append((x1, y1, x2, y2, OBJECT_CLASSES[cls_id]))

        # Count people near objects based on center distance
        counted_people = set()

        def is_within_diagonal_distance(
            obj_center_x,
            obj_center_y,
            person_center_x,
            person_center_y,
            display_width,
            display_height,
        ):
            """Check if the person's center is within the diagonal distance from the object's center."""
            # Calculate the Euclidean distance between the two centers
            horizontal_distance = (obj_center_x - person_center_x) ** 2
            vertical_distance = (obj_center_y - person_center_y) ** 2
            distance = math.sqrt(horizontal_distance + vertical_distance)

            # Calculate the diagonal of the screen using the Pythagorean theorem
            screen_diagonal = math.sqrt(display_width**2 + display_height**2)

            return distance <= screen_diagonal / 2

        for p_box in people_boxes:
            # Get the center of the person
            p_center_x = (p_box[0] + p_box[2]) / 2
            p_center_y = (p_box[1] + p_box[3]) / 2

            for o_box in object_boxes:
                # Get the center of the object
                o_center_x = (o_box[0] + o_box[2]) / 2
                o_center_y = (o_box[1] + o_box[3]) / 2

                # Check if the person's center is within half the diagonal distance from the object's center
                if is_within_diagonal_distance(
                    o_center_x, o_center_y, p_center_x, p_center_y, width, height
                ):
                    counted_people.add(p_box)
                    cv2.rectangle(
                        frame,
                        (p_box[0], p_box[1]),
                        (p_box[2], p_box[3]),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        "Person",
                        (p_box[0], p_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    break  # No need to check further objects for this person

        people_count = len(counted_people)

        # Draw detected objects
        for ox1, oy1, ox2, oy2, obj_name in object_boxes:
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                obj_name,
                (ox1, oy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        # Display people count on the image
        cv2.putText(
            frame,
            f"People Near Objects: {people_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Save processed image
        cv2.imwrite(output_path, frame)
        print(f"Processed {file_name}: Image saved to {output_path}")


if __name__ == "__main__":
    count_people_near_objects(model_name, model_path, input_folder, output_folder)
