import os

import cv2
from ultralytics import YOLO

# Specify the model and directories
model_name = "yolov8l"
model_path = "../models/yolov8l.pt"  # Path to your model
input_folder = "../images"  # Folder containing image files
output_folder = "../output_near_images_crop"  # Folder to save the output images


def count_people_near_objects(model_name, model_path, input_folder, output_folder):
    print(
        f"Detecting objects and counting people near TVs, chairs, or trucks in PNG images in '{input_folder}' using {model_name}..."
    )

    # Load the model
    model = YOLO(model_path).to(
        "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    )

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Object class IDs for YOLOv8
    OBJECT_CLASSES = {
        63: "TV",
        56: "Chair",
        7: "Truck",
        0: "Person",
    }  # COCO dataset class IDs

    # Iterate through all PNG image files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".png"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            print(f"Processing {input_path}...")

            # Read image file
            frame = cv2.imread(input_path)
            if frame is None:
                print(f"Error: Could not open image file {input_path}.")
                continue

            # Perform inference
            results = model(frame)
            detections = results[0].boxes

            object_boxes = []

            # Identify objects
            for det in detections:
                cls_id = int(det.cls)
                if (
                    cls_id in OBJECT_CLASSES and cls_id != 0
                ):  # Ignore people for object detection
                    x1, y1, x2, y2 = map(int, det.xyxy.cpu().numpy().flatten())
                    object_boxes.append((x1, y1, x2, y2, OBJECT_CLASSES[cls_id]))

            # Process each detected object by cropping around it
            for ox1, oy1, ox2, oy2, obj_name in object_boxes:
                cx, cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2  # Center of detected object
                h, w, _ = frame.shape

                # Compute crop dimensions (1/4 of original width and height)
                crop_w = w // 3
                crop_h = h // 3

                # Compute crop coordinates ensuring they stay within the image bounds
                x1_crop = max(cx - crop_w, 0)
                x2_crop = min(cx + crop_w, w)
                y1_crop = max(cy - crop_h, 0)
                y2_crop = min(cy + crop_h, h)

                cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                # Perform inference on cropped image
                cropped_results = model(cropped_frame)
                cropped_detections = cropped_results[0].boxes

                people_count = 0
                for det in cropped_detections:
                    cls_id = int(det.cls)
                    if cls_id == 0:  # Person
                        px1, py1, px2, py2 = map(int, det.xyxy.cpu().numpy().flatten())
                        people_count += 1
                        cv2.rectangle(
                            cropped_frame, (px1, py1), (px2, py2), (0, 255, 0), 2
                        )
                        cv2.putText(
                            cropped_frame,
                            "Person",
                            (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                # Save cropped image with detections
                cropped_output_path = os.path.join(
                    output_folder, f"cropped_{file_name}"
                )
                cv2.imwrite(cropped_output_path, cropped_frame)
                print(
                    f"Processed {file_name}: Cropped image saved to {cropped_output_path}, People Count: {people_count}"
                )


if __name__ == "__main__":
    count_people_near_objects(model_name, model_path, input_folder, output_folder)
