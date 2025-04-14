"""
Convert json labels to txt labels.

Labelme.exe returns labels, but YOLO takes txt labels to train. Furthermore,
YOLO is different in that the class name is an integer, instead of top left and
bottom right bounding box coordinates


"""

import json
import os

# Input directory (current directory)
json_dir = "../data/source_data/images_part_9"

# Output directory
output_dir = "../data/source_data/big_labels"
os.makedirs(output_dir, exist_ok=True)  # Create if it doesn't exist

# Define class mapping
class_mapping = {"casset": 0, "tien": 1, "atm": 2}

# Loop through all JSON files in the current directory
for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        json_path = os.path.join(json_dir, json_file)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            # Get image dimensions
            image_width = data.get("imageWidth", 1)
            image_height = data.get("imageHeight", 1)

            # Get image name without extension
            image_name = os.path.splitext(json_file)[0]
            output_file = os.path.join(output_dir, f"{image_name}.txt")

            with open(output_file, "w") as f:
                for shape in data.get("shapes", []):
                    label = (
                        shape.get("label", "").strip().lower()
                    )  # Ensure lowercase & trim spaces
                    points = shape.get("points", [])

                    if label not in class_mapping:
                        print(f"Skipping {json_file}: Unknown class '{label}'")
                        continue  # Ignore unknown labels

                    if len(points) < 2:
                        print(f"Skipping {json_file}: Invalid bounding box format")
                        continue

                    x_min, y_min = points[0]
                    x_max, y_max = points[1]

                    # Ensure proper ordering of coordinates
                    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

                    # Convert to YOLO format
                    x_center = (x_min + x_max) / 2 / image_width
                    y_center = (y_min + y_max) / 2 / image_height
                    width = (x_max - x_min) / image_width
                    height = (y_max - y_min) / image_height

                    class_id = class_mapping[label]  # Get class ID from mapping

                    # Write to file
                    f.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )

            print(f"Converted: {json_file} → {output_file}")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

print(f"\nYOLO labels saved in '{output_dir}'")
