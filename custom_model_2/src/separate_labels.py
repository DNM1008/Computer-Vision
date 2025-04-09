"""
label_image_splitter.py

This script processes YOLO-format label files by splitting and organising the data by class.
For each label file, it:

1. Reads bounding box annotations and class IDs.
2. Filters annotations into three classes: 'casset', 'tien', and 'atm'.
3. Writes new label files into class-specific folders.
4. Copies corresponding image files into the same class folders.

Input:
- `labels_dir`: Directory containing YOLO `.txt` label files.
- `images_dir`: Directory containing corresponding images (.jpg, .jpeg, .png).

Output:
- A main `output_root` directory with three subfolders: `casset/`, `tien/`, and `atm/`,
  each containing both the label files and their matching images.

Usage:
- Edit the `labels_dir`, `images_dir`, and `output_root` variables in `process_labels_and_images()`.
- Run this script with: `python label_image_splitter.py`
"""

import os
import shutil
import concurrent.futures


def process_label_file(file, labels_dir, images_dir, class_dirs, image_extensions):
    """
    Processes a single YOLO label file:
    - Splits the annotations by class.
    - Writes new label files for each class.
    - Copies the corresponding image into the same class folder.

    Args:
        file (str): The label file name (e.g., 'image1.txt').
        labels_dir (str): Directory containing original YOLO label files.
        images_dir (str): Directory containing the images.
        class_dirs (dict): Mapping of class names to their output directories.
        image_extensions (list): Allowed image extensions to search for.
    """
    file_path = os.path.join(labels_dir, file)
    base_name = os.path.splitext(file)[0]

    # Store label lines for each class
    class_data = {"casset": [], "tien": [], "atm": []}

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = parts[0]

            if class_id == "0":
                class_data["casset"].append(line)
            elif class_id == "1":
                class_data["tien"].append(line)
            elif class_id == "2":
                class_data["atm"].append(line)

    # Write new label files and copy images into corresponding class directories
    for class_name, lines in class_data.items():
        if lines:
            class_dir = class_dirs[class_name]
            os.makedirs(class_dir, exist_ok=True)

            # Write filtered label
            new_label_path = os.path.join(class_dir, f"{base_name}.txt")
            with open(new_label_path, "w") as f:
                f.writelines(lines)

            # Copy image only once for this class
            for ext in image_extensions:
                image_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(image_path):
                    new_image_path = os.path.join(class_dir, base_name + ext)
                    shutil.copy(image_path, new_image_path)
                    break


def process_labels_and_images():
    """
    Orchestrates the label/image splitting and saving process:
    - Reads label files.
    - Splits them by class using `process_label_file()`.
    - Saves both label and image into class-specific folders.
    """
    # ==== SET DIRECTORIES ====
    labels_dir = "../data/source_data/big_labels/"
    images_dir = "../data/source_data/big_images/"
    output_root = os.path.join(os.getcwd(), "../data/source_data/")
    # =========================

    image_extensions = [".jpg", ".jpeg", ".png"]

    # Create output folder structure for each class
    class_dirs = {
        "casset": os.path.join(output_root, "casset"),
        "tien": os.path.join(output_root, "tien"),
        "atm": os.path.join(output_root, "atm"),
    }

    label_files = [file for file in os.listdir(labels_dir) if file.endswith(".txt")]

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(
            lambda file: process_label_file(
                file, labels_dir, images_dir, class_dirs, image_extensions
            ),
            label_files,
        )


if __name__ == "__main__":
    process_labels_and_images()
