import os
import shutil
import concurrent.futures


def process_label_file(
    file, labels_dir, class_folders, label_folders, image_extensions, current_dir
):
    file_path = os.path.join(labels_dir, file)
    base_name = os.path.splitext(file)[0]

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

    # Write filtered label files to separate label folders
    for class_name, lines in class_data.items():
        if lines:
            new_label_path = os.path.join(label_folders[class_name], f"{base_name}.txt")
            with open(new_label_path, "w") as f:
                f.writelines(lines)

    # Move corresponding images
    for ext in image_extensions:
        image_path = os.path.join(current_dir, base_name + ext)
        if os.path.exists(image_path):
            for class_name in class_data:
                if class_data[class_name]:  # Only copy if labels exist
                    new_image_name = f"{base_name}{ext}"
                    new_image_path = os.path.join(
                        class_folders[class_name], new_image_name
                    )
                    shutil.copy(image_path, new_image_path)
            break  # Stop checking other extensions once found


def process_labels_and_images():
    current_dir = os.getcwd()
    labels_dir = os.path.join(current_dir, "yolo_labels")
    image_extensions = {".jpg", ".jpeg", ".png"}

    # Ensure output folders exist in current directory
    class_folders = {
        "casset": os.path.join(current_dir, "image_casset"),
        "tien": os.path.join(current_dir, "image_tien"),
        "atm": os.path.join(current_dir, "image_atm"),
    }

    label_folders = {
        "casset": os.path.join(current_dir, "labels_casset"),
        "tien": os.path.join(current_dir, "labels_tien"),
        "atm": os.path.join(current_dir, "labels_atm"),
    }

    for folder in class_folders.values():
        os.makedirs(folder, exist_ok=True)

    for folder in label_folders.values():
        os.makedirs(folder, exist_ok=True)

    label_files = [file for file in os.listdir(labels_dir) if file.endswith(".txt")]

    # Process label files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(
            lambda file: process_label_file(
                file,
                labels_dir,
                class_folders,
                label_folders,
                image_extensions,
                current_dir,
            ),
            label_files,
        )


if __name__ == "__main__":
    process_labels_and_images()
