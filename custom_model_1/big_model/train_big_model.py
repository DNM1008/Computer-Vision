"""
Importing modules

"""

import os
import warnings

warnings.filterwarnings("ignore")

import shutil
import matplotlib.pyplot as plt
import yaml
import torch
import ultralytics
from tqdm.auto import tqdm
from ultralytics import YOLO

# Set base directory to the current working directory
base_dir = os.getcwd()  # Automatically gets the current directory
print(f"Using base directory: {base_dir}")

# Define dataset paths
train_source_images_dir = os.path.join(base_dir, "images/train")
train_source_labels_dir = os.path.join(base_dir, "labels/train")
val_source_images_dir = os.path.join(base_dir, "images/val")
val_source_labels_dir = os.path.join(base_dir, "labels/val")

# Set local working directory for training
train_destination_dir = os.path.join(base_dir, "train")
val_destination_dir = os.path.join(base_dir, "val")

# Create directories if they don't exist
os.makedirs(train_destination_dir, exist_ok=True)
os.makedirs(val_destination_dir, exist_ok=True)


# Function to copy files from source to destination
def copy_files(source_dir, destination_dir):
    """
    Move the files to their correct directories

    Args:
        source_dir (str): the source directory, where the files currently are
        destination_dir (): the target directory
    """
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} does not exist. Skipping.")
        return
    for filename in tqdm(os.listdir(source_dir), desc=f"Copying from {source_dir}"):
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(destination_dir, filename)
        shutil.copy(src_path, dst_path)


# Copy train images and labels
copy_files(train_source_images_dir, train_destination_dir)
copy_files(train_source_labels_dir, train_destination_dir)

# Copy val images and labels
copy_files(val_source_images_dir, val_destination_dir)
copy_files(val_source_labels_dir, val_destination_dir)


# Validate file consistency (Image-Label pairs check)
def validate_files(directory):
    """
    Ensure that each image has a matching label file

    Args:
        directory (str): the directory where the images and labels are
    """
    file_list = sorted(os.listdir(directory))
    counter = 0
    for i in tqdm(range(0, len(file_list), 2), desc=f"Validating {directory}"):
        if (
            file_list[i][:-4] == file_list[i + 1][:-4]
        ):  # Compare filenames without extensions
            counter += 1
        else:
            print(f"File mismatch found: {file_list[i]} and {file_list[i+1]}")
    print(f"Total matched pairs in {directory}: {counter}")


validate_files(train_destination_dir)
validate_files(val_destination_dir)

# Create dataset.yaml configuration
dataset_config = {
    "nc": 3,  # Number of classes (make sure it matches the number of labels in your dataset)
    "names": ["casset", "tien", "atm"],  # Class names
    "train": train_destination_dir,
    "val": val_destination_dir,
    "device": 0,
}

# Save YAML file
yaml_path = os.path.join(base_dir, "dataset.yaml")
with open(yaml_path, "w") as file:
    yaml.dump(dataset_config, file, default_flow_style=False)

# Check Ultralytics setup
print(ultralytics.checks())

# Load YOLO model
# model = YOLO("yolov11n.pt")
model = YOLO("yolov8n.pt")

# Enable Automatic Mixed Precision for better performance
if torch.cuda.is_available():
    model.half()  # Converts model to half precision

# Move model to GPU if available
if torch.cuda.is_available():
    model.to(device=0)

# Train the model using the local dataset
results = model.train(data=yaml_path, batch=10, epochs=60, imgsz=640)

# Plot training results
result_img_path = os.path.join(base_dir, "runs/detect/train/results.png")
if os.path.exists(result_img_path):
    result_img = plt.imread(result_img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(result_img)
    plt.axis("off")
    plt.show()
else:
    print("Training results image not found!")
