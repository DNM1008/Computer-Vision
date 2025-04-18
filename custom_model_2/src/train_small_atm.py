"""
This program trains an AI Vision model that would detect opened ATMs (atm).

It does so by doing the following:
    - Make sure that the labels and the image files match. For every image file,
      there must be a label file. A label file is an .txt file that has the
      following format:
          ` class x_center y_center width height`
          where class is an interger denoting the class and the coordinates and
          sizes are ratios relative to the image's actual size. For example, if
          the image is 1920x1080 and a dot is at 960x540, its x_center and y_center
          would be 0.5 and 0.5.
    - It generates a dataset yaml configuration file for later training or if the need
      arises for reconstruction and saves it to `../conf`
    - It checks for a YOLOv8 model (n in this case) and downloads to
      `../conf/source_model/` if there isn't one there.
    - It then trains the model to recognise the objects and save the results in
      `../data/results/big_model/` and tries to display the result image.
    - It tries to clean up the .cache files
"""

import os
import warnings
import glob
import shutil
import yaml

import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import ultralytics
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# Set base directory to the current working directory
base_dir = os.getcwd()  # Automatically gets the current directory
print(f"Using base directory: {base_dir}")

# Define dataset paths

# Set local working directory for training
train_destination_dir = os.path.join(base_dir, "../data/small_atm_train")
val_destination_dir = os.path.join(base_dir, "../data/small_atm_val")

# Create directories if they don't exist
os.makedirs(train_destination_dir, exist_ok=True)
os.makedirs(val_destination_dir, exist_ok=True)


# Validate file consistency (Image-Label pairs check)
def validate_files(directory):
    """
    Ensure that each image has a matching label file

    Args:
        directory (str): the directory where the images and labels are
    """
    images = {
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.endswith((".jpg", ".png"))
    }
    labels = {
        os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(".txt")
    }

    matched = images & labels
    unmatched_images = images - labels
    unmatched_labels = labels - images

    print(f"Matched pairs: {len(matched)}")
    if unmatched_images:
        print(f"Images without labels: {unmatched_images}")
    if unmatched_labels:
        print(f"Labels without images: {unmatched_labels}")


def delete_cache_files(folder_path):
    """
    Remove the .cache file in a directory

    Args:
        folder_path (str): path to the directory where the .cache files are
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".cache"):
                file_path = os.path.join(root, file)
                print(f"Deleting: {file_path}")
                os.remove(file_path)


validate_files(train_destination_dir)
validate_files(val_destination_dir)

# Create dataset.yaml configuration
dataset_config = {
    "nc": 1,  # Number of classes (make sure it matches the number of labels in your dataset)
    "names": ["atm"],  # Class names
    "train": train_destination_dir,
    "val": val_destination_dir,
    "device": 0,
}

# Save YAML file
yaml_path = os.path.join(base_dir, "../conf/small_atm_dataset.yaml")
with open(yaml_path, "w") as file:
    yaml.dump(dataset_config, file, default_flow_style=False)

# Check Ultralytics setup
print(ultralytics.checks())

# Load YOLO model
# model = YOLO("yolov11n.pt")
model = YOLO("../conf/source_model/yolov8n.pt")

# Enable Automatic Mixed Precision for better performance
if torch.cuda.is_available():
    model.half()  # Converts model to half precision

# Move model to GPU if available
if torch.cuda.is_available():
    model.to(device=0)

# Train the model using the local dataset
results = model.train(
    data=yaml_path,
    batch=10,
    epochs=60,
    project=os.path.join(base_dir, "../data/results"),
    name="small_atm",
    imgsz=640,
)

# Plot training results
result_img_path = os.path.join(base_dir, "../data/results/small_atm/results.png")
if os.path.exists(result_img_path):
    result_img = plt.imread(result_img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(result_img)
    plt.axis("off")
    plt.show()


else:
    print("Training results image not found!")


delete_cache_files(os.path.join(base_dir, "../data/"))
