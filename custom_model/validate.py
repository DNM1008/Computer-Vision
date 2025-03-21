import torch
import os
from pathlib import Path
from ultralytics import YOLO


def validate_yolo_model():
    model_path = Path("runs/detect/train/weights/best.pt")
    images_dir = Path("images/train")
    output_dir = Path("runs/detect/validate")

    # Ensure the model exists
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Ensure images directory exists
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Error: Image directory '{images_dir}' not found.")
        return

    # Load YOLO model
    model = YOLO(model_path).to("cpu")

    # Get all image files (jpg and png)
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    if not image_files:
        print("No image files found in the directory.")
        return

    # Run detection
    output_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict(
        image_files,
        save=True,
        project=output_dir.parent,
        name=output_dir.name,
        exist_ok=True,
    )

    print("Detection complete. Results saved in:", output_dir)


if __name__ == "__main__":
    validate_yolo_model()
