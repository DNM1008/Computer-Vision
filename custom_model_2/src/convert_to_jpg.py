"""
Convert jpg files to png files for easy management.

"""

import os
from PIL import Image

# Directory containing the images (change this if needed)
input_dir = "../data/source_data/data_full"

# Supported formats
convert_extensions = [".jpeg", ".png"]

# Loop through all files in the directory
for filename in os.listdir(input_dir):
    name, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext in convert_extensions:
        file_path = os.path.join(input_dir, filename)
        output_path = os.path.join(input_dir, f"{name}.jpg")

        try:
            with Image.open(file_path) as img:
                rgb_img = img.convert("RGB")  # Ensure no alpha channel
                rgb_img.save(output_path, "JPEG")

            # Delete the original file after successful conversion
            os.remove(file_path)
            print(f"Converted and deleted: {filename} -> {name}.jpg")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
