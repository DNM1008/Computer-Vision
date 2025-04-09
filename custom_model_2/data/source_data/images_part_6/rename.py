import os

# Directory where your files are located
directory = "."

# List all files in the directory
files = os.listdir(directory)

# Dictionary to store pairs of JPG and JSON files
file_pairs = {}

# Iterate through all files and group JPG and JSON files with matching names
for file in files:
    if file.endswith(".jpg"):
        name, _ = os.path.splitext(file)
        json_file = f"{name}.json"
        if json_file in files:
            file_pairs[name] = (file, json_file)

# Rename files based on the grouped pairs
count = 1
for name, (jpg_file, json_file) in file_pairs.items():
    new_name = f"part_6_image_{count}"
    os.rename(
        os.path.join(directory, jpg_file), os.path.join(directory, f"{new_name}.jpg")
    )
    os.rename(
        os.path.join(directory, json_file), os.path.join(directory, f"{new_name}.json")
    )
    count += 1
