import os

# Directory where your files are located
directory = "."

# List all files in the directory
files = os.listdir(directory)

# Supported image extensions
image_extensions = [".jpg", ".jpeg", ".png"]

# Dictionary to store matched image and JSON files
file_pairs = {}

# Iterate through all files and find image-json pairs
for file in files:
    name, ext = os.path.splitext(file)
    if ext.lower() in image_extensions:
        json_file = f"{name}.json"
        if json_file in files:
            file_pairs[name] = (file, json_file)

# Rename matched image-json pairs
count = 1
for name, (image_file, json_file) in file_pairs.items():
    new_name = f"part_4_image_{count}"
    new_ext = os.path.splitext(image_file)[1].lower()  # keep original extension
    os.rename(
        os.path.join(directory, image_file),
        os.path.join(directory, f"{new_name}{new_ext}"),
    )
    os.rename(
        os.path.join(directory, json_file), os.path.join(directory, f"{new_name}.json")
    )
    count += 1

print(f"Renamed {count - 1} image/json pairs.")
