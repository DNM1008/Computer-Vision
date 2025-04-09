import os

# Define your directories
labels_dir = (
    "../data/source_data/labels_full"  # <-- Replace with your actual labels folder
)
images_dir = (
    "../data/source_data/images_full"  # <-- Replace with your actual images folder
)

# Supported image extensions
image_extensions = [".jpeg", ".jpg", ".png"]

# Get all label filenames (without extension)
label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
label_basenames = {os.path.splitext(f)[0] for f in label_files}

# Get all image filenames (without extension), filtering by valid extensions
image_files = [
    f
    for f in os.listdir(images_dir)
    if os.path.splitext(f)[1].lower() in image_extensions
]
image_basenames = {os.path.splitext(f)[0] for f in image_files}

# Labels without matching images
missing_images = label_basenames - image_basenames
# Images without matching labels
missing_labels = image_basenames - label_basenames

# Report
if missing_images:
    print("❌ These label files have no corresponding image files:")
    for name in sorted(missing_images):
        print(f"{name}.txt")

if missing_labels:
    print("\n❌ These image files have no corresponding label files:")
    for name in sorted(missing_labels):
        print(f"{name}.[jpeg|jpg|png]")

if not missing_images and not missing_labels:
    print("✅ All labels and images match!")
