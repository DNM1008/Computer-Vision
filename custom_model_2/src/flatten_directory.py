"""
This program flatten the directory so that there are no needless
folders/subfolders

"""

import os
import shutil
from collections import defaultdict


def flatten_directory(base_dir):
    """
    Efficiently moves all files from subdirectories into the base directory
    and removes empty folders.

    Args:
        base_dir (str): the directory that needs to be flattened.
    """
    file_counter = defaultdict(int)

    for root, _, files in os.walk(
        base_dir, topdown=False
    ):  # Process deepest folders first
        for file in files:
            src = os.path.join(root, file)
            dest = os.path.join(base_dir, file)

            # Ensure filename uniqueness
            while os.path.exists(dest):
                file_counter[file] += 1
                name, ext = os.path.splitext(file)
                dest = os.path.join(base_dir, f"{name}_{file_counter[file]}{ext}")

            shutil.move(src, dest)
            print(f"Moved: {src} -> {dest}")

        # Remove empty folders
        if root != base_dir:
            try:
                os.rmdir(root)
                print(f"Removed empty folder: {root}")
            except OSError:
                pass  # Ignore non-empty folders


if __name__ == "__main__":
    flatten_directory(os.getcwd())
