# src/get_dataset.py

import os
import tarfile
import requests
from tqdm import tqdm

url = "https://ndownloader.figshare.com/files/5976015"
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
tgz_path = os.path.join(out_dir, "lfw_funneled.tgz")
lfw_dir = os.path.join(out_dir, "lfw_funneled")

os.makedirs(out_dir, exist_ok=True)


# Download with progress
def download_with_progress(url, filepath):
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(filepath, "wb") as file, tqdm(
        desc="📥 Downloading LFW",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))


if not os.path.exists(tgz_path):
    download_with_progress(url, tgz_path)
else:
    print("✅ LFW archive already downloaded.")

# Extract
if not os.path.exists(lfw_dir):
    print("📦 Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=out_dir)
    print(f"✅ Extracted to {lfw_dir}")
else:
    print("✅ LFW already extracted.")
