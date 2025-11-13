import os
import shutil

import kagglehub

from abstract2title.paths import DATA_DIR

# Download latest version
path = kagglehub.dataset_download("Cornell-University/arxiv")
print("Path to dataset files:", path)

# Define the destination directory (current directory)
destination_dir = DATA_DIR / "raw"

# Move the downloaded files to the destination directory
try:
    shutil.move(path, destination_dir)
    print(f"Dataset moved to: {destination_dir}")
except shutil.Error as e:
    print(f"Error: {e}. Files may already be in the destination directory.")
