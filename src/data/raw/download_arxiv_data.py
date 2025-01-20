import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("Cornell-University/arxiv")
print("Path to dataset files:", path)

# Define the destination directory (current directory)
destination_dir = os.getcwd()

# Move the downloaded files to the destination directory
# shutil.move will raise an error if moving a directory that already exists in the destination,
# so we can handle it in a try-except block
try:
    shutil.move(path, destination_dir)
    print(f"Dataset moved to: {destination_dir}")
except shutil.Error as e:
    print(f"Error: {e}. Files may already be in the destination directory.")
