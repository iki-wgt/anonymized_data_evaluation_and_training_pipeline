# Script to get random sample from a dataset - percentage can be configured.
# Creates symlinks to data, does not make copies!
# Configure paths at end of file.

import os
import random
from tqdm import tqdm

def create_symlinks(src_dir, alt_dir, dest_dir, alt_dest_dir, percentage):
    # Ensure the destination directories exist
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(alt_dest_dir, exist_ok=True)
        
    # List all files in the source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    # Calculate the number of files to select based on the percentage
    num_files_to_select = int(len(files) * percentage / 100)
    
    # Select a random subset of files
    selected_files = random.sample(files, num_files_to_select)
    
    # Prepare list of label files
    label_files = []
    for file in selected_files:
        base_name, _ = os.path.splitext(file)
        alt_file_path = os.path.join(alt_dir, base_name + '.txt')
        if os.path.exists(alt_file_path):
            label_files.append(file)
    
    # Initialize progress bars
    image_pbar = tqdm(total=len(selected_files), desc='Creating image symlinks')
    label_pbar = tqdm(total=len(label_files), desc='Creating label symlinks')
    
    # Create symlinks
    for file in selected_files:
        base_name, ext = os.path.splitext(file)
        
        # Paths for the original file
        src_file_path = os.path.join(src_dir, file)
        dest_file_path = os.path.join(dest_dir, file)
        
        # Symlink for image
        if not os.path.exists(dest_file_path):
            os.symlink(src_file_path, dest_file_path)
        
        # Update image progress bar
        image_pbar.update(1)
        
        # Check if label exists
        alt_file_path = os.path.join(alt_dir, base_name + '.txt')
        alt_dest_file_path = os.path.join(alt_dest_dir, base_name + '.txt')
        if file in label_files:
            if not os.path.exists(alt_dest_file_path):
                os.symlink(alt_file_path, alt_dest_file_path)
            # Update label progress bar
            label_pbar.update(1)
    
    # Close progress bars
    image_pbar.close()
    label_pbar.close()

# img
## Original img
src_directory = 'path_to_repo/data/coco/coco_anonymized/fb_train_person'
## Symlink dir img
dest_directory = 'path_to_repo/src/train_YOLOv10/datasets/images/fb_train_person'

# label
## Original label
alt_directory = 'path_to_repo/src/train_YOLOv10/datasets/labels/train2017'
## Symlink dir label
alt_dest_directory = 'path_to_repo/src/train_YOLOv10/datasets/labels/fb_train_person'

percentage = 40  # Select % of files randomly from source dir
# is is roughly around 20% of original data of coco train 2017

create_symlinks(src_directory, alt_directory, dest_directory, alt_dest_directory, percentage)