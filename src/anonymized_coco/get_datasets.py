import os
import requests
from tqdm import tqdm
from zipfile import ZipFile
import subprocess
from pycocotools.coco import COCO
import shutil

# URLs for COCO dataset
urls = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "test_images": "http://images.cocodataset.org/zips/test2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "test_annotations": "http://images.cocodataset.org/annotations/image_info_test2017.zip"
}

# Directories to save the downloaded and processed data
coco_dir = "path_to_repo/data/coco_data"
anonymized_dir = "/root/deep_privacy2/data/coco_anonymized"

# DeepPrivacy2 anonymization mode
mode = "fb" # "fb" = full-body; "face" = face

os.makedirs(coco_dir, exist_ok=True)
os.makedirs(anonymized_dir, exist_ok=True)

def download_and_extract(url, output_dir):
    """
    Downloads and extracts a zip archive from a given URL into the specified directory.

    Args:
        url (str): URL pointing to the zip file.
        output_dir (str): Directory where the file will be downloaded and extracted.

    Returns:
        None
    """
    local_filename = os.path.join(output_dir, url.split("/")[-1])
    
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {local_filename}") as t:
            with open(local_filename, 'wb') as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
    
    if local_filename.endswith(".zip"):
        with ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted {local_filename}")
    
    os.remove(local_filename)

def get_images_with_person_annotations(coco_annotation_file):
    """
    Extracts the image IDs that contain person annotations from a COCO annotation file.

    Args:
        coco_annotation_file (str): Path to the COCO annotations JSON file.

    Returns:
        list[int]: List of image IDs containing at least one 'person' instance.
    """
    coco = COCO(coco_annotation_file)
    person_category_id = coco.getCatIds(catNms=['person'])[0]
    image_ids_with_person = coco.getImgIds(catIds=[person_category_id])
    print(f'Image count for anonymization: {len(image_ids_with_person)}')
    return image_ids_with_person

def copy_images_with_person_annotations(image_ids, input_dir, output_dir):
    """
    Copies images whose IDs match the list of image IDs to a new directory.

    Args:
        image_ids (list[int]): List of image IDs to include.
        input_dir (str): Directory containing the source images.
        output_dir (str): Destination directory for the selected images.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_id = int(file.split('.')[0])  # Assuming filename is the image ID
                if image_id in image_ids:
                    shutil.copy2(os.path.join(subdir, file), os.path.join(output_dir, file))
    print(f"Copied {len(os.listdir(output_dir))} images to {output_dir}")

def anonymize_images(image_dirs, output_dir, mode):
    """
    Applies anonymization to directories of person-containing images using external scripts.

    Args:
        image_dirs (list[str]): List of directories containing input images.
        output_dir (str): Root directory to store anonymized outputs.
        mode (str): Anonymization mode ('face' or 'fb').

    Returns:
        None
    """
    for image_dir in image_dirs:
        # Create a subdirectory for each image_dir within the mode's output directory
        mode_output_subdir = os.path.join(output_dir, f"{mode}_{os.path.basename(image_dir)}")
        os.makedirs(mode_output_subdir, exist_ok=True)

        # Run the external script, which should now save directly to the output_subdir
        run_external_processing_script(image_dir, mode_output_subdir, mode)

def run_external_processing_script(input_dir, output_dir, mode):
    """
    Runs an external anonymization script with specified configuration based on the mode.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to save anonymized images.
        mode (str): Mode of anonymization ('face' or 'fb').

    Returns:
        None
    """
    script_path = 'anonymize.py'

    if mode == "face":
        config_file = "/root/deep_privacy2/configs/anonymizers/face.py"
    elif mode == "fb":
        config_file = "/root/deep_privacy2/configs/anonymizers/FB_cse.py"
    else:
        print(f"Error processing mode {mode}: No valid anonymization mode chosen -> valid: face, fb")
        return

    try:
        subprocess.run(["python3", script_path, config_file, "-i", input_dir, "--output_path", output_dir], check=True)
        print(f"Processed directory {input_dir} -> {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_dir}: {e}")


def main():
    """
    Main execution function for preparing the COCO dataset and running anonymization.

    Steps:
        1. Parses COCO annotations to find images containing people.
        2. Copies corresponding images to new directories.
        3. Applies anonymization (face or full-body) to the selected image sets.
    """

    # Paths to annotation files
    train_annotation_file = os.path.join(coco_dir, 'annotations', 'instances_train2017.json')
    val_annotation_file = os.path.join(coco_dir, 'annotations', 'instances_val2017.json')
    test_annotation_file = os.path.join(coco_dir, 'annotations', 'image_info_test2017.json')

    # Get image IDs with person annotations
    image_ids_with_person_train = get_images_with_person_annotations(train_annotation_file)
    image_ids_with_person_val = get_images_with_person_annotations(val_annotation_file)
    image_ids_with_person_test = get_images_with_person_annotations(test_annotation_file)

    # Directories for images
    train_images_dir = os.path.join(coco_dir, 'train2017')
    val_images_dir = os.path.join(coco_dir, 'val2017')
    test_images_dir = os.path.join(coco_dir, 'test2017')

    # Create subdirectories for images with person annotations
    person_train_dir = os.path.join(coco_dir, 'train_person')
    person_val_dir = os.path.join(coco_dir, 'val_person')
    person_test_dir = os.path.join(coco_dir, 'test_person')

    copy_images_with_person_annotations(image_ids_with_person_train, train_images_dir, person_train_dir)
    copy_images_with_person_annotations(image_ids_with_person_val, val_images_dir, person_val_dir)
    copy_images_with_person_annotations(image_ids_with_person_test, test_images_dir, person_test_dir)

    # Anonymize the newly created subdirectories
    image_dirs = [person_train_dir, person_val_dir, person_test_dir]
    anonymize_images(image_dirs, anonymized_dir, mode)

if __name__ == "__main__":
    main()
