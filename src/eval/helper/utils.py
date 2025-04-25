"""
Processing Helper Script

Script provides functions for various helper functions, including:
- Checking and creating directories
- Loading YAML configuration files
- Converting detections to COCO style format
- Adding bounding boxes to images and saving them
"""
import os
import yaml
import numpy as np
from PIL import Image
from bounding_box import bounding_box as bb
import cv2

def check_path(path) -> bool:
    """
    Check if a directory path exists, and create it if it doesn't.

    Arguments:
    - path (str): The path to check or create.

    Returns:
    - bool: True if the directory exists or was successfully created, False otherwise.
    """

    try:
        os.makedirs(path)
        return True
    except FileExistsError:
        # directory already exists
        return True
    except Exception as e:
        print("An error occurred:",e)
        return False

def load_config(path):
    """
    Load a YAML configuration file.

    Arguments:
    - path (str): The path to the YAML configuration file.

    Returns:
    - dict: Dictionary containing the configuration settings.
    """

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def detections_to_COCO(image_id, cat_ids, bboxes, scores):
    """
    Convert detections to COCO style structure.

    Arguments:
    - image_id (int): The ID of the image.
    - cat_ids (list): A list of category IDs.
    - bboxes (list): List of bounding boxes in COCO syle [x_min, y_min, width, height]
    - scores (list): List of confidence scores.

    Returns:
    - list: List of dictionaries representing the detections in COCO style format.
    """

    if not len(cat_ids) == len(bboxes) == len(scores):
        print("Wrong size of lists - all must be equal in lenght")
        return None

    results = []
    for cat_id, bbox, score in zip(cat_ids, bboxes, scores):
        entry = {}
        entry["image_id"] = image_id
        entry["category_id"] = cat_id
        entry["bbox"] = bbox
        entry["score"] = score

        results.append(entry)

    return results

def add_bbox_and_save(img, bboxes, path):
    """
    Add bounding boxes to an image and save it.

    Attention:
    This function does not check if path exists - do it before calling!

    Arguments:
    - img (PIL.Image.Image): The image to which bounding boxes will be added.
    - bboxes (list): A list of lists containing bounding box coordinates and additional information.
    - path (str): The path to save the image with bounding boxes.
    """

    # check if any of th lists is empty
    is_any_empty = any(len(sublist) == 0 for sublist in bboxes)
    if is_any_empty:
        # no box to add - just save plain img
        img.save(path)
        return

    # color conversion to BGR (for OpenCV)
    np_img = np.array(img, dtype=np.uint8)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # draw bboxes
    for label, conf, box, color in zip(bboxes[0],bboxes[1],bboxes[2],bboxes[3]):
        tl_x, tl_y, br_x, br_y = tuple(box)
        edit_conf = f'{conf:.2f}' if conf else ""
        bb.add(np_img, tl_x, tl_y, br_x, br_y, f'  {label}  {edit_conf}  ', color)


    # color conversion back to RGB
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    # save img
    boxed_img = Image.fromarray(np.uint8(np_img))
    boxed_img.save(path) 
