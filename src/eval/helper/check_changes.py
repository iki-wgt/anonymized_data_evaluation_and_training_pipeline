"""
Evaluation script for computing structural similarity (SSIM) between original and anonymized images
in a COCO-format dataset. Focuses on object categories co-occurring with persons.

This script extracts image subsets based on class presence, computes SSIM at the image and object
bounding box level, visualizes differences, and saves the corresponding statistics.
"""
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from bounding_box import bounding_box as bb
from pycocotools.coco import COCO
from tqdm import tqdm
import pandas as pd
import json
import shutil
import copy

from Dataset import Dataset


# --- Configuration ---
gt_config_path = "path_to_repo/config/eval/org_on_org_coco.yaml"
anonym_config_path = "path_to_repo/config/eval/fb_on_anonymized.yaml"

# Target object classes for evaluation
classes = ["car","stop sign","traffic light",
            "cow", "umbrella", "bench",
            "knife", "bed", "chair",
            "potted plant", "clock", "tv"]

# Output directory for evaluation results
results_path = "path_to_repo/data/eval/changes_person"

def get_img_names_containing(class_name, coco_obj):
    """
    Retrieves image IDs containing both the specified class and the 'person' class (ID=1).
    """
    category_id = coco_obj.getCatIds(catNms=[class_name])[0]
    cat_ids = [category_id,1] # person = cat 1
    img_ids_per_cat = [set(coco_obj.getImgIds(catIds=[cat_id])) for cat_id in cat_ids]
    common_img_ids = set.intersection(*img_ids_per_cat)

    return common_img_ids 

def get_img_path_org(img_id, dataset_obj):
    """
    Constructs file path for original image based on COCO-style zero-padded image ID.
    """
    return dataset_obj.get_img_path() + "/" + f"{img_id:012}.jpg"

def get_img_path_anonym(img_id, dataset_obj):
    """
    Constructs file path for anonymized image based on COCO-style zero-padded image ID.
    """
    return dataset_obj.get_img_path() + "/" + f"{img_id:012}.png"

def get_diff_image(img1, img2, threshold=1):
    """
    Computes per-pixel absolute difference and masks regions with changes above threshold.
    """
     # Check if both images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError(f"Image dimensions do not match: {img1.shape} vs {img2.shape}")

    # Compute absolute difference
    diff = cv2.absdiff(img1, img2)

    # Convert to grayscale
    diff_mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply threshold to create a binary mask
    _, binary_mask = cv2.threshold(diff_mask, threshold, 255, cv2.THRESH_BINARY)

    # Create a boolean mask
    img_mask = binary_mask > 0

    # Initialize difference image
    diff_img = np.zeros_like(img2, np.uint8)

    # Apply mask to set changed pixels
    diff_img[img_mask] = img2[img_mask]

    return diff_img

def get_ssim_score(img1, img2):
    """
    Computes SSIM score between two images. Assumes 3-channel color images.
    Original SSIM score paper: https://ece.uwaterloo.ca/%7Ez70wang/publications/ssim.pdf
    """

    # get score
    (score, diff) = structural_similarity(img1, img2, full=True, channel_axis=2)

    return score

def get_ssim_statistics(scores):
    """
    Computes statistical descriptors for SSIM scores.
    """
    statistics = {
            "mean": None,
            "min": None,
            "max": None,
            "count": None,
            "scores": None
        }

    if not scores:
        return statistics

    # Calculating statistics
    total = sum(scores)
    statistics["count"] = len(scores)
    statistics["mean"] = total / statistics["count"]
    statistics["min"] = min(scores)
    statistics["max"] = max(scores)
    statistics["scores"] = scores

    return statistics

def get_ssim_score_bboxes(org_img, anonym_img, img_id, category_id, coco_obj, gt_data, anonym_data):
    """
    Compute SSIM scores within bounding boxes for a specific image and category.

    Parameters:
    - img_id (int): The ID of the image.
    - category_id (int): The category ID to filter annotations.
    - coco_obj (COCO): The COCO object containing annotations.
    - gt_data (Dataset): The ground truth dataset object.
    - anonym_data (Dataset): The anonymized dataset object.

    Returns:
    - list: List of SSIM scores for the bounding boxes in the image.
    """
    scores = []

    # Load annotations for the current image and category
    ann_ids = coco_obj.getAnnIds(imgIds=img_id, catIds=category_id, iscrowd=None)
    anns = coco_obj.loadAnns(ann_ids)

    if not anns:
        print(f"Warning: No annotations found for category ID {category_id} in image ID {img_id}. Skipping.")
        return scores

    for ann in anns:
        # Extract and validate bounding box
        bbox = ann['bbox']  # COCO format: [x, y, width, height]
        x, y, w, h = bbox
        x1, y1 = int(max(x, 0)), int(max(y, 0))
        x2, y2 = int(min(x + w, org_img.shape[1])), int(min(y + h, org_img.shape[0]))

        if x1 >= x2 or y1 >= y2:
            print(f"Warning: Invalid bounding box {bbox} in image ID {img_id}. Skipping this bbox.")
            continue

        # Extract regions of interest (ROIs) from both images
        roi_org = org_img[y1:y2, x1:x2]
        roi_anonym = anonym_img[y1:y2, x1:x2]

        if roi_org.size == 0 or roi_anonym.size == 0:
            print(f"Warning: Empty ROI for bounding box {bbox} in image ID {img_id}. Skipping this bbox.")
            continue

        # Compute SSIM for the ROI
        try:
            ssim_score = get_ssim_score(roi_org, roi_anonym)
            scores.append(ssim_score)
        except ValueError as e:
            print(f"Error computing SSIM for image ID {img_id}, bbox {bbox}: {e}")
            continue

    return scores

def save_ssim_stats(stats, output_dir, filename):
    """
    Saves SSIM statistics to an Excel (.xlsx) file using pandas.

    Parameters:
    - stats (dict): Dictionary containing statistics for each category.
    - output_dir (str): Directory where the Excel file will be saved.
    - filename (str): Name of the Excel file (should end with .xlsx).
    """
    os.makedirs(output_dir, exist_ok=True)
    stats_file_path = os.path.join(output_dir, filename)

    # Prepare summary statistics data
    summary_data = []
    for cat, data in stats.items():
        summary_data.append({
            'Category': cat,
            'Mean SSIM': data['mean'],
            'Min SSIM': data['min'],
            'Max SSIM': data['max'],
            'Count': data['count']
        })

    summary_df = pd.DataFrame(summary_data)

    # Prepare detailed SSIM scores data
    detailed_data = []
    for cat, data in stats.items():
        if data['scores']:
            for score in data['scores']:
                detailed_data.append({'Category': cat, 'SSIM Score': score})
        else:
            detailed_data.append({'Category': cat, 'SSIM Score': 'No scores available.'})

    detailed_df = pd.DataFrame(detailed_data)

    # Write to Excel with two sheets: 'Summary' and 'Detailed Scores'
    with pd.ExcelWriter(stats_file_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
        detailed_df.to_excel(writer, sheet_name='Detailed SSIM Scores', index=False)

    print(f"SSIM statistics saved to {stats_file_path}")

def draw_bboxes(img, bboxes):
    """
    Draws bounding boxes with class labels and confidence scores on the input image.
    """
    for label, conf, box, color in zip(bboxes[0],bboxes[1],bboxes[2],bboxes[3]):
        tl_x, tl_y, br_x, br_y = tuple(box)
        edit_conf = f'{conf:.2f}' if conf else ""
        bb.add(img, tl_x, tl_y, br_x, br_y, f'  {label}  {edit_conf}  ', color)

    return img

def save_img(img, path, filename):
    """
    Saves image to disk at the specified path and filename.
    """
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path , filename), img)

def save_combined_image(org_img, diff_img, anonym_img, path, filename):
    """
    Combines the original, diff, and anonymized images horizontally with labels and saves the result.
    
    Parameters:
    - org_img (numpy.ndarray): Original image.
    - diff_img (numpy.ndarray): Difference image.
    - anonym_img (numpy.ndarray): Anonymized image.
    - path (str): Directory path to save the combined image.
    - filename (str): Filename for the combined image.
    """
    # Ensure all images have the same height
    height = max(org_img.shape[0], diff_img.shape[0], anonym_img.shape[0])
    widths = [img.shape[1] for img in [org_img, diff_img, anonym_img]]
    max_width = max(widths)
    
    # Function to resize images to the same height
    def resize_to_height(img, target_height):
        if img.shape[0] != target_height:
            scale_ratio = target_height / img.shape[0]
            new_width = int(img.shape[1] * scale_ratio)
            return cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
        return img
    
    # Resize images
    org_img_resized = resize_to_height(org_img, height)
    diff_img_resized = resize_to_height(diff_img, height)
    anonym_img_resized = resize_to_height(anonym_img, height)
    
    # If widths are different, pad images to have the same width
    def pad_to_width(img, target_width):
        if img.shape[1] < target_width:
            padding = target_width - img.shape[1]
            return cv2.copyMakeBorder(img, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img
    
    target_width = max_width
    org_img_padded = pad_to_width(org_img_resized, target_width)
    diff_img_padded = pad_to_width(diff_img_resized, target_width)
    anonym_img_padded = pad_to_width(anonym_img_resized, target_width)
    
    # Add labels to each image
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 1
    label_color = (255, 255, 255)  # White color
    label_thickness = 2
    label_bg_color = (0, 0, 0)  # Black background for text
    
    def add_label(img, label):
        text_size, _ = cv2.getTextSize(label, label_font, label_scale, label_thickness)
        text_w, text_h = text_size
        # Add a filled rectangle as background for text
        cv2.rectangle(img, (0, 0), (text_w + 10, text_h + 10), label_bg_color, -1)
        # Put the text above the image
        cv2.putText(img, label, (5, text_h + 5), label_font, label_scale, label_color, label_thickness, cv2.LINE_AA)
        return img
    
    org_img_labeled = add_label(org_img_padded.copy(), "Original")
    diff_img_labeled = add_label(diff_img_padded.copy(), "Diff")
    anonym_img_labeled = add_label(anonym_img_padded.copy(), "Anonymized")
    
    # Concatenate images horizontally
    combined_img = cv2.hconcat([org_img_labeled, diff_img_labeled, anonym_img_labeled])
    
    # Save the combined image
    save_img(combined_img, path, filename)

def load_img(path):
    """
    Loads image from disk using OpenCV.
    """
    return cv2.imread(path)

def setup_ssim_eval_set(imgids_per_class, gt_data, anonym_data, results_path):
    """
    Prepares evaluation dataset by extracting and copying relevant images and annotations.
    """
    # get unique imgs
    unique_ids = set(item for sublist in imgids_per_class.values() for item in sublist)

    # copy images into eval folder
    destination_org = results_path + "/org/imgs/"
    destination_anonym = results_path + "/anonym/imgs/"
    os.makedirs(os.path.dirname(destination_org), exist_ok=True)
    os.makedirs(os.path.dirname(destination_anonym), exist_ok=True)
    
    for img_id in unique_ids:

        # get original and anonym file
        org_img_path = get_img_path_org(img_id, gt_data)
        anonym_img_path = get_img_path_anonym(img_id, anonym_data)
     
        shutil.copy(org_img_path, destination_org)
        shutil.copy(anonym_img_path, destination_anonym)

    # remove unwanted annos and imgs from GT
    gt_org = None
    with open(gt_data.get_anno_path(), 'r') as f:
        gt_org = json.load(f)
    gt_new = copy.deepcopy(gt_org)
    
    filtered_images = [image for image in gt_org['images'] if image['id'] in unique_ids] #keep only those that reference valid image IDs
    filtered_annotations = [annotation for annotation in gt_org['annotations'] if annotation['image_id'] in unique_ids] #keep only those that reference valid image IDs
    
    gt_new['images'] = filtered_images
    gt_new['annotations'] = filtered_annotations

    print(f"saving: unique img ids {len(unique_ids)}, imgs in anno {len(filtered_images)}")

    # save new gt for ssim dataset
    json_path = results_path + "/anno/"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path+"/ssim_gt.json", 'w') as f:
        json.dump(gt_new, f, indent=4)


def main():
    """
    Main evaluation procedure:
    - Load datasets
    - Identify relevant image subsets
    - Compute SSIM statistics per category
    - Prepare output datasets for further use
    """
    gt_data = Dataset(gt_config_path)
    anonym_data = Dataset(anonym_config_path)

    relevant_imgs = {}
    coco_GT = COCO(gt_data.get_anno_path())
    for cat in classes:
        imgs = get_img_names_containing(cat, coco_GT)
        relevant_imgs[cat] = imgs

    stats = {}
    stats_bboxes = {}
    for cat in classes:
        category_id = coco_GT.getCatIds(catNms=[cat])[0]

        scores = []
        scores_bboxes = []

        image_ids = relevant_imgs[cat]
        total_images = len(image_ids)

        if total_images == 0:
            print(f"No images found for category '{cat}'. Skipping.")
            continue

        stats[cat] = get_ssim_statistics(scores)
        stats_bboxes[cat] = get_ssim_statistics(scores_bboxes)

    setup_ssim_eval_set(relevant_imgs, gt_data, anonym_data, results_path)


if __name__ == "__main__":
    main()