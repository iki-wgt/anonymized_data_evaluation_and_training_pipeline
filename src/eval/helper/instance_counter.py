"""
Dataset Overview Statistics Script

This script analyzes COCO-format annotations and image directories to compute statistics
on instance counts and image occurrences per class. It compares:
    (i) full COCO training data,
    (ii) person-containing subsets,
    (iii) training images used for fine-tuning.

Outputs are saved to Excel with structured summary tables.
"""
import os
from collections import Counter
from tqdm import tqdm
import pandas as pd
from pycocotools.coco import COCO


# --- Paths ---
path_anno = 'path_to_repo/data/coco/annotations/instances_train2017.json'
img_path = 'path_to_repo/src/train_YOLOv10/datasets/images/fb_train_person/'
result_path = 'path_to_repo/data/eval/'

# --- Utility Functions ---
def get_used_img_id(image_dir):
    """
    Extracts image IDs from filenames in a given directory. Assumes filenames are zero-padded COCO-style image IDs.

    Args:
        image_dir (str): Path to the directory containing image files.

    Returns:
        list[int]: List of image IDs (as integers) extracted from the filenames.
    """
    return [int(os.path.splitext(filename)[0].lstrip('0')) 
            for filename in os.listdir(image_dir) 
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

def get_instances_per_class(anno, category_dict):
    """
    Counts the number of object instances per class based on annotation data.

    Args:
        anno (list[dict]): List of annotation dictionaries (e.g., COCO-style).
        category_dict (dict[int, str]): Mapping from category IDs to category names.

    Returns:
        dict[str, int]: Dictionary mapping class names to instance counts, sorted in descending order.
    """
    category_counts = Counter(d['category_id'] for d in anno)
    instances = {category_dict[cat_id]: count for cat_id, count in category_counts.items()}
    # Sort by count descending
    return dict(sorted(instances.items(), key=lambda item: item[1], reverse=True))

def get_images_per_class(anno, category_dict):
    """
    Computes the number of unique images per class and the total number of unique images.

    Args:
        anno (list[dict]): List of annotation dictionaries (e.g., COCO-style).
        category_dict (dict[int, str]): Mapping from category IDs to category names.

    Returns:
        dict[str, int]: Dictionary mapping class names to unique image counts. Includes "Total Unique Images".
    """
    image_counts = {cat_id: set() for cat_id in category_dict.keys()}
    all_unique_images = set()

    for d in anno:
        cat_id = d['category_id']
        img_id = d['image_id']
        image_counts[cat_id].add(img_id)
        all_unique_images.add(img_id)

    category_image_counts = {category_dict[cat_id]: len(img_ids) for cat_id, img_ids in image_counts.items()}
    sorted_images = dict(sorted(category_image_counts.items(), key=lambda item: item[1], reverse=True))
    # Add total unique images
    sorted_images["Total Unique Images"] = len(all_unique_images)
    return sorted_images

def save_results_to_excel(dict_pairs_list, file_name="output.xlsx", header_text=""):
    """
    Saves multiple sets of statistics (e.g., instance and image counts) into a single Excel file.

    Args:
        dict_pairs_list (list[dict[str, dict[str, int]]]): List of dictionaries, where each maps a header label 
            to a results dictionary (category name → value).
        file_name (str): Output Excel filename (with path).
        header_text (str): Optional header string to insert at the top of the Excel sheet.

    Returns:
        None: Writes an Excel file with the aggregated data.
    """
    dir_name = os.path.dirname(file_name)
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    ext = os.path.splitext(file_name)[1]

    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    version = 1
    new_file_name = file_name
    while os.path.exists(new_file_name):
        new_file_name = os.path.join(dir_name, f"{base_name}({version}){ext}")
        version += 1

    first_data_dict = dict_pairs_list[0]
    first_dict = next(iter(first_data_dict.values()))
    sorted_categories = list(first_dict.keys())

    columns = ["Category"] + [header for dict_pair in dict_pairs_list for header in dict_pair.keys()]

    data = []
    for category in sorted_categories:
        row = {"Category": category}
        for dict_pair in dict_pairs_list:
            for header, data_dict in dict_pair.items():
                row[header] = data_dict.get(category, 0)
        data.append(row)

    # Add a total row
    total_row = {"Category": "Total"}
    for dict_pair in dict_pairs_list:
        for header, data_dict in dict_pair.items():
            # Exclude "Total Unique Images" from summation
            sum_val = sum(v for k, v in data_dict.items() if k != 'Total Unique Images')
            total_row[header] = sum_val
    data.append(total_row)

    df = pd.DataFrame(data, columns=columns)

    with pd.ExcelWriter(new_file_name, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False, startrow=2)
        if header_text:
            workbook = writer.book
            worksheet = workbook['Results']
            worksheet.insert_rows(1)
            worksheet.cell(row=1, column=1, value=header_text)

    print(f"Results have been saved to {new_file_name}")

def main():
    coco_GT = COCO(path_anno)
    cats = coco_GT.loadCats(coco_GT.getCatIds())
    category_dict = {category['id']: category['name'] for category in cats}

    # Convert img_ids and PersonImgIds to sets for faster membership tests
    img_ids = set(get_used_img_id(img_path))
    PersonIds = coco_GT.getCatIds(catNms=['person'])
    PersonImgIds = set(coco_GT.getImgIds(catIds=PersonIds))

    # Extract all annotations once
    all_anns = list(coco_GT.anns.values())

    # Filter relevant annotations using comprehensions and sets
    relevant_anns = [v for v in all_anns if v['image_id'] in img_ids]
    org_anns = all_anns  # Just rename for clarity, no filtering needed
    anns_for_img_with_persons = [v for v in all_anns if v['image_id'] in PersonImgIds]

    # Remove unnecessary keys in a single pass per list
    keys_to_remove = {'segmentation', 'area', 'iscrowd'}

    def slim_down(ann_list):
        return [
            {key: val for key, val in d.items() if key not in keys_to_remove}
            for d in ann_list
        ]

    anno_slim = slim_down(relevant_anns)
    org_anno_slim = slim_down(org_anns)
    person_anno_slim = slim_down(anns_for_img_with_persons)

    # Count instances and images
    instances = get_instances_per_class(anno_slim, category_dict)
    images = get_images_per_class(anno_slim, category_dict)

    org_coco_instances = get_instances_per_class(org_anno_slim, category_dict)
    org_coco_images = get_images_per_class(org_anno_slim, category_dict)

    person_coco_instances = get_instances_per_class(person_anno_slim, category_dict)
    person_coco_images = get_images_per_class(person_anno_slim, category_dict)

    dict_pairs_list = [
        {'instances person coco': person_coco_instances,
         'images person coco': person_coco_images},
        {'instances coco': org_coco_instances,
         'images coco': org_coco_images},
        {'instances finetune': instances,
         'images finetune': images}
    ]

    save_results_to_excel(
        dict_pairs_list,
        file_name=result_path+"dataset_overview.xlsx",
        header_text="Source: " + img_path
    )

if __name__ == "__main__":
    main()