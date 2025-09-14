"""
This module wraps Ultralytics YOLO object detection models for batch inference 
on COCO-style datasets. It performs detection, filters classes, and converts 
YOLO outputs to COCO format for downstream evaluation. The class also includes 
runtime logging and optional result visualization.
"""
from ultralytics import YOLO
from pycocotools.coco import COCO
import numpy as np
import json
import time
import os

from helper.utils import check_path
from helper.utils import detections_to_COCO

class YoloHandler():
    """
    Object detection using YOLO models (ultralytics) and conversion of detections to COCO format.
    """
    
    def __init__(self, model_name, anno_path) -> None:
        """
        Initialize YoloHandler object.

        Visualization is by default off.

        Arguments:
        - model_name (str): Name of the YOLO model.
        - anno_path (str): Path to the COCO annotations file.
        """

        # set model
        self.model = YOLO(model_name)  # load pretrained model
        
        # coco annotations for conversion
        self.coco_GT = COCO(anno_path)

        # detection param
        self.conf_thres = 0.25
        self.iou_thres = 0.7

        # visualization
        self.use_vis = False

    def detect(self, model_name, img_ids, img_paths, project_path, label_path, coco_path, class_ids):
        """
        Object detection on images using YOLO model (ultralytics) and save results.

        Arguments:
        - model_name (str): Name of the YOLO model.
        - img_ids (list): List of image IDs.
        - img_paths (list): List of paths to input images.
        - label_path (str): Path to save prediction results (txt per image with labels, bbox).
        - detection_path (str): Path to save coco-ready converted detections.
        - class_ids (list): List of class IDs.
        """

        # get wanted yolo ids from given coco ids
        yolo_classes = self.model.names
        yolo_ids = []

        coco_cats = self.coco_GT.loadCats(class_ids)
        for coco_cat in coco_cats:
            # coco_super_cat = coco_cat['supercategory']
            coco_id = coco_cat['id']
            coco_name = coco_cat['name']

            yolo_id = next((id for id, name in yolo_classes.items() if name == coco_name), None)
            if yolo_id is not None:
                yolo_ids.append(yolo_id)
            else:
                print(f"Dropping given COCO id {coco_id}-{coco_name} - No matching class in YOLO")

        # get start time
        start = time.time()

        # detect
        results = self.model.predict(
            source =img_paths,
            project = project_path,
            name = model_name,
            conf = self.conf_thres,
            iou = self.iou_thres,
            classes=yolo_ids,
            save = self.use_vis,
            save_txt = True,
            save_conf = True,
            stream=False)

        # get start time
        end = time.time()

        # save results in coco format
        self._save_as_coco(img_ids, label_path, coco_path)

        # save runtime
        runtime_path = f"{project_path}/runtime"
        if check_path(runtime_path):
            self._save_runtime(end-start,len(img_ids),runtime_path,model_name)

    def set_vis_flag(self, use_vis):
        """
        Set visualization flag.

        Arguments:
        - use_vis (bool): Flag to enable(true)/disable(false) visualization.
        """

        self.use_vis = use_vis

    def _save_as_coco(self, img_ids, label_path, detection_path):
        """
        Convert YOLO detections to COCO format and save as JSON.
        COCO format: [x_min, y_min, width_pix, height_pix]

        Arguments:
        - model (str): Name of the YOLO model.
        - img_ids (list): List of image IDs.
        - detection_path (str): Path to save detection results.
        """

        detections_coco = []
        
        for img_id in img_ids:

            base_name = os.path.basename(self.coco_GT.loadImgs(img_id)[0]['file_name'])  # e.g., "172889775563560901.png"
            img_name = int(os.path.splitext(base_name)[0])

            img_name = f"{img_id:012}"
            txt_path = f'{label_path}/{img_name}.txt'
            if not os.path.exists(txt_path):
                print(f'Could not find path {txt_path}')
                continue

            with open(txt_path, 'r') as f:
                lines = f.readlines()

            class_ids = []
            bboxes = []
            scores = []
            for line in lines:
                data = line.strip().split(' ')

                name = self.model.names[int(data[0])]
                coco_id = self.coco_GT.getCatIds(catNms=[name])[0]
                class_ids.append(coco_id)
                
                if name != self.coco_GT.loadCats(coco_id)[0]['name']:
                    print("!!!!!!!!WRONG CLASS MATCHING!!!!!!!")

                x_center = float(data[1])
                y_center = float(data[2])
                width = float(data[3])
                height = float(data[4])
                scores.append(float(data[5]))

                img_info = self.coco_GT.loadImgs(img_id)[0]
                image_width = img_info['width']
                image_height = img_info['height']

                x_min = int((x_center - (width / 2)) * image_width)
                y_min = int((y_center - (height / 2)) * image_height)
                width_pix = int(width * image_width)
                height_pix = int(height * image_height)

                bboxes.append([x_min, y_min, width_pix, height_pix])

            detections_coco.extend(detections_to_COCO(img_id,class_ids,bboxes,scores))
        
        #save coco styled json
        with open(detection_path, 'w') as json_file:
            json.dump(detections_coco, json_file)

    def _save_runtime(self,runtime_full, img_count,save_path,model_name):
        """
        Save runtime.

        Arguments:
        - runtime_full (float): Total runtime.
        - img_count (int): Number of images processed.
        - save_path (str): Path to save the runtime.
        """

        with open(f"{save_path}/{model_name}_runtime.txt", 'w') as file:     
            file.write(f'time = {runtime_full}\nimages = {img_count}')
