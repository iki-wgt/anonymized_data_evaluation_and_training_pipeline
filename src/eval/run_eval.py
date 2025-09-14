"""
Script for performing object detection using YOLO models (Ultralytics) and 
evaluating detection performance using COCO metrics. The script loads a 
configuration file, runs detection over a dataset, and computes evaluation 
scores including mAP, precision, recall, and F-score variants.

 Workflow:
     1. Parses command-line arguments for config file and model name.
     2. Runs inference using a YOLO model on images specified in the config.
     3. Saves detections in YOLO and COCO format.
     4. Evaluates the predictions using pycocotools COCO evaluation API.
"""
import argparse

from helper.Dataset import Dataset
from helper.YoloHandler import YoloHandler
from helper.CoCoEval import CocoEvaluation

from pycocotools.coco import COCO

def yolo_detector(model,path_to_config, use_vis):
    """
    Perform object detection using YOLO model.

    Arguments:
    - model (str): Name of the YOLO model.
    - path_to_config (str): Path to the configuration file incl. file name.
    - use_vis (bool): Flag, if true saves images for detected objects.
    """

    # Setup
    data = Dataset(path_to_config)
    model_name = "last.pt"
    model_path = data.get_model_path()+"/"+model+"/weights/"+model_name
    print(f"eval of model: {model_path}")
    
    # YOLO
    yolo = YoloHandler(model_path, data.get_anno_path())
    yolo.set_vis_flag(use_vis)
    yolo.detect(
        model_name=model,
        img_ids=data.get_img_ids(),
        img_paths=data.get_img_path_glob(),
        project_path=data.get_scene_path(model),
        label_path=data.get_label_save_path(model),
        coco_path=data.get_coco_detections_path(model),
        class_ids=data.get_cat_ids())

def parse_script_arguments():
    """
    Parse command-line arguments.

    Returns:
        tuple: A tuple containing the paths to the configuration file, network to use and visualization flag
    """

    parser = argparse.ArgumentParser(
    description="Object Detection for Yolo models"
    )
    parser.add_argument("-config", required=True, type=str, help="Path to config file incl. config file name")
    parser.add_argument("-net", required=True, type=str, help="Yolo type used for detection.")
    args = parser.parse_args()

    return args.config, args.net

def main():
    """
    Main function to execute object detection and evaluation.
    Parses command-line arguments, validates network choice, performs detection and evaluation.
    """

    path_to_config,netmodel = parse_script_arguments()

    # Detect
    print(f"Starting detection with {netmodel}...")

    yolo_detector(netmodel,path_to_config,use_vis=False)

    print(f"Run of {netmodel} finished")

    # Evaluate
    print(f"Starting evaluation of {netmodel} results...")

    coco_eval = CocoEvaluation(path_to_config,netmodel)
    coco_eval.eval()

    print("--- Done ---")

if __name__ == "__main__":
    main()
