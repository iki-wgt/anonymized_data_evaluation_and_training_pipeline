#!/bin/bash

# Define the path to your data configuration file
MODE="org" # can be "fb", "face", "org
DATA_CONFIG_PATH="path_to_repo/config/eval/${MODE}_on_org_coco.yaml"


# Array of model sizes
declare -a ModelSizes=("yolov10n" "yolov10s" "yolov10m" "yolov10l" "yolov10x")


# Loop through the model sizes and run the training script for each
for MODEL_SIZE in "${ModelSizes[@]}"
do
    echo "Starting evaluation for model size: $MODEL_SIZE"
    python3 run_eval.py -config=$DATA_CONFIG_PATH -net=$MODE"_"$MODEL_SIZE
    echo "Evaluation completed for $MODEL_SIZE"
done

echo "All models evaluated and saved successfully."