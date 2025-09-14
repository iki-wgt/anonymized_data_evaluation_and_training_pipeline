#!/bin/bash

# Define the path to your data configuration file
DATA_CONFIG_PATH="/data_private/ss24_weiss_privacy_dataset/config/train/fb_coco10.yaml"

# Optional parameters
EPOCHS=100
BATCH_SIZE=40
IMG_SIZE=640
OPTIMIZER="SGD"

# Array of model sizes
declare -a ModelSizes=("yolov10n.yaml" "yolov10s.yaml" "yolov10m.yaml" "yolov10l.yaml" "yolov10x.yaml")

# Loop through the model sizes and run the training script for each
for MODEL_SIZE in "${ModelSizes[@]}"
do
    echo "Starting training for model size: $MODEL_SIZE"
    python3 train.py "$DATA_CONFIG_PATH" --untrained_model=$MODEL_SIZE --epochs=$EPOCHS --batch_size=$BATCH_SIZE --img_size=$IMG_SIZE --optimizer=$OPTIMIZER --momentum=0.937 --weight_decay=0.0005
    echo "Training completed for $MODEL_SIZE"
done

echo "All models trained and saved successfully."