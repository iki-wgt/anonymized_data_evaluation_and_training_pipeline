#!/bin/bash

# Define the path to your data configuration file
MODE="freeze"
DATA_CONFIG_PATH="path_to_repo/config/eval/${MODE}_on_org.yaml" # edit if needed

# Array of model sizes
declare -a MODELS=("nothing" "back" "back_no_psa" "neck_head_psa" "neck" "neck_head")


# Loop through the model sizes and run the training script for each
for MODEL in "${MODELS[@]}"
do
    echo "Starting evaluation for model : $MODEL"
    python3 run_eval.py -config=$DATA_CONFIG_PATH -net=$MODE"_"$MODEL
    echo "Evaluation completed for $MODEL_SIZE"
done

echo "All models evaluated and saved successfully."
