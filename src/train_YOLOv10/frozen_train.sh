#!/bin/bash
# Define the path to your data configuration file
MODEL_SIZE="yolov10m.yaml"
MODEL_PATH="path_to_repo/src/train_YOLOv10/runs/detect/train/weights/last.pt"
DATA_CONFIG_PATH="path_to_repo/config/train/tune_coco10.yaml"

#Layer definitions
# BACKBONE=(0 1 2 3 4 5 6 7 8 9 10)
# BACKBONE_WITHOUT_PSA=(0 1 2 3 4 5 6 7 8 9)

# NECK=(11 12 13 14 15 16 17 18 19 20 21 22)
# NECK_WITH_HEAD=(11 12 13 14 15 16 17 18 19 20 21 22 23)

# NECK_HEAD_PSA=(10 11 12 13 14 15 16 17 18 19 20 21 22 23)

# Optional parameters
EPOCHS=30
BATCH_SIZE=40
IMG_SIZE=640
OPTIMIZER="SGD"
LR0= 0.001
LRF= 0.01
MOMENTUM=0.85
WEIGHT_DECAY=0.0007
WARMUP_EPOCHS=4
WARMUP_MOMENTUM=0.8

echo "Pt. 1/6 - Controll Group - Starting finetuning wihtout freeze for model size: $MODEL_SIZE"
python3 train.py "$DATA_CONFIG_PATH" \
    --model_path=$MODEL_PATH \
    --untrained_model=$MODEL_SIZE \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --img_size=$IMG_SIZE \
    --optimizer=$OPTIMIZER \
    --lr0=$LR0 \
    --lrf=$LRF \
    --momentum=$MOMENTUM \
    --weight_decay=$WEIGHT_DECAY \
    --warmup_epochs=$WARMUP_EPOCHS \
    --warmup_momentum=$WARMUP_MOMENTUM
echo "Pt. 1/6 - Controll Group - Training completed"

echo "Pt. 2/6 - Starting finetuning BACKBONE for model size: $MODEL_SIZE"
IDX_TO_FREEZE=(0 1 2 3 4 5 6 7 8 9 10)
# Convert the array to a string that represents a Python list
IDX_TO_FREEZE_STR=$(printf ",%s" "${IDX_TO_FREEZE[@]}")
IDX_TO_FREEZE_STR="[${IDX_TO_FREEZE_STR:1}]"
python3 train.py "$DATA_CONFIG_PATH" \
    --model_path=$MODEL_PATH \
    --untrained_model=$MODEL_SIZE \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --img_size=$IMG_SIZE \
    --optimizer=$OPTIMIZER \
    --lr0=$LR0 \
    --lrf=$LRF \
    --momentum=$MOMENTUM \
    --weight_decay=$WEIGHT_DECAY \
    --warmup_epochs=$WARMUP_EPOCHS \
    --warmup_momentum=$WARMUP_MOMENTUM \
    --idx_to_freeze="$IDX_TO_FREEZE_STR"
echo "Pt. 2/6 - Training completed"

echo "Pt. 3/6 - Starting finetuning NECK for model size: $MODEL_SIZE"
IDX_TO_FREEZE=(11 12 13 14 15 16 17 18 19 20 21 22)
# Convert the array to a string that represents a Python list
IDX_TO_FREEZE_STR=$(printf ",%s" "${IDX_TO_FREEZE[@]}")
IDX_TO_FREEZE_STR="[${IDX_TO_FREEZE_STR:1}]"
python3 train.py "$DATA_CONFIG_PATH" \
    --model_path=$MODEL_PATH \
    --untrained_model=$MODEL_SIZE \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --img_size=$IMG_SIZE \
    --optimizer=$OPTIMIZER \
    --lr0=$LR0 \
    --lrf=$LRF \
    --momentum=$MOMENTUM \
    --weight_decay=$WEIGHT_DECAY \
    --warmup_epochs=$WARMUP_EPOCHS \
    --warmup_momentum=$WARMUP_MOMENTUM \
    --idx_to_freeze="$IDX_TO_FREEZE_STR"
echo "Pt. 3/6 - Training completed"

echo "Pt. 4/6 - Starting finetuning NECK_WITH_HEAD for model size: $MODEL_SIZE"
IDX_TO_FREEZE=(11 12 13 14 15 16 17 18 19 20 21 22 23)
# Convert the array to a string that represents a Python list
IDX_TO_FREEZE_STR=$(printf ",%s" "${IDX_TO_FREEZE[@]}")
IDX_TO_FREEZE_STR="[${IDX_TO_FREEZE_STR:1}]"
python3 train.py "$DATA_CONFIG_PATH" \
    --model_path=$MODEL_PATH \
    --untrained_model=$MODEL_SIZE \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --img_size=$IMG_SIZE \
    --optimizer=$OPTIMIZER \
    --lr0=$LR0 \
    --lrf=$LRF \
    --momentum=$MOMENTUM \
    --weight_decay=$WEIGHT_DECAY \
    --warmup_epochs=$WARMUP_EPOCHS \
    --warmup_momentum=$WARMUP_MOMENTUM \
    --idx_to_freeze="$IDX_TO_FREEZE_STR"
echo "Pt. 4/6 - Training completed"

echo "Pt. 5/6 - Starting finetuning BACKBONE_WITHOUT_PSA for model size: $MODEL_SIZE"
IDX_TO_FREEZE=(0 1 2 3 4 5 6 7 8 9)
# Convert the array to a string that represents a Python list
IDX_TO_FREEZE_STR=$(printf ",%s" "${IDX_TO_FREEZE[@]}")
IDX_TO_FREEZE_STR="[${IDX_TO_FREEZE_STR:1}]"
python3 train.py "$DATA_CONFIG_PATH" \
    --model_path=$MODEL_PATH \
    --untrained_model=$MODEL_SIZE \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --img_size=$IMG_SIZE \
    --optimizer=$OPTIMIZER \
    --lr0=$LR0 \
    --lrf=$LRF \
    --momentum=$MOMENTUM \
    --weight_decay=$WEIGHT_DECAY \
    --warmup_epochs=$WARMUP_EPOCHS \
    --warmup_momentum=$WARMUP_MOMENTUM \
    --idx_to_freeze="$IDX_TO_FREEZE_STR"
echo "Pt. 5/6 - Training completed"

echo "Pt. 6/6 - Starting finetuning NECK_HEAD_PSA for model size: $MODEL_SIZE"
IDX_TO_FREEZE=(10 11 12 13 14 15 16 17 18 19 20 21 22 23)
# Convert the array to a string that represents a Python list
IDX_TO_FREEZE_STR=$(printf ",%s" "${IDX_TO_FREEZE[@]}")
IDX_TO_FREEZE_STR="[${IDX_TO_FREEZE_STR:1}]"
python3 train.py "$DATA_CONFIG_PATH" \
    --model_path=$MODEL_PATH \
    --untrained_model=$MODEL_SIZE \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --img_size=$IMG_SIZE \
    --optimizer=$OPTIMIZER \
    --lr0=$LR0 \
    --lrf=$LRF \
    --momentum=$MOMENTUM \
    --weight_decay=$WEIGHT_DECAY \
    --warmup_epochs=$WARMUP_EPOCHS \
    --warmup_momentum=$WARMUP_MOMENTUM \
    --idx_to_freeze="$IDX_TO_FREEZE_STR"
echo "Pt. 6/6 - Training completed"

echo "All models trained and saved successfully."