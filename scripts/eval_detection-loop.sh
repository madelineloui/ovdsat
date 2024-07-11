#!/bin/bash

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate test

DATA_DIR=data
backbone=dinov2 #dinov2 georsclip-14 remoteclip-14 openclip-14 clip-14
dataset=simd
finetune=init_prototypes #init_prototypes or train
finetune_type=boxes #masks or boxes
N=10
INIT_PROTOTYPES_PATH=run/init_prototypes
sc=True

for backbone in openclip-14 georsclip-14 remoteclip-14 dinov2 clip-14
do
    python eval_detection.py \
        --dataset ${dataset} \
        --val_root_dir ${DATA_DIR}/${dataset}/val \
        --save_dir run/eval/detection/${dataset}/${finetune}/backbone_${backbone}_${finetune_type}/N${N} \
        --val_annotations_file ${DATA_DIR}/${dataset}/val_coco.json \
        --prototypes_path run/${finetune}/${finetune_type}/${dataset}_N${N}/prototypes_${backbone}.pt \
        --bg_prototypes_path run/init_prototypes/boxes/${dataset}_N${N}/bg_prototypes_${backbone}.pt \
        --backbone_type ${backbone} \
        --classification box \
        --target_size 602 602 \
        --batch_size 16 \
        --num_workers 8 \
        --scale_factor 1 \
        --sc ${sc}
done