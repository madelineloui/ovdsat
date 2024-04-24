#!/bin/bash

DATA_DIR=/mnt/ddisk/boux/code/data
INIT_PROTOTYPES_PATH=run/init_prototypes_obbs
backbone=dinov2

for dataset in dior
do
    #for N in 5 10 30
    for N in 10
    do
        echo "Training box classifier model for the ${dataset} dataset using ${backbone} features with N=${N}"
        python train.py \
            --train_root_dir  ${DATA_DIR}/${dataset}/train \
            --val_root_dir  ${DATA_DIR}/${dataset}/train \
            --save_dir "run/train/obbs/${dataset}_N${N}" \
            --train_annotations_file ${DATA_DIR}/${dataset}/train_coco_subset_N${N}_with_obbs.json \
            --val_annotations_file ${DATA_DIR}/${dataset}/train_coco_finetune_val_with_obbs.json \
            --prototypes_path ${INIT_PROTOTYPES_PATH}/${dataset}_N${N}/prototypes_${backbone}.pt \
            --backbone_type ${backbone} \
            --num_epochs 200 \
            --lr 2e-4 \
            --target_size 602 602 \
            --batch_size 4 \
            --num_neg 8 \
            --num_workers 1 \
            --iou_thr 0.1 \
            --conf_thres 0.2 \
            --scale_factor 1 \
            --annotations obb \
            --only_train_prototypes
    done
done