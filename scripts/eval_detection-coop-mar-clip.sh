#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

DATA_DIR=data
dataset=mar
prototype_type=coop_prototypes
finetune_type=boxes
backbone=clip-14
COOP_NAME=mar_clip_2

for N in 1 3 5 10 30
do
    for M in 1 2 3
    do
        python -u eval_detection.py \
            --dataset ${dataset} \
            --val_root_dir ${DATA_DIR}/${dataset}/JPEGImages \
            --val_annotations_file ${DATA_DIR}/${dataset}/val_coco-${M}.json \
            --save_dir run/eval/detection/${dataset}/${prototype_type}/backbone_${backbone}_${finetune_type}/${COOP_NAME}/N${N}-${M} \
            --prototypes_path run/${prototype_type}/${finetune_type}/${dataset}_N${N}-${M}/${COOP_NAME}/prototypes_${backbone}.pt \
            --backbone_type ${backbone} \
            --classification box \
            --target_size 602 602 \
            --batch_size 128 \
            --num_workers 0 \
            --scale_factor 1 \
            --conf_thres 0.5 \
            --prototype_type ${prototype_type}
    done
done