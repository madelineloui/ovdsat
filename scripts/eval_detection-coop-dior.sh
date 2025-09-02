#!/bin/bash
#SBATCH --gres=gpu:volta:2

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

DATA_DIR=data
dataset=dior
prototype_type=coop_prototypes
finetune_type=boxes
backbone=remoteclip-14
bg=no_bg #with_bg

for N in 5 10 30
do
    for M in 1 2 3 4 5
    do
        python eval_detection.py \
            --dataset ${dataset} \
            --val_root_dir ${DATA_DIR}/${dataset}/JPEGImages \
            --save_dir run/eval/detection/${dataset}/${prototype_type}/backbone_${backbone}_${finetune_type}/N${N}-${M} \
            --val_annotations_file ${DATA_DIR}/${dataset}/val_coco-${M}.json \
            --prototypes_path run/${prototype_type}/${finetune_type}/${dataset}_N${N}-${M}/prototypes_${backbone}.pt \
            --backbone_type ${backbone} \
            --classification box \
            --target_size 602 602 \
            --batch_size 16 \
            --num_workers 8 \
            --scale_factor 1 \
            --t
    done
done
#--bg_prototypes_path run/${prototype_type}/${finetune_type}/${bg}/${dataset}_N${N}-${M}/bg_prototypes_${backbone}.pt \