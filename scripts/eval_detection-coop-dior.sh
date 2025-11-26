#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

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

for N in 1 3 5 10 30
do
    for M in 1
    do
        python -u eval_detection.py \
            --dataset ${dataset} \
            --val_root_dir ${DATA_DIR}/${dataset}/JPEGImages \
            --val_annotations_file ${DATA_DIR}/${dataset}/val_coco-${M}.json \
            --save_dir run/eval/detection/${dataset}/${prototype_type}/backbone_${backbone}_${finetune_type}/test50samples/N${N}-${M} \
            --prototypes_path run/${prototype_type}/${finetune_type}/${dataset}_N${N}-${M}/prototypes_${backbone}.pt \
            --backbone_type ${backbone} \
            --classification box \
            --target_size 602 602 \
            --batch_size 128 \
            --num_workers 0 \
            --scale_factor 1 \
            --t
    done
done

# Batch size can be 128
#--bg_prototypes_path run/${prototype_type}/${finetune_type}/${bg}/${dataset}_N${N}-${M}/bg_prototypes_${backbone}.pt \
#--val_annotations_file ${DATA_DIR}/${dataset}/val_coco-${M}.json \
#--val_annotations_file ${DATA_DIR}/${dataset}/val_coco_N${N}-${M}.json \
#--val_annotations_file ${DATA_DIR}/${dataset}/train_coco_subset_N${N}-${M}.json \
# train_all_annotations_N${N}-${M}.json