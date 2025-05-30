#!/bin/bash
#SBATCH --gres=gpu:volta:2

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

DATA_DIR=data
dataset=dior
finetune=init_prototypes #init_prototypes or train
finetune_type=boxes #masks or boxes
INIT_PROTOTYPES_PATH=run/init_prototypes
M=1

for backbone in longclip-14-fmow #clip-14-fmow-test #dinov2 clip-14 openclip-14 georsclip-14 remoteclip-14 clip-14-cap0 clip-14-cap1 clip-14-cap2 clip-14-gpt0-512-epoch23 clip-14-gpt0-512-epoch50 clip-14-gpte-512-epoch26 clip-14-gpte-512-epoch50 clip-14-gpt0-1024-epoch26 clip-14-gpt0-1024-epoch50 clip-14-gpte-1024-epoch24 clip-14-gpte-1024-epoch50
do
    for M in 1 2 3 4 5
    do
        for N in 5 10 30
        do
            python eval_detection.py \
                --dataset ${dataset} \
                --val_root_dir ${DATA_DIR}/${dataset}/JPEGImages \
                --save_dir run/eval/detection/${dataset}/${finetune}/backbone_${backbone}_${finetune_type}/N${N}-${M} \
                --val_annotations_file ${DATA_DIR}/${dataset}/val_coco-${M}.json \
                --prototypes_path run/${finetune}/${finetune_type}/${dataset}_N${N}-${M}/prototypes_${backbone}.pt \
                --bg_prototypes_path run/init_prototypes/boxes/${dataset}_N${N}-${M}/bg_prototypes_${backbone}.pt \
                --backbone_type ${backbone} \
                --classification box \
                --target_size 602 602 \
                --batch_size 16 \
                --num_workers 8 \
                --scale_factor 1
        done
    done
done