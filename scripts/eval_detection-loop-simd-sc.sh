#!/bin/bash
#SBATCH --gres=gpu:volta:2

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

DATA_DIR=data
dataset=simd
finetune=init_prototypes #init_prototypes or train
finetune_type=boxes #masks or boxes
INIT_PROTOTYPES_PATH=run/init_prototypes
sc=1

for backbone in clip-14-gpt0 #openclip-14 clip-14 clip-14-cap0 clip-14-cap1 clip-14-cap2
do
    for N in 10
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
done
