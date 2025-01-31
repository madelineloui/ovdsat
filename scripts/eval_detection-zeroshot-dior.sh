#!/bin/bash
#SBATCH --gres=gpu:volta:2

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

DATA_DIR=data
dataset=dior
prototype_type=text_prototypes #init_prototypes or train
finetune_type=boxes #masks or boxes
M=1

for backbone in clip-14-gpte-1024-epoch50 #openclip-14 clip-14 remoteclip-14 georsclip-14 clip-14-gpte-1024-epoch50
do
    python eval_detection.py \
        --dataset ${dataset} \
        --val_root_dir ${DATA_DIR}/${dataset}/JPEGImages \
        --save_dir run/eval/detection/${dataset}/${prototype_type}/backbone_${backbone}_${finetune_type}/zeroshot_${M} \
        --val_annotations_file ${DATA_DIR}/${dataset}/val_coco-${M}.json \
        --prototypes_path run/${prototype_type}/${finetune_type}/${dataset}/prototypes_${backbone}.pt \
        --bg_prototypes_path run/${prototype_type}/${finetune_type}/${dataset}/bg_prototypes_${backbone}.pt \
        --backbone_type ${backbone} \
        --classification box \
        --target_size 602 602 \
        --batch_size 16 \
        --num_workers 8 \
        --scale_factor 1 \
        --t
done
