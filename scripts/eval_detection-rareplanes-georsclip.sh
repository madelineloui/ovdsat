#!/bin/bash
#SBATCH --gres=gpu:volta:2

# Loading the required modules
source /etc/profile
module load anaconda/2022a

# Activate conda env
source activate ovdsat

DATA_DIR=data
dataset=rareplanes
finetune=init_prototypes #init_prototypes or train
finetune_type=boxes #masks or boxes
INIT_PROTOTYPES_PATH=run/init_prototypes

for backbone in georsclip-14
do
    for M in 1 2 3
    do
        for N in 1 5 10
        do
            python eval_detection-sample.py \
                --dataset ${dataset} \
                --val_root_dir ${DATA_DIR}/${dataset}/PNGImages \
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
