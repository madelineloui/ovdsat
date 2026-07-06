#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

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
aug=pixel

for backbone in dinov2
do
    for level in 0.9 0.7
    do
        for M in 1 2 3
        do
            for N in 10
            do
                python eval_detection.py \
                    --dataset ${dataset} \
                    --val_root_dir ${DATA_DIR}/${dataset}/JPEGImages \
                    --save_dir run/eval/detection/${dataset}/${finetune}/backbone_${backbone}_${finetune_type}/${aug}-${level}/N${N}-${M} \
                    --val_annotations_file ${DATA_DIR}/${dataset}/val_coco-${M}.json \
                    --prototypes_path run/${finetune}/${finetune_type}/${dataset}_N${N}-${M}-${aug}-${level}/prototypes_${backbone}.pt \
                    --bg_prototypes_path run/init_prototypes/boxes/${dataset}_N${N}-${M}-${aug}-${level}/bg_prototypes_${backbone}.pt \
                    --backbone_type ${backbone} \
                    --classification box \
                    --target_size 602 602 \
                    --batch_size 16 \
                    --num_workers 8 \
                    --scale_factor 1
            done
        done
    done
done
