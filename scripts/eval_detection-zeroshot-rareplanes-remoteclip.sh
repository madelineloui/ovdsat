#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

DATA_DIR=data
dataset=rareplanes
prototype_type=text_prototypes #init_prototypes or train
finetune_type=boxes #masks or boxes
conf=0.5

for M in 1 2 3
do
    for backbone in remoteclip-14
    do
        python eval_detection.py \
            --dataset ${dataset} \
            --val_root_dir ${DATA_DIR}/${dataset}/PNGImages \
            --save_dir run/eval/detection/${dataset}/${prototype_type}/backbone_${backbone}_${finetune_type}/${conf}/zeroshot_${M} \
            --val_annotations_file ${DATA_DIR}/${dataset}/val_coco-${M}.json \
            --prototypes_path run/${prototype_type}/${finetune_type}/${dataset}/prototypes_${backbone}.pt \
            --bg_prototypes_path run/${prototype_type}/${finetune_type}/${dataset}/bg_prototypes_${backbone}.pt \
            --backbone_type ${backbone} \
            --classification box \
            --target_size 602 602 \
            --batch_size 128 \
            --num_workers 0 \
            --scale_factor 1 \
            --conf_thres ${conf} \
            --prototype_type ${prototype_type}
    done
done