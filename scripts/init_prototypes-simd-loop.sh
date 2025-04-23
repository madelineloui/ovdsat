#!/bin/bash
##Slurm sbatch options
#SBATCH --gres=gpu:volta:1

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

DATA_DIR=data

for backbone in openclip-14-fmow #dinov2 clip-14 openclip-14 georsclip-14 remoteclip-14 clip-14-cap0 clip-14-cap1 clip-14-cap2 clip-14-gpt0-512-epoch23 clip-14-gpt0-512-epoch50 clip-14-gpte-512-epoch26 clip-14-gpte-512-epoch50 clip-14-gpt0-1024-epoch26 clip-14-gpt0-1024-epoch50 clip-14-gpte-1024-epoch24 clip-14-gpte-1024-epoch50
do
    for dataset in simd
    do
        for N in 5 10 30
        do
            echo "Creating prototypes for the ${dataset} dataset using ${backbone} features with N=${N}"

            python build_prototypes.py \
                --data_dir ${DATA_DIR}/${dataset}/train \
                --save_dir run/init_prototypes/boxes/${dataset}_N${N} \
                --annotations_file ${DATA_DIR}/${dataset}/train_coco_subset_N${N}.json \
                --backbone_type ${backbone} \
                --target_size 602 602 \
                --window_size 224 \
                --scale_factor 1 \
                --num_b 10 \
                --k 200 \
                --store_bg_prototypes
        done
    done
done
