#!/bin/bash
##Slurm sbatch options
#SBATCH --gres=gpu:volta:1

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

DATA_DIR=data
aug=cover

for backbone in dinov2
do
    for dataset in simd
    do
        for level in 0.05 0.15
        do
            for M in 1 2 3
            do
                for N in 10
                do
                    echo "Creating prototypes for the ${dataset} dataset using ${backbone} features with N=${N}"

                    python build_prototypes.py \
                        --data_dir ${DATA_DIR}/${dataset}/JPEGImages \
                        --save_dir run/init_prototypes/boxes/${dataset}_N${N}-${M}-${aug}-${level} \
                        --annotations_file ${DATA_DIR}/${dataset}/train_coco_subset_N${N}-${M}.json \
                        --backbone_type ${backbone} \
                        --target_size 602 602 \
                        --window_size 224 \
                        --scale_factor 1 \
                        --num_b 10 \
                        --k 200 \
                        --store_bg_prototypes \
                        --crop_path data/cropped_data/${dataset}/train/${dataset}_N${N}-${M}-${aug}-${level}
                done
            done
        done
    done
done
#--store_bg_prototypes \
