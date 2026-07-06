#!/bin/bash
##Slurm sbatch options
#SBATCH --gres=gpu:volta:1

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

DATA_DIR=data
M=1

for backbone in dinov2
do
    for dataset in nwpu
    do
        for M in 1 2 3
        do
            for N in 1 3 5 10 30
            do
                echo "Creating prototypes for the ${dataset} dataset using ${backbone} features with N=${N}"

                python build_prototypes.py \
                    --data_dir ${DATA_DIR}/${dataset}/positive_image_set \
                    --save_dir run/init_prototypes/boxes/${dataset}_N${N}-${M} \
                    --annotations_file ${DATA_DIR}/${dataset}/train_coco_subset_N${N}-${M}.json \
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
done
