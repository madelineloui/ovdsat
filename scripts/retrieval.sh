#!/bin/bash
#SBATCH --gres=gpu:volta:2

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

for dataset in sydney rsitmd
do
    for backbone in clip openclip remoteclip georsclip clip-fmow openclip-fmow
    do
        python retrieval.py \
            --model-name ViT-L-14 \
            --dataset ${dataset} \
            --backbone_name ${backbone} \
            --batch-size 64 \
            --workers 8 \
            --output-dir /home/gridsan/manderson/ovdsat/run/eval/retrieval
    done
done