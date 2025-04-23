#!/bin/bash
##Slurm sbatch options
#SBATCH --gres=gpu:volta:1

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

DATASET=dior
LABELS_DIR=/home/gridsan/manderson/ovdsat/data/text/dior_labels.txt
BG_PROMPTS=/home/gridsan/manderson/ovdsat/data/text/background_prompts.txt

for backbone in clip-14-fmow-test #clip-14 openclip-14 remoteclip-14 georsclip-14 clip-14-gpte-1024-epoch50
do
    python text_prototypes.py \
        --save_dir run/text_prototypes/boxes/${DATASET} \
        --backbone_type ${backbone} \
        --labels_dir ${LABELS_DIR} \
        --bg_prompts ${BG_PROMPTS}
done