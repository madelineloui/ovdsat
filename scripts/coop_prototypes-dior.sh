#!/bin/bash
# #Slurm sbatch options
#SBATCH --gres=gpu:volta:1

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate dassl

BACKBONE=remoteclip-14
BACKBONE_SHORT=remote
DATASET=dior
LABELS_DIR=/home/gridsan/manderson/ovdsat/data/text/${DATASET}_labels-coop.txt
COOP_NAME=test8
COOP_EPOCHS=50

for N in 3
do
    for seed in 1
    do
        python coop_prototypes.py \
            --save_dir run/coop_prototypes/boxes/${DATASET}_N${N}-${seed} \
            --backbone_type ${BACKBONE} \
            --labels_dir ${LABELS_DIR} \
            --ctx_path CoOp/output/dior/CoOp/${COOP_NAME}_${N}shots/nctx4_cscFalse_ctpmiddle/seed${seed}/prompt_learner/model.pth.tar-${COOP_EPOCHS}
    done
done