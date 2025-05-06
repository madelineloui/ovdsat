#!/bin/bash
##Slurm sbatch options
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

for seed in 4 5
do
    N=5
    python coop_prototypes.py \
        --save_dir run/coop_prototypes/boxes/${DATASET}_N${N}-${seed} \
        --backbone_type ${BACKBONE} \
        --labels_dir ${LABELS_DIR} \
        --ctx_path CoOp/output/${DATASET}/CoOp/vit_l14_${BACKBONE_SHORT}-ep100-ctx2_${N}shots/nctx4_cscFalse_ctpmiddle/seed${seed}/prompt_learner/model.pth.tar-100

    N=10
    python coop_prototypes.py \
        --save_dir run/coop_prototypes/boxes/${DATASET}_N${N}-${seed} \
        --backbone_type ${BACKBONE} \
        --labels_dir ${LABELS_DIR} \
        --ctx_path CoOp/output/dior/CoOp/vit_l14_${BACKBONE_SHORT}-ep200-ctx2_${N}shots/nctx4_cscFalse_ctpmiddle/seed${seed}/prompt_learner/model.pth.tar-200

    N=30
    python coop_prototypes.py \
        --save_dir run/coop_prototypes/boxes/${DATASET}_N${N}-${seed} \
        --backbone_type ${BACKBONE} \
        --labels_dir ${LABELS_DIR} \
        --ctx_path CoOp/output/dior/CoOp/vit_l14_${BACKBONE_SHORT}-ep300-ctx2_${N}shots/nctx4_cscFalse_ctpmiddle/seed${seed}/prompt_learner/model.pth.tar-300
done
