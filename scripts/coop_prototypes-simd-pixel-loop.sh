#!/bin/bash
##Slurm sbatch options
#SBATCH --gres=gpu:volta:1

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate dassl

DATASET=simd
LABELS_DIR=/home/gridsan/manderson/ovdsat/data/text/${DATASET}_labels.txt
AUG=pixel
COOP_CONFIG=2
COOP_EPOCHS=100
nctx=4
ctp=end

for SHORT_NAME in clip remoteclip georsclip
do
    for LEVEL in 0.9 0.8 0.7
    do
        for N in 10
        do
            for seed in 1 2 3
            do
                BACKBONE=${SHORT_NAME}-14
                COOP_NAME_SHORT=${DATASET}_${SHORT_NAME}_${COOP_CONFIG}
                COOP_NAME=${COOP_NAME_SHORT}-${AUG}-${LEVEL}
                python coop_prototypes.py \
                    --save_dir run/coop_prototypes/boxes/${DATASET}_N${N}-${seed}-${AUG}-${LEVEL}/${COOP_NAME_SHORT}/nctx${nctx}_cscFalse_ctp${ctp} \
                    --backbone_type ${BACKBONE} \
                    --labels_dir ${LABELS_DIR} \
                    --ctx_path CoOp/output/${DATASET}/CoOp/${COOP_NAME}_${N}shots/nctx${nctx}_cscFalse_ctp${ctp}/seed${seed}/prompt_learner/model.pth.tar-${COOP_EPOCHS} \
                    --NCTX ${nctx} \
                    --CTP ${ctp}
            done
        done
    done
done
