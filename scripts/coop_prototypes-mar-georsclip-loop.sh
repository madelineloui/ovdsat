#!/bin/bash
##Slurm sbatch options
#SBATCH --gres=gpu:volta:1

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate dassl

BACKBONE=georsclip-14
DATASET=mar
LABELS_DIR=/home/gridsan/manderson/ovdsat/data/text/${DATASET}_labels.txt
COOP_NAME=mar_georsclip_2
COOP_EPOCHS=100

for nctx in 8 16
do
    for ctp in middle
    do
        for N in 10
        do
            for seed in 1 2 3
            do
                python coop_prototypes.py \
                    --save_dir run/coop_prototypes/boxes/${DATASET}_N${N}-${seed}/${COOP_NAME}/nctx${nctx}_cscFalse_ctp${ctp} \
                    --backbone_type ${BACKBONE} \
                    --labels_dir ${LABELS_DIR} \
                    --ctx_path CoOp/output/${DATASET}/CoOp/${COOP_NAME}_${N}shots/nctx${nctx}_cscFalse_ctp${ctp}/seed${seed}/prompt_learner/model.pth.tar-${COOP_EPOCHS} \
                    --NCTX ${nctx} \
                    --CTP ${ctp}
            done
        done
    done
done