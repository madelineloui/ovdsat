#!/bin/bash
#SBATCH --gres=gpu:volta:2

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

PATH_CKPT_CLIP14=weights/clip-vit-large-patch14
PATH_CKPT_GEORSCLIP_14=weights/RS5M_ViT-L-14.pt
PATH_CKPT_REMOTECLIP_14=weights/RemoteCLIP-ViT-L-14.pt
PATH_CKPT_CLIP14_FMOW=/home/gridsan/manderson/train-CLIP/run/fmow/fmow-test-4.pth
PATH_CKPT_OPENCLIP14_FMOW=/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-fmow-4.pt

# for mode in linear_probe #zero_shot
# do
#     for seed in 2 3 4 5 #note that zero shot does not use seeds, just do once on test
#     do
#         for dataset in eurosat #aid patternnet resisc eurosat
#         do
#             for backbone in openclip openclip-fmow
#             do
#                 python classification.py \
#                     --model-name ViT-L-14 \
#                     --dataset ${dataset} \
#                     --backbone_name ${backbone} \
#                     --batch-size 64 \
#                     --workers 8 \
#                     --output-dir /home/gridsan/manderson/ovdsat/run/eval/classification \
#                     --seed ${seed} \
#                     --num-shots 1 2 4 8 16 \
#                     --mode ${mode} \
#             done
#         done
#     done
# done


for mode in linear_probe zero_shot
do
    for seed in 1 2 3 #note that zero shot does not use seeds, just do once on test
    do
        for dataset in eurosat #aid patternnet resisc eurosat
        do
            for backbone in clip
            do
                python classification.py \
                    --model-name ViT-L-14 \
                    --dataset ${dataset} \
                    --backbone_name ${backbone} \
                    --batch-size 64 \
                    --workers 8 \
                    --output-dir /home/gridsan/manderson/ovdsat/run/eval/classification \
                    --seed ${seed} \
                    --num-shots 1 2 4 8 16 \
                    --mode ${mode}
            done
        done
    done
done