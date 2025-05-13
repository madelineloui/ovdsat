#!/bin/bash
#SBATCH --gres=gpu:volta:2

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

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
            for backbone in longclip-fmow
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