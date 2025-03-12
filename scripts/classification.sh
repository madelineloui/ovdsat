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

python classification.py \
    --model-name ViT-L-14 \
    --dataset-root /home/gridsan/manderson/ovdsat/data/eurosat/EuroSAT \
    --backbone openclip \
    --batch-size 64 \
    --workers 8 \
    --output-dir /home/gridsan/manderson/ovdsat/run/eval/classification
    --clip-path /home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-fmow-4.pt
    #--num-shots