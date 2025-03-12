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

python retrieval.py \
    --model-name ViT-L-14 \
    --retrieval-json-dir /home/gridsan/manderson/ovdsat/data/sydney_captions/dataset.json \
    --retrieval-images-dir /home/gridsan/manderson/ovdsat/data/sydney_captions/images \
    --backbone clip \
    --batch-size 64 \
    --workers 8
    #--clip-path /home/gridsan/manderson/train-CLIP/run/fmow/fmow-test-4.pth \