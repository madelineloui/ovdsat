#!/bin/bash
# SBATCH --gres=gpu:volta:2
# SBATCH --mem=100G

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate dynnet

N=5
M=1
backbone_type=resnet50
num_classes=7
segmodel_type=unet
split_dir=/home/gridsan/manderson/ovdsat/data/monthly_dynnet/splits
epochs=50
lr=0.001
batch_size=32
crop_size=512
num_workers=8
model_dir=/home/gridsan/manderson/ovdsat/run/seg

python train_seg.py \
    --backbone_type ${backbone_type} \
    --num_classes ${num_classes} \
    --segmodel_type ${segmodel_type} \
    --train_split ${split_dir}/train_split_${N}-${M}.txt \
    --val_split ${split_dir}/val_split_${N}-${M}.txt \
    --test_split ${split_dir}/test.txt \
    --epochs ${epochs} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --crop_size ${crop_size} \
    --num_workers ${num_workers} \
    --model_dir ${model_dir} \
    --exp_name ${backbone_type}-${segmodel_type}-${N}-${M}
