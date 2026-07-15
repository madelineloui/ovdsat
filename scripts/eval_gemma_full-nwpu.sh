#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate mllm

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR=data
dataset=nwpu

model_name=gemma-4-E4B-it
MODEL_PATH=/home/gridsan/manderson/ovdsat/weights/${model_name}

for M in 2 3
do
    python -u eval_gemma_detection.py \
        --dataset ${dataset} \
        --val_root_dir ${DATA_DIR}/${dataset}/positive_image_set \
        --val_annotations_file ${DATA_DIR}/${dataset}/val_coco-${M}.json \
        --model_path ${MODEL_PATH} \
        --save_dir run/eval/detection/${dataset}/gemma/${model_name}/M${M} \
        --max_new_tokens 512
done