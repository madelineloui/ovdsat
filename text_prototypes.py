'''
1 - Provide directory with object examples and their segmentation masks
2 - Create model (CLIP / DINO) to extract features
3 - For each class, extract the name and features of the image masked by the segmentation mask
4 - Average the features over the patches that are not masked, which cointain the object
5 - Repeat until done for each object in each class separately
6 - For each class, stack, normalize and average the features of all objects in the class
7 - Save the features as a tensor and the class names as a list in a dictionary and save the prototype
'''

import os
import cv2
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
import os.path as osp
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torchvision import transforms
from transformers import CLIPModel
from torchvision import transforms
from argparse import ArgumentParser
from utils_dir.backbones_utils import load_backbone_and_tokenizer, extract_backbone_features, get_backbone_params
from utils_dir.coco_to_seg import coco_to_seg


def build_background_text_prototypes(args, tokenizer, model, device):
    '''
    Build zero-shot text prototypes by creating prompts and processing them with CLIP

    Args:
        args (argparse.Namespace): Input arguments
        tokenizer (TODO): Clip tokenizer
        model (torch.nn.Module): Backbone text encoder model
        device (str): Device to run the model on
    '''
    with open(args.bg_prompts, "r") as f:
        bg_prompts = [line.strip() for line in f]
    print(f'{len(bg_prompts)} background prompts found')

    if any(b in args.backbone_type for b in ('openclip', 'remoteclip', 'georsclip')):
        tokenized_bg_prompts = tokenizer(bg_prompts).to(device)
        bg_text_features = model.encode_text(tokenized_bg_prompts).to(device)
    else:
        tokenized_bg_prompts = tokenizer(bg_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        bg_text_features = model.get_text_features(**tokenized_bg_prompts).to(device)

    norm_bg_text_features = F.normalize(bg_text_features, p=2, dim=-1)
    bg_classes = ['bg_class_{}'.format(i+1) for i in range(len(bg_prompts))]

    category_dict = {
        'prototypes': norm_bg_text_features.cpu(),
        'label_names': bg_classes
    }
    
    prototype_shape = category_dict['prototypes'].shape
    print(f'Shape of text prototypes: {prototype_shape}')
    
    return category_dict

    
def build_text_prototypes(args, tokenizer, model, device):
    '''
    Build zero-shot text prototypes by creating prompts and processing them with CLIP

    Args:
        args (argparse.Namespace): Input arguments
        tokenizer (TODO): Clip tokenizer
        model (torch.nn.Module): Backbone text encoder model
        device (str): Device to run the model on
    '''
    
    # Read args.labels_dir
    with open(args.labels_dir, "r") as f:
        classes = [line.strip() for line in f]
    print(f'{len(classes)} class labels found')
        
    # Turn labels into prompts "a satellite image of a {label}"
    prompts = [f'a satellite image of a {c}' for c in classes]
    
    # Tokenize and extract text features
    if any(b in args.backbone_type for b in ('openclip', 'remoteclip', 'georsclip')):
        tokenized_prompts = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokenized_prompts).to(device)
    else:
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = model.get_text_features(**tokenized_prompts).to(device)
        # Pass through projection layer to match image embedding dimension
        text_features = model.text_projection(text_features)
    
    # Normalize
    norm_text_features = F.normalize(text_features, p=2, dim=-1)

    category_dict = {
        'prototypes': norm_text_features.cpu(),
        'label_names': classes
    }
    
    prototype_shape = category_dict['prototypes'].shape
    print(f'Shape of text prototypes: {prototype_shape}')
    
    return category_dict

def main(args):
    '''
    Main function to build object and background prototypes.

    Args:
        args (argparse.Namespace): Input arguments
    '''

    print('\nBuilding text prototypes...')
    print(f'Loading model: {args.backbone_type}...')
    print(f'Using labels in {args.labels_dir}')
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_backbone_and_tokenizer(args.backbone_type)
    model = model.to(device)
    model.eval()
    
    # Build text prototypes
    obj_category_dict = build_text_prototypes(args, tokenizer, model, device)
    
    # Create save directory if it does not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Build background prototypes if specified
    if args.bg_prompts is not None:
        bg_category_dict = build_background_text_prototypes(args, tokenizer, model, device)
        save_name = f'bg_prototypes_{args.backbone_type}.pt'
        torch.save(bg_category_dict, os.path.join(args.save_dir, save_name))
        print(f'Saved background prototypes to {os.path.join(args.save_dir, save_name)}')

    save_name = f'prototypes_{args.backbone_type}.pt'
    torch.save(obj_category_dict, os.path.join(args.save_dir, save_name))
    print(f'Saved prototypes to {os.path.join(args.save_dir, save_name)}')


if __name__ == '__main__':
    parser = ArgumentParser()
    #parser.add_argument('--data_dir', type=str, default='data/simd_subset_10')
    parser.add_argument('--save_dir', type=str, default='run/text_prototypes/boxes/dior')
    #parser.add_argument('--annotations_file', type=str, default='/mnt/ddisk/boux/code/data/simd/train_coco_subset_N10.json')
    parser.add_argument('--backbone_type', type=str, default='clip-14')
    parser.add_argument('--labels_dir', type=str, default='/home/gridsan/manderson/ovdsat/data/text/dior_labels.txt')
    #parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(602, 602))
    #parser.add_argument('--window_size', type=int, default=224)
    #parser.add_argument('--scale_factor', type=int, default=1)
    #parser.add_argument('--num_b', type=int, default=10, help='Number of background samples to extract per image')
    #parser.add_argument('--k', type=int, default=200, help='Number of background prototypes (clusters for k-means)')
    #parser.add_argument('--store_bg_prototypes', action='store_true', default=False)
    parser.add_argument('--bg_prompts', type=str, default=None)
    args = parser.parse_args()

    main(args)
