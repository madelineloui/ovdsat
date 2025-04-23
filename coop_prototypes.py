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
    Can use with CoOp?

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
    elif 'customclip' in args.backbone_type:
        tokenized_bg_prompts = tokenizer(bg_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        bg_text_features = model.encode_text(tokenized_bg_prompts).to(device)
        #print(bg_text_features.shape)
        #print(torch.mean(bg_text_features))
        #bg_text_features = F.normalize(model.encode_text(tokenized_bg_prompts), dim=1).to(device)
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

    
def build_coop_prototypes(args, model, device):
    '''
    Build zero-shot text prototypes by creating prompts and processing them with CLIP

    Args:
        args (argparse.Namespace): Input arguments
        tokenizer (TODO): Clip tokenizer
        model (torch.nn.Module): Backbone text encoder model
        device (str): Device to run the model on
    '''
    
    # note this is only working on cpu currently
        
    # Read args.labels_dir
    with open(args.labels_dir, "r") as f:
        classes = [line.strip() for line in f]
    print(f'{len(classes)} class labels found')
    
    model, tokenizer = load_backbone_and_tokenizer(args.backbone_type)
    print('model device:', next(model.parameters()).device)
    
    context = torch.load(args.ctx_path, map_location=torch.device('cpu') )
    
    prefix = context['state_dict']['token_prefix']
    ctx = context['state_dict']['ctx']
    suffix = context['state_dict']['token_suffix']
    
    if len(ctx.shape) < 3: #unified context, dupicate n_cls times
        n_cls = len(prefix)
        ctx = ctx.unsqueeze(0)
        ctx = ctx.expand(n_cls, -1, -1) 
    
    # print(prefix.shape)
    # print(ctx.shape)
    # print(suffix.shape)
    
    prompts = torch.cat(
        [
            prefix,  # (n_cls, 1, dim)
            ctx,     # (n_cls, n_ctx, dim)
            suffix,  # (n_cls, *, dim)
        ],
        dim=1,
    )
    
    # Reorder for inference
    # desired_order = [
    #     "groundtrackfield", "baseballfield", "bridge", "Expressway-toll-station", "vehicle",
    #     "airplane", "airport", "tenniscourt", "trainstation", "storagetank", "stadium", "windmill",
    #     "ship", "golffield", "overpass", "chimney", "dam", "basketballcourt", "harbor", "Expressway-Service-area"
    # ]
    desired_order = classes
    alphabetical_order = sorted(desired_order)
    name_to_idx = {name: i for i, name in enumerate(alphabetical_order)}
    reorder_indices = [name_to_idx[name] for name in desired_order]
    prompts = prompts[reorder_indices]
    
    print('prompts shape:', prompts.shape)
    prompts = prompts.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    
#     text_encoder = model.transformer
#     text_feats = text_encoder(prompts)
    
#     text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
#     text_feats = text_feats[:, 0, :] #CLS token
#     text_feats.shape # Final shape of text prototypes should be [n_classes, dim=768]

    with torch.no_grad():
        text_feats = model.transformer(prompts)  # (n_cls, ctx_len, embed_dim)

    text_feats = F.normalize(text_feats[:, 0, :], dim=-1)

    category_dict = {
        'prototypes': text_feats.cpu(),
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

    print('\nBuilding CoOp text prototypes...')
    print(f'Loading model: {args.backbone_type}...')
    print(f'Using labels in {args.labels_dir}')
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_backbone_and_tokenizer(args.backbone_type)
    model = model.to(device)
    model.eval()
    
    # Build text prototypes
    obj_category_dict = build_coop_prototypes(args, model, device)

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
    parser.add_argument('--save_dir', type=str, default='')
    #parser.add_argument('--annotations_file', type=str, default='/mnt/ddisk/boux/code/data/simd/train_coco_subset_N10.json')
    parser.add_argument('--backbone_type', type=str, default='')
    parser.add_argument('--ctx_path', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, default='')
    #parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(602, 602))
    #parser.add_argument('--window_size', type=int, default=224)
    #parser.add_argument('--scale_factor', type=int, default=1)
    #parser.add_argument('--num_b', type=int, default=10, help='Number of background samples to extract per image')
    #parser.add_argument('--k', type=int, default=200, help='Number of background prototypes (clusters for k-means)')
    #parser.add_argument('--store_bg_prototypes', action='store_true', default=False)
    parser.add_argument('--bg_prompts', type=str, default=None)
    args = parser.parse_args()

    main(args)
