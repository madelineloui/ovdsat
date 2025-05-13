import os
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
#from CoOp.trainers.coop import CoOp, PromptLearner, TextEncoder, CustomCLIP
from CoOp.clip import clip
import torch.nn as nn

# TODO should enable cfg from CoOp for more seamless integration

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
        #print('TextEncoder dtype:', self.dtype)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def load_clip_to_cpu():
    
    backbone_name = "ViT-L/14" # TODO hardcoded
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    
    # TODO: hardcoded
    #state_dict = torch.load('/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-fmow-4.pt', map_location="cpu")
    state_dict = torch.load('/home/gridsan/manderson/ovdsat/weights/RemoteCLIP-ViT-L-14.pt', map_location="cpu")
    #state_dict = torch.load('/home/gridsan/manderson/ovdsat/weights/RS5M_ViT-L-14.pt', map_location="cpu")
    model = clip.build_model(state_dict)
                             
    #print('load_clip_to_cpu model dtype:', model.dtype)
    model.float()
        
    return model

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

    
# def build_coop_prototypes(args, model, device):
#     '''
#     Build zero-shot text prototypes by creating prompts and processing them with CLIP

#     Args:
#         args (argparse.Namespace): Input arguments
#         tokenizer (TODO): Clip tokenizer
#         model (torch.nn.Module): Backbone text encoder model
#         device (str): Device to run the model on
#     '''
    
#     # note this is only working on cpu currently
        
#     # Read args.labels_dir
#     with open(args.labels_dir, "r") as f:
#         classes = [line.strip() for line in f]
#     print(f'{len(classes)} class labels found')
#     print(classes)
    
#     model, tokenizer = load_backbone_and_tokenizer(args.backbone_type)
#     print('model device:', next(model.parameters()).device)
    
#     context = torch.load(args.ctx_path, map_location=torch.device('cpu') )
    
#     prefix = context['state_dict']['token_prefix']
#     ctx = context['state_dict']['ctx']
#     suffix = context['state_dict']['token_suffix']
    
#     name_lens = [len(tokenizer.encode(name)) for name in classes]
#     n_ctx = 4
#     n_cls = len(classes)

#     if ctx.dim() == 2:
#         ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

#     # For middle, unified context 
#     half_n_ctx = n_ctx // 2
#     prompts = []
#     for i in range(n_cls):
#         name_len = name_lens[i]
#         prefix_i = prefix[i : i + 1, :, :]
#         class_i = suffix[i : i + 1, :name_len, :]
#         suffix_i = suffix[i : i + 1, name_len:, :]
#         ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
#         ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
#         prompt = torch.cat(
#             [
#                 prefix_i,     # (1, 1, dim)
#                 ctx_i_half1,  # (1, n_ctx//2, dim)
#                 class_i,      # (1, name_len, dim)
#                 ctx_i_half2,  # (1, n_ctx//2, dim)
#                 suffix_i,     # (1, *, dim)
#             ],
#             dim=1,
#         )
#         prompts.append(prompt)
#     prompts = torch.cat(prompts, dim=0)
    
#     print('prompts shape:', prompts.shape)
#     #prompts = prompts.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    
#     prompt_prefix = " ".join(["X"] * n_ctx)
#     prompts_for_token = [prompt_prefix + " " + name + "." for name in classes]
#     tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_for_token])
    
#     clip_model = load_clip_to_cpu()
#     text_encoder = TextEncoder(clip_model)
#     text_feats = text_encoder(prompts, tokenized_prompts)

#     category_dict = {
#         'prototypes': text_feats.cpu(),
#         'label_names': classes
#     }
    
#     prototype_shape = category_dict['prototypes'].shape
#     print(f'Shape of text prototypes: {prototype_shape}')
    
#     return category_dict


def build_coop_prototypes(args, model, device):
    '''
    Build zero-shot text prototypes using CLIP.
    Returns foreground prototypes and a separate 'bg_class_1' if "background" is present.
    '''
    # Load class names
    with open(args.labels_dir, "r") as f:
        classes = [line.strip() for line in f]

    print(f'{len(classes)} class labels found: {classes}')

    has_background = "background" in classes
    bg_index = classes.index("background") if has_background else None

    # Build prompts
    prompt_prefix = "X X X X"
    prompt_texts = [f"{prompt_prefix} {name}." for name in classes]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompt_texts])

    # Load model + context
    model, tokenizer = load_backbone_and_tokenizer(args.backbone_type)
    context = torch.load(args.ctx_path, map_location="cpu")
    prefix = context['state_dict']['token_prefix']
    ctx = context['state_dict']['ctx']
    suffix = context['state_dict']['token_suffix']

    n_ctx = 4
    half_n_ctx = n_ctx // 2
    name_lens = [len(tokenizer.encode(name)) for name in classes]

    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0).expand(len(classes), -1, -1)

    prompts = []
    for i, name_len in enumerate(name_lens):
        prefix_i = prefix[i:i+1]
        class_i = suffix[i:i+1, :name_len]
        suffix_i = suffix[i:i+1, name_len:]
        ctx_half1 = ctx[i:i+1, :half_n_ctx]
        ctx_half2 = ctx[i:i+1, half_n_ctx:]
        prompt = torch.cat([prefix_i, ctx_half1, class_i, ctx_half2, suffix_i], dim=1)
        prompts.append(prompt)
    prompts = torch.cat(prompts, dim=0)

    # Encode prompts
    clip_model = load_clip_to_cpu()
    text_encoder = TextEncoder(clip_model)
    text_feats = text_encoder(prompts, tokenized_prompts).cpu()
    print('text_feats.shape before')
    print(text_feats.shape)

    # Split background if present
    if has_background:
        background_feat = text_feats[bg_index:bg_index+1]
        print('background_feat')
        print(background_feat.shape)
        background_dict = {
            'prototypes': background_feat,
            'label_names': ['bg_class_1']
        }
        # Remove background from main set
        del classes[bg_index]
        text_feats = torch.cat([text_feats[:bg_index], text_feats[bg_index+1:]], dim=0)
    else:
        background_dict = None
        
    print('text_feats.shape after')
    print(text_feats.shape)

    category_dict = {
        'prototypes': text_feats,
        'label_names': classes
    }

    return category_dict, background_dict


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
    obj_category_dict, bg_category_dict = build_coop_prototypes(args, model, device)

    # Create save directory if it does not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Build background prototypes if specified
    if bg_category_dict is not None:
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
