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


NEW_CNAMES = {
    "airplane": "airplane",
    "airport": "airport",
    "background": "background",
    "baseballfield": "baseball field",
    "basketballcourt": "basketball court",
    "bridge": "bridge",
    "chimney": "chimney",
    "dam": "dam",
    "Expressway-Service-area": "expressway service area",
    "Expressway-toll-station": "expressway toll station",
    "golffield": "golf field",
    "groundtrackfield": "ground track field",
    "harbor": "harbor",
    "overpass": "overpass",
    "ship": "ship",
    "stadium": "stadium",
    "storagetank": "storage tank",
    "tenniscourt": "tennis court",
    "trainstation": "train station",
    "vehicle": "vehicle",
    "windmill": "windmill",
}


# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype
        
#         #print('TextEncoder dtype:', self.dtype)

#     def forward(self, prompts, tokenized_prompts):
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)

#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

#         return x
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

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
    
    # TODO: hardcoded to use remoteclip
    state_dict = torch.load('/home/gridsan/manderson/ovdsat/weights/RemoteCLIP-ViT-L-14.pt', map_location="cpu")
    #state_dict = torch.load('/home/gridsan/manderson/ovdsat/weights/RS5M_ViT-L-14.pt', map_location="cpu")
    model = clip.build_model(state_dict)
                             
    print('DEBUG load_clip_to_cpu model dtype:', model.dtype)
    #model.float()
    print('DEBUG load_clip_to_cpu model dtype:', model.dtype)
        
    return model


def build_coop_prototypes(args, model, device):
    '''
    Build zero-shot text prototypes by creating prompts and processing them with CLIP

    Args:
        args (argparse.Namespace): Input arguments
        tokenizer (TODO): Clip tokenizer
        model (torch.nn.Module): Backbone text encoder model
        device (str): Device to run the model on
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    # Read args.labels_dir
    with open(args.labels_dir, "r") as f:
        classes = [line.strip() for line in f]
    print(f'{len(classes)} class labels found')
    print(classes)
    
    n_ctx = 4 # TODO don't hardcode
    n_cls = len(classes)
    
    #TODO removed background for now
    #bg_index = classes.index('background')
    class_names = [c for c in classes if c != 'background']
    print('CLASS NAMES USED FOR MAPPING?')
    print(class_names)
    
    model, tokenizer = load_backbone_and_tokenizer(args.backbone_type)
    print('model device:', next(model.parameters()).device)
    
    print('DEBUG args.ctx_path')
    print(args.ctx_path)
    context = torch.load(args.ctx_path, map_location=torch.device('cpu') )
    
    prefix = context['state_dict']['token_prefix']
    ctx = context['state_dict']['ctx']
    suffix = context['state_dict']['token_suffix']
    
    name_lens = [len(tokenizer.encode(NEW_CNAMES[name])) for name in classes]
    print('name_lens')
    print(name_lens)

    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

    # For middle, unified context 
    half_n_ctx = n_ctx // 2
    prompts = []
    for i in range(n_cls):
        name_len = name_lens[i]
        prefix_i = prefix[i : i + 1, :, :]
        class_i = suffix[i : i + 1, :name_len, :]
        suffix_i = suffix[i : i + 1, name_len:, :]
        ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
        ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
        prompt = torch.cat(
            [
                prefix_i,     # (1, 1, dim)
                ctx_i_half1,  # (1, n_ctx//2, dim)
                class_i,      # (1, name_len, dim)
                ctx_i_half2,  # (1, n_ctx//2, dim)
                suffix_i,     # (1, *, dim)
            ],
            dim=1,
        )
        prompts.append(prompt)
    prompts = torch.cat(prompts, dim=0)
    
    print('DEBUG')
    save_name = f'prompts_{args.backbone_type}.pt'
    torch.save(prompts, os.path.join(args.save_dir, save_name))
    print(f'save prompts to {os.path.join(args.save_dir, save_name)}')
    
    print('prompts shape:', prompts.shape)
    #prompts = prompts.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    
    #prompt_prefix = " ".join(["X"] * n_ctx)
    clip_model = load_clip_to_cpu()
    prompt_prefix = "a satellite image of"
    prompts_for_token = [prompt_prefix + " " + NEW_CNAMES[name] + "." for name in classes]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_for_token])
    # DEBUG
    save_name = f'tokenized_prompts_{args.backbone_type}.pt'
    torch.save(tokenized_prompts, os.path.join(args.save_dir, save_name))
    
    with torch.no_grad():
        text_encoder = TextEncoder(clip_model).to(device)
        class_feats = text_encoder(prompts.to(device), tokenized_prompts.to(device))#.detach().numpy()
    
    #bg_feats = torch.from_numpy(text_feats[bg_index:bg_index+1])
    #class_feats = torch.from_numpy(np.delete(text_feats, bg_index, axis=0))
    
    print(f'Class feats shape: {class_feats.shape}')
    #print(f'Background feats shape: {bg_feats.shape}')

    class_dict = {
        'prototypes': class_feats.cpu(), #.cpu(),
        'label_names': class_names
    }
    
    # bg_dict = {
    #     'prototypes': bg_feats.cpu(),
    #     'label_names': ['bg_class_1']
    # }
    
    class_shape = class_dict['prototypes'].shape
    print(f'Shape of class prototypes: {class_shape}')
    #bg_shape = bg_dict['prototypes'].shape
    #print(f'Shape of background prototypes: {bg_shape}')
    
    return class_dict #, bg_dict


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
    
    # Create save directory if it does not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Build text prototypes
    obj_category_dict = build_coop_prototypes(args, model, device)
    #obj_category_dict, bg_category_dict = build_coop_prototypes(args, model, device)

    # Save background prototypes if specified
    # if args.store_bg_prototypes:
    #     save_name = f'bg_prototypes_{args.backbone_type}.pt'
    #     torch.save(bg_category_dict, os.path.join(args.save_dir, save_name))
    #     print(f'Saved background prototypes to {os.path.join(args.save_dir, save_name)}')

    # Save normal class prototypes
    save_name = f'prototypes_{args.backbone_type}.pt'
    torch.save(obj_category_dict, os.path.join(args.save_dir, save_name))
    print(f'Saved class prototypes to {os.path.join(args.save_dir, save_name)}\n')


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
    parser.add_argument('--store_bg_prototypes', action='store_true', default=False)
    #parser.add_argument('--bg_prompts', type=str, default=None)
    args = parser.parse_args()

    main(args)
