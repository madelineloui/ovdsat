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
from CoOp.clip import clip
#from CoOp.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import open_clip
import torch.nn as nn


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
    "SU-35":  "SU-35 aircraft",
    "C-130":  "C-130 aircraft",
    "C-17":   "C-17 aircraft",
    "C-5":    "C-5 aircraft",
    "F-16":   "F-16 aircraft",
    "TU-160": "TU-160 aircraft",
    "E-3":    "E-3 aircraft",
    "B-52":   "B-52 aircraft",
    "P-3C":   "P-3C aircraft",
    "B-1B":   "B-1B aircraft",
    "E-8":    "E-8 aircraft",
    "TU-22":  "TU-22 aircraft",
    "F-15":   "F-15 aircraft",
    "KC-135": "KC-135 aircraft",
    "F-22":   "F-22 aircraft",
    "FA-18":  "FA-18 aircraft",
    "TU-95":  "TU-95 aircraft",
    "KC-1":   "KC-1 aircraft",
    "SU-34":  "SU-34 aircraft",
    "SU-24":  "SU-24 aircraft",
    "car": "car",
    "truck": "truck",
    "airliner": "airliner",
    "stairtruck": "stair truck",
    "van": "van",
    "bus": "bus",
    "longvehicle": "long vehicle",
    "boat": "boat",
    "propeller": "propeller aircraft",
    "chartered": "chartered aircraft",
    "pushbacktruck": "pushback truck",
    "other": "others",
    "fighter": "fighter aircraft",
    "trainer": "trainer aircraft",
    "helicopter": "helicopter",
}
    
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

# def load_clip_to_cpu():
    
#     backbone_name = "ViT-L/14"
#     url = clip._MODELS[backbone_name]
#     model_path = clip._download(url)

#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None

#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")

#     model = clip.build_model(state_dict or model.state_dict())
    
#     # TODO: hardcoded to use remoteclip
#     state_dict = torch.load('/home/gridsan/manderson/ovdsat/weights/RemoteCLIP-ViT-L-14.pt', map_location="cpu")
#     model = clip.build_model(state_dict)
                                 
#     return model


def load_clip_to_cpu(backbone_name):
    
    print('-> using backbone:', backbone_name)
    
    # Start with this for all
    url = clip._MODELS["ViT-L/14"]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    
    if backbone_name == 'clip-14':
        print('LOADED CLIP-14!')
        
    if backbone_name == 'openclip-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        model.dtype = next(model.visual.parameters()).dtype
        print('LOADED OPENCLIP-14!')
    
    elif backbone_name == 'remoteclip-14':
        state_dict = torch.load('/home/gridsan/manderson/ovdsat/weights/RemoteCLIP-ViT-L-14.pt', map_location="cpu")
        model = clip.build_model(state_dict) 
        print('LOADED REMOTECLIP-14!')
        
    elif backbone_name == 'georsclip-14':
        state_dict = torch.load('/home/gridsan/manderson/ovdsat/weights/RS5M_ViT-L-14.pt', map_location="cpu")
        model = clip.build_model(state_dict) 
        print('LOADED GEORSCLIP-14!')
    
    elif backbone_name == 'openclip-14-remote-fmow':
        state_dict = torch.load('/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-remote-fmow.pt', map_location="cpu")
        model = clip.build_model(state_dict) 
        print('LOADED RemoteCLIP-14+FMOW!')
        
    elif backbone_name == 'openclip-14-geors-fmow':
        state_dict = torch.load('/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-geors-fmow.pt', map_location="cpu")
        model = clip.build_model(state_dict) 
        print('LOADED GEORSCLIP-14+FMOW!')
        
    return model


def build_coop_prototypes(args, device):
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
    
    n_ctx = args.NCTX
    ctp = args.CTP
    prompt_prefix = "a satellite image of"
    n_cls = len(classes)
    
    #class_names = [c for c in classes if c != 'background']
    class_names = [c for c in classes]
    # if args.store_bg_prototypes:
    #     class_names.append('background')
    #     n_cls += 1
    print('\nCLASS NAMES USED FOR MAPPING')
    print(class_names)
    print('\nMAPPED NAMES')
    print([NEW_CNAMES[name] if name in NEW_CNAMES else name for name in class_names])
    
    bg_index=None
    if 'background' in class_names:
        bg_index = class_names.index('background')
    
    _, tokenizer = load_backbone_and_tokenizer(args.backbone_type)
    
    #print('DEBUG args.ctx_path')
    #print(args.ctx_path)
    context = torch.load(args.ctx_path, map_location=torch.device('cpu') )
    
    prefix = context['state_dict']['token_prefix']
    ctx = context['state_dict']['ctx']
    suffix = context['state_dict']['token_suffix']
    
    #name_lens = [len(tokenizer.encode(NEW_CNAMES[name])) for name in class_names]
    # name_lens = [
    #     len(tokenizer.encode(NEW_CNAMES[name] if name in NEW_CNAMES else name))
    #     for name in class_names
    #     ]
    name_lens = [
        len(tokenizer.encode(NEW_CNAMES[name] if name in NEW_CNAMES else name))
        for name in class_names
        ]
    print('name_lens')
    print(len(name_lens))
    print(name_lens)

    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

    # # For middle, unified context 
    # half_n_ctx = n_ctx // 2
    # prompts = []
    # for i in range(n_cls):
    #     name_len = name_lens[i]
    #     prefix_i = prefix[i : i + 1, :, :]
    #     class_i = suffix[i : i + 1, :name_len, :]
    #     suffix_i = suffix[i : i + 1, name_len:, :]
    #     ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
    #     ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
    #     prompt = torch.cat(
    #         [
    #             prefix_i,     # (1, 1, dim)
    #             ctx_i_half1,  # (1, n_ctx//2, dim)
    #             class_i,      # (1, name_len, dim)
    #             ctx_i_half2,  # (1, n_ctx//2, dim)
    #             suffix_i,     # (1, *, dim)
    #         ],
    #         dim=1,
    #     )
    #     prompts.append(prompt)
    # prompts = torch.cat(prompts, dim=0)
    
    print(f'Using CTP={ctp}!')
    if ctp == "end":
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

    elif ctp == "middle":
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

    elif ctp == "front":
        prompts = []
        for i in range(n_cls):
            name_len = name_lens[i]
            prefix_i = prefix[i : i + 1, :, :]
            class_i = suffix[i : i + 1, :name_len, :]
            suffix_i = suffix[i : i + 1, name_len:, :]
            ctx_i = ctx[i : i + 1, :, :]
            prompt = torch.cat(
                [
                    prefix_i,  # (1, 1, dim)
                    class_i,   # (1, name_len, dim)
                    ctx_i,     # (1, n_ctx, dim)
                    suffix_i,  # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
    
    # For debugging
    # save_name = f'prompts_{args.backbone_type}.pt'
    # torch.save(prompts, os.path.join(args.save_dir, save_name))
    # print(f'save prompts to {os.path.join(args.save_dir, save_name)}')
    
    #print('prompts shape:', prompts.shape)
    #prompts = prompts.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    
    #prompt_prefix = " ".join(["X"] * n_ctx)
    clip_model = load_clip_to_cpu(args.backbone_type)
    prompt_prefix = "a satellite image of"
    #prompts_for_token = [prompt_prefix + " " + NEW_CNAMES[name] + "." for name in class_names]
    prompts_for_token = [
        prompt_prefix + " " + (NEW_CNAMES[name] if name in NEW_CNAMES else name) + "."
        for name in class_names
        ]
    
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_for_token])
    
    # For debugging
    # save_name = f'tokenized_prompts_{args.backbone_type}.pt'
    # torch.save(tokenized_prompts, os.path.join(args.save_dir, save_name))
    
    with torch.no_grad():
        text_encoder = TextEncoder(clip_model).to(device)
        text_feats = text_encoder(prompts.to(device), tokenized_prompts.to(device))#.detach().numpy()
        
    #print('DEBUG text_feats.shape', text_feats.shape)

    if bg_index:
        bg_feats = text_feats[bg_index:bg_index+1]
        bg_dict = {
            'prototypes': bg_feats.cpu(),
            'label_names': ['bg_class_1']
        }
        bg_shape = bg_dict['prototypes'].shape
        print(f'Shape of background prototypes: {bg_shape}')
    
        class_feats = torch.from_numpy(np.delete(np.array(text_feats.cpu()), bg_index, axis=0))
    else:
        class_feats = text_feats
    
    #print(f'Class feats shape: {class_feats.shape}')

    class_dict = {
        'prototypes': class_feats.cpu(),
        'label_names': class_names
    }
    
    class_shape = class_dict['prototypes'].shape
    print(f'Shape of class prototypes: {class_shape}')
    
    if bg_index:
        return class_dict, bg_dict
    else:
        return class_dict, None


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
    #model, tokenizer = load_backbone_and_tokenizer(args.backbone_type)
    #model = model.to(device)
    #model.eval()
    
    # Create save directory if it does not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Build text prototypes
    #obj_category_dict = build_coop_prototypes(args, model, device)
    obj_category_dict, bg_category_dict = build_coop_prototypes(args, device)

    # Save background prototypes if specified
    if bg_category_dict:
        save_name = f'bg_prototypes_{args.backbone_type}.pt'
        torch.save(bg_category_dict, os.path.join(args.save_dir, save_name))
        print(f'Saved background prototypes to {os.path.join(args.save_dir, save_name)}')

    # Save normal class prototypes
    save_name = f'prototypes_{args.backbone_type}.pt'
    torch.save(obj_category_dict, os.path.join(args.save_dir, save_name))
    print(f'Saved class prototypes to {os.path.join(args.save_dir, save_name)}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--backbone_type', type=str, default='')
    parser.add_argument('--ctx_path', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, default='') #make sure order is same as in CoOp!
    parser.add_argument('--store_bg_prototypes', action='store_true', default=False)
    parser.add_argument('--NCTX', type=int, default=4)
    parser.add_argument('--CTP', type=str, default='middle')
    parser.add_argument('--prompt_prefix', type=str, default='a satellite image of')
    args = parser.parse_args()

    main(args)
