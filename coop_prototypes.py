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
from utils_dir.backbones_utils import load_clip_to_cpu, load_backbone_and_tokenizer, extract_backbone_features, get_backbone_params
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
    "Large Civil Transport/Utility": "Large Civil Transport or Utility aircraft",
    "Medium Civil Transport/Utility": "Medium Civil Transport or Utility aircraft",
    "Military Fighter/Interceptor/Attack": "Military Fighter or Interceptor or Attack aircraft",
    "Military Trainer": "Military Trainer aircraft",
    "Military Transport/Utility/AWAC": "Military Transport or Utility or AWAC aircraft",
    "Small Civil Transport/Utility": "Small Civil Transport or Utility aircraft",
    "car": "a small road vehicle with four wheels and a compact body",
    "truck": "a large road vehicle with a cargo bed or box-shaped rear section",
    "airliner": "a large passenger aircraft with swept wings, wing-mounted jet engines, a long fuselage, and a swept tail",
    "stairtruck": "an airport truck with a raised passenger staircase",
    "van": "a medium-sized road vehicle with a tall enclosed rectangular body",
    "bus": "a long passenger road vehicle with a large enclosed body and many windows",
    "longvehicle": "a very long road vehicle with an extended body or trailer",
    "boat": "a small watercraft with a long narrow hull",
    "propeller": "an aircraft with straight or swept wings, propeller engines, a medium-length fuselage, and a conventional tail",
    "chartered": "a small passenger aircraft with straight or swept wings, small jet or propeller engines, a short fuselage, and a conventional tail",
    "pushbacktruck": "a low airport vehicle with a wide rectangular body used to move aircraft",
    "other": "another miscellaneous vehicle with varied shape and size",
    "fighter": "a compact military aircraft with swept wings, rear-mounted jet engines, a narrow fuselage with pointed nose, and one or two vertical tails", 
    "trainer": "a small training aircraft with straight wings, small jet or propeller engines, a short fuselage with prominent cockpit, and a conventional tail",
    "helicopter": "an aircraft with large overhead rotor blades, engines above the fuselage, a compact body, and a long narrow tail boom",
}

NEW_CNAMES_DESC = {
    "SU-35":  "an aircraft with broad swept wings, two rear-mounted jet engines, a long pointed fuselage, and two outward-angled vertical tails.",
    "C-130":  "an aircraft with straight high-mounted wings, four wing-mounted propeller engines, a thick boxy cargo fuselage, and a broad vertical tail.",
    "C-17":   "an aircraft with broad high-mounted swept wings, four wing-mounted jet engines, a wide rounded cargo fuselage, and a large T-shaped tail.",
    "C-5":    "an aircraft with very large high-mounted swept wings, four wing-mounted jet engines, an extremely large elongated cargo fuselage, and a large T-shaped tail.",
    "F-16":   "an aircraft with compact swept wings, one rear-mounted jet engine, a narrow pointed fuselage, and one vertical tail.",
    "TU-160": "an aircraft with large variable-sweep wings, four rear-mounted jet engines, a long narrow streamlined fuselage, and one swept vertical tail.",
    "E-3":    "an aircraft with swept wings, four wing-mounted jet engines, a long narrow fuselage with a large circular radar disk above it, and a swept tail.",
    "B-52":   "an aircraft with very long narrow swept wings, eight wing-mounted jet engines, a long slender fuselage, and one swept vertical tail.",
    "P-3C":   "an aircraft with straight low-mounted wings, four wing-mounted propeller engines, a long narrow fuselage, and a tall tail with a pointed rear extension.",
    "B-1B":   "an aircraft with variable-sweep wings, four rear-mounted jet engines, a long narrow pointed fuselage, and one swept vertical tail.",
    "E-8":    "an aircraft with swept wings, four wing-mounted jet engines, a long narrow airliner-shaped fuselage, and a conventional swept tail.",
    "TU-22":  "an aircraft with variable-sweep wings, two rear-mounted jet engines, a long narrow pointed fuselage, and one tall swept vertical tail.",
    "F-15":   "an aircraft with broad swept wings, two rear-mounted jet engines, a wide angular fuselage, and two tall vertical tails.",
    "KC-135": "an aircraft with long swept wings, four wing-mounted jet engines, a long narrow tanker fuselage, and one swept vertical tail.",
    "F-22":   "an aircraft with broad diamond-shaped wings, two rear-mounted jet engines, a wide angular stealth fuselage, and two outward-angled vertical tails.",
    "FA-18":  "an aircraft with swept trapezoidal wings, two rear-mounted jet engines, a broad pointed fuselage, and two outward-angled vertical tails.",
    "TU-95":  "an aircraft with long swept wings, four wing-mounted propeller engines, a long slender fuselage, and one swept vertical tail.",
    "KC-1":   "an aircraft with swept wings, three rear and wing-mounted jet engines, a wide airliner-shaped fuselage, and one swept vertical tail.",
    "SU-34":  "an aircraft with broad swept wings, two rear-mounted jet engines, a wide flattened nose and long fuselage, and two vertical tails.",
    "SU-24":  "an aircraft with variable-sweep wings, two rear-mounted jet engines, a long narrow angular fuselage, and one swept vertical tail.",
    "car": "a small road vehicle with four wheels and a compact body",
    "truck": "a large road vehicle with a cargo bed or box-shaped rear section",
    "airliner": "a large passenger aircraft with swept wings, wing-mounted jet engines, a long fuselage, and a swept tail",
    "stairtruck": "an airport truck with a raised passenger staircase",
    "van": "a medium-sized road vehicle with a tall enclosed rectangular body",
    "bus": "a long passenger road vehicle with a large enclosed body and many windows",
    "longvehicle": "a very long road vehicle with an extended body or trailer",
    "boat": "a small watercraft with a long narrow hull",
    "propeller": "an aircraft with straight or swept wings, propeller engines, a medium-length fuselage, and a conventional tail",
    "chartered": "a small passenger aircraft with straight or swept wings, small jet or propeller engines, a short fuselage, and a conventional tail",
    "pushbacktruck": "a low airport vehicle with a wide rectangular body used to move aircraft",
    "other": "another miscellaneous vehicle with varied shape and size",
    "fighter": "a compact military aircraft with swept wings, rear-mounted jet engines, a narrow fuselage with pointed nose, and one or two vertical tails", 
    "trainer": "a small training aircraft with straight wings, small jet or propeller engines, a short fuselage with prominent cockpit, and a conventional tail",
    "helicopter": "an aircraft with large overhead rotor blades, engines above the fuselage, a compact body, and a long narrow tail boom",
}

NEW_CNAMES_LABEL_DESC = {
    "SU-35":  "a SU-35 aircraft with broad swept wings, two rear-mounted jet engines, a long pointed fuselage, and two outward-angled vertical tails.",
    "C-130":  "a C-130 aircraft with straight high-mounted wings, four wing-mounted propeller engines, a thick boxy cargo fuselage, and a broad vertical tail.",
    "C-17":   "a C-17 aircraft with broad high-mounted swept wings, four wing-mounted jet engines, a wide rounded cargo fuselage, and a large T-shaped tail.",
    "C-5":    "a C-5 aircraft with very large high-mounted swept wings, four wing-mounted jet engines, an extremely large elongated cargo fuselage, and a large T-shaped tail.",
    "F-16":   "a F-16 aircraft with compact swept wings, one rear-mounted jet engine, a narrow pointed fuselage, and one vertical tail.",
    "TU-160": "a TU-160 aircraft with large variable-sweep wings, four rear-mounted jet engines, a long narrow streamlined fuselage, and one swept vertical tail.",
    "E-3":    "a E-3 aircraft with swept wings, four wing-mounted jet engines, a long narrow fuselage with a large circular radar disk above it, and a swept tail.",
    "B-52":   "a B-52 aircraft with very long narrow swept wings, eight wing-mounted jet engines, a long slender fuselage, and one swept vertical tail.",
    "P-3C":   "a P-3C aircraft with straight low-mounted wings, four wing-mounted propeller engines, a long narrow fuselage, and a tall tail with a pointed rear extension.",
    "B-1B":   "a B-1B aircraft with variable-sweep wings, four rear-mounted jet engines, a long narrow pointed fuselage, and one swept vertical tail.",
    "E-8":    "a E-8 aircraft with swept wings, four wing-mounted jet engines, a long narrow airliner-shaped fuselage, and a conventional swept tail.",
    "TU-22":  "a TU-22 aircraft with variable-sweep wings, two rear-mounted jet engines, a long narrow pointed fuselage, and one tall swept vertical tail.",
    "F-15":   "a F-15 aircraft with broad swept wings, two rear-mounted jet engines, a wide angular fuselage, and two tall vertical tails.",
    "KC-135": "a KC-135 aircraft with long swept wings, four wing-mounted jet engines, a long narrow tanker fuselage, and one swept vertical tail.",
    "F-22":   "a F-22 aircraft with broad diamond-shaped wings, two rear-mounted jet engines, a wide angular stealth fuselage, and two outward-angled vertical tails.",
    "FA-18":  "a FA-18 aircraft with swept trapezoidal wings, two rear-mounted jet engines, a broad pointed fuselage, and two outward-angled vertical tails.",
    "TU-95":  "a TU-95 aircraft with long swept wings, four wing-mounted propeller engines, a long slender fuselage, and one swept vertical tail.",
    "KC-1":   "a KC-1 aircraft with swept wings, three rear and wing-mounted jet engines, a wide airliner-shaped fuselage, and one swept vertical tail.",
    "SU-34":  "a SU-34 aircraft with broad swept wings, two rear-mounted jet engines, a wide flattened nose and long fuselage, and two vertical tails.",
    "SU-24":  "a SU-24 aircraft with variable-sweep wings, two rear-mounted jet engines, a long narrow angular fuselage, and one swept vertical tail.",
    "car": "a car, a small road vehicle with four wheels and a compact body",
    "truck": "a truck, a large road vehicle with a cargo bed or box-shaped rear section",
    "airliner": "an airliner, a large passenger aircraft with swept wings, wing-mounted jet engines, a long fuselage, and a swept tail",
    "stairtruck": "a stair truck, an airport truck with a raised passenger staircase",
    "van": "a van, a medium-sized road vehicle with a tall enclosed rectangular body",
    "bus": "a bus, a long passenger road vehicle with a large enclosed body and many windows",
    "longvehicle": "a long vehicle, a very long road vehicle with an extended body or trailer",
    "boat": "a boat, a small watercraft with a long narrow hull",
    "propeller": "a propeller aircraft, an aircraft with straight or swept wings, propeller engines, a medium-length fuselage, and a conventional tail",
    "chartered": "a chartered aircraft, a small passenger aircraft with straight or swept wings, small jet or propeller engines, a short fuselage, and a conventional tail",
    "pushbacktruck": "a pushback truck, a low airport vehicle with a wide rectangular body used to move aircraft",
    "other": "another vehicle, a miscellaneous vehicle with varied shape and size",
    "fighter": "a fighter aircraft, a compact military aircraft with swept wings, rear-mounted jet engines, a narrow fuselage with pointed nose, and one or two vertical tails",
    "trainer": "a trainer aircraft, a small training aircraft with straight wings, small jet or propeller engines, a short fuselage with prominent cockpit, and a conventional tail",
    "helicopter": "a helicopter, an aircraft with large overhead rotor blades, engines above the fuselage, a compact body, and a long narrow tail boom",
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
    if args.desc_type == 'label_only':
        print([NEW_CNAMES[name] if name in NEW_CNAMES else name for name in class_names])
    elif args.desc_type == 'desc_only':
        print([NEW_CNAMES_DESC[name] if name in NEW_CNAMES_DESC else name for name in class_names])
    elif args.desc_type == 'label_and_desc':
        print([NEW_CNAMES_LABEL_DESC[name] if name in NEW_CNAMES_LABEL_DESC else name for name in class_names])
    
    bg_index=None
    if 'background' in class_names:
        bg_index = class_names.index('background')
    
    _, tokenizer = load_backbone_and_tokenizer(args.backbone_type)

    print(f'Using context from {args.ctx_path}')
    context = torch.load(args.ctx_path, map_location=torch.device('cpu') )
    
    prefix = context['state_dict']['token_prefix']
    ctx = context['state_dict']['ctx']
    suffix = context['state_dict']['token_suffix']

    if args.desc_type == 'label_only':
        name_lens = [
            len(tokenizer.encode(NEW_CNAMES[name] if name in NEW_CNAMES else name))
            for name in class_names
            ]
    elif args.desc_type == 'desc_only':
        name_lens = [
            len(tokenizer.encode(NEW_CNAMES_DESC[name] if name in NEW_CNAMES_DESC else name))
            for name in class_names
            ]
    elif args.desc_type == 'label_and_desc':
        name_lens = [
            len(tokenizer.encode(NEW_CNAMES_LABEL_DESC[name] if name in NEW_CNAMES_LABEL_DESC else name))
            for name in class_names
            ]
    print('name_lens')
    print(len(name_lens))
    print(name_lens)

    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)
    
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

    if args.desc_type == 'label_only':
        prompts_for_token = [
            prompt_prefix + " " + (NEW_CNAMES[name] if name in NEW_CNAMES else name) + "."
            for name in class_names
            ]
    elif args.desc_type == 'desc_only':
        prompts_for_token = [
            prompt_prefix + " " + (NEW_CNAMES_DESC[name] if name in NEW_CNAMES_DESC else name) + "."
            for name in class_names
            ]
    elif args.desc_type == 'label_and_desc':
        prompts_for_token = [
            prompt_prefix + " " + (NEW_CNAMES_LABEL_DESC[name] if name in NEW_CNAMES_LABEL_DESC else name) + "."
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
    parser.add_argument('--desc_type', type=str, default="label_only") # options: label_only, desc_only, label_and_desc
    args = parser.parse_args()

    main(args)
