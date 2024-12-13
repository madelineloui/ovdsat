'''
All the logic and functions related extracting robust features with pre-trained backbones is found here. 
This way, training, eval and the model itself can all use the same code.
'''

import torch
import torch.nn.functional as F
from transformers import CLIPModel
from huggingface_hub import hf_hub_download
import open_clip

# Paths to the pre-trained models
PATH_CKPT_GEORSCLIP_32 = 'weights/RS5M_ViT-B-32.pt'
PATH_CKPT_GEORSCLIP_14 = 'weights/RS5M_ViT-H-14.pt'
PATH_CKPT_REMOTECLIP_32 = 'weights/RemoteCLIP-ViT-B-32.pt'
PATH_CKPT_REMOTECLIP_14 = 'weights/RemoteCLIP-ViT-L-14.pt'
PATH_CKPT_CLIP14_CAP0 = '/home/gridsan/manderson/train-CLIP/run/cap0/model-epoch=33-val_loss=0.99_clip.pth'
PATH_CKPT_CLIP14_CAP1 = '/home/gridsan/manderson/train-CLIP/run/cap1/model-epoch=45-val_loss=0.79_clip.pth' 
PATH_CKPT_CLIP14_CAP2 = '/home/gridsan/manderson/train-CLIP/run/cap2/model-epoch=36-val_loss=0.69_clip.pth' 
PATH_CKPT_CLIP14_GPT0_512_EPOCH23 = '/home/gridsan/manderson/train-CLIP/run/gpt_n0/test/model-epoch23_clip.pth'
PATH_CKPT_CLIP14_GPT0_512_EPOCH50 = '/home/gridsan/manderson/train-CLIP/run/gpt_n0/last_clip.pth'
PATH_CKPT_CLIP14_GPTe_512_EPOCH26 = '/home/gridsan/manderson/train-CLIP/run/gpt_e/test/model-epoch-26_clip.pth'
PATH_CKPT_CLIP14_GPTe_512_EPOCH50 = '/home/gridsan/manderson/train-CLIP/run/gpt_e/last_clip.pth'
PATH_CKPT_CLIP14_GPT0_1024_EPOCH26 = '/home/gridsan/manderson/train-CLIP/run/gpt_n0-1024/test/model-epoch26_clip.pth'
PATH_CKPT_CLIP14_GPT0_1024_EPOCH50 = '/home/gridsan/manderson/train-CLIP/run/gpt_n0-1024/last_clip.pth'
PATH_CKPT_CLIP14_GPTe_1024_EPOCH24 = '/home/gridsan/manderson/train-CLIP/run/gpt_e-1024/model-epoch24.pth'
PATH_CKPT_CLIP14_GPTe_1024_EPOCH50 = '/home/gridsan/manderson/train-CLIP/run/gpt_e-1024/last_clip.pth'

def load_backbone(backbone_type):
    '''
    Load a pre-trained backbone model.

    Args:
        backbone_type (str): Backbone type
    '''
    if backbone_type == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    elif backbone_type == 'dinov2-reg':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', force_reload=True)
    elif backbone_type == 'clip-32':
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-base-patch32").vision_model
    elif backbone_type == 'clip-14':
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14").vision_model
    elif backbone_type == 'openclip-32':
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model = model.visual
        model.output_tokens = True
    elif backbone_type == 'openclip-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        model = model.visual
        model.output_tokens = True
    elif backbone_type == 'georsclip-32':
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32')
        ckpt = torch.load(PATH_CKPT_GEORSCLIP_32, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.visual
        model.output_tokens = True
    elif backbone_type == 'georsclip-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-H-14')
        ckpt = torch.load(PATH_CKPT_GEORSCLIP_14, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.visual
        model.output_tokens = True
    elif backbone_type == 'remoteclip-32':
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32')
        ckpt = torch.load(PATH_CKPT_REMOTECLIP_32, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.visual
        model.output_tokens = True
    elif backbone_type == 'remoteclip-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14')
        ckpt = torch.load(PATH_CKPT_REMOTECLIP_14, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.visual
        model.output_tokens = True
    elif backbone_type == 'clip-14-cap0':
        # backbone is a new pretrained model      
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_CAP0, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_CAP0}')
    elif backbone_type == 'clip-14-cap1':
        # backbone is a new pretrained model      
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_CAP1, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_CAP1}')
    elif backbone_type == 'clip-14-cap2':
        # backbone is a new pretrained model      
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_CAP2, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_CAP2}')
    elif backbone_type == 'clip-14-gpt0-512-epoch23':
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_GPT0_512_EPOCH23, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_GPT0_512_EPOCH23}')
    elif backbone_type == 'clip-14-gpt0-512-epoch50':
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_GPT0_512_EPOCH50, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_GPT0_512_EPOCH50}')
    elif backbone_type == 'clip-14-gpte-512-epoch26':
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_GPTe_512_EPOCH26, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_GPTe_512_EPOCH26}')
    elif backbone_type == 'clip-14-gpte-512-epoch50':
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_GPTe_512_EPOCH50, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_GPTe_512_EPOCH50}')
    elif backbone_type == 'clip-14-gpt0-1024-epoch26':
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_GPT0_1024_EPOCH26, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_GPT0_1024_EPOCH26}')
    elif backbone_type == 'clip-14-gpt0-1024-epoch50':
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_GPT0_1024_EPOCH50, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_GPT0_1024_EPOCH50}')
    elif backbone_type == 'clip-14-gpte-1024-epoch24':
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_GPTe_1024_EPOCH24, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_GPTe_1024_EPOCH24}')
    elif backbone_type == 'clip-14-gpte-1024-epoch50':
        model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
        ckpt = torch.load(PATH_CKPT_CLIP14_GPTe_1024_EPOCH50, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.vision_model
        model.output_tokens = True
        print(f'Using checkpoint {PATH_CKPT_CLIP14_GPTe_1024_EPOCH50}')


    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    return model

def prepare_image_for_backbone(input_tensor, backbone_type):
    '''
    Preprocess an image for the backbone model given an input tensor and the backbone type.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (B, C, H, W)
        backbone_type (str): Backbone type
    '''
    
    if input_tensor.shape[1] == 4:
        input_tensor = input_tensor[:, :3, :, :]  # Discard the alpha channel (4th channel)

    # Define mean and std for normalization depending on the backbone type
    mean = torch.tensor([0.485, 0.456, 0.406]).to(input_tensor.device) if 'dinov2' in backbone_type else torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(input_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(input_tensor.device) if 'dinov2' in backbone_type else torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(input_tensor.device)
    
    # Scale the values to range from 0 to 1
    input_tensor /= 255.0
    
    # Normalize the tensor
    normalized_tensor = (input_tensor - mean[:, None, None]) / std[:, None, None]

    return normalized_tensor

def get_backbone_params(backbone_type):
    '''
    Get the parameters patch size and embedding dimensionality of the backbone model given the backbone type.

    Args:
        backbone_type (str): Backbone type
    '''

    if backbone_type == 'georsclip-14':
        patch_size = 14
        D = 1280
    elif '14' in backbone_type or 'dinov2' in backbone_type:
        patch_size = 14
        D = 1024
    else:
        patch_size = 32
        D = 768
    return patch_size, D


def extract_clip_features(images, model, backbone_type, tile_size=224):
    '''
    Extract features from a CLIP pre-trained backbone using a sliding window approach to handle images of variable sizes.

    Args:
        images (torch.Tensor): Input tensor with shape (B, C, H, W)
        model (torch.nn.Module): CLIP model
        backbone_type (str): Backbone type
        tile_size (int): Size of the tiles to process the image. Set to 224 as CLIP pre-trained models use 224x224 images.
    '''
    # Extract size and number of tiles
    B, _, image_size, _ = images.shape
    
    patch_size, D = get_backbone_params(backbone_type)

    num_tiles = (image_size // tile_size)**2 if image_size % tile_size == 0 else (image_size // tile_size + 1)**2
    num_tiles_side = int(num_tiles**0.5)

    # Create full image features tensor and a counter for aggregation
    output_features = torch.zeros((B, image_size // patch_size, image_size // patch_size, D)).to(images.device)
    count_tensor = torch.zeros((B, image_size // patch_size, image_size // patch_size,)).to(images.device)

    # Process tiles through CLIP
    with torch.no_grad():
        for i in range(num_tiles_side):
            for j in range(num_tiles_side):

                # Update tile coords
                start_i = i * tile_size
                start_j = j * tile_size
                end_i = min(start_i + tile_size, image_size)
                end_j = min(start_j + tile_size, image_size)

                # If tile exceeds, make new tile containing more image content
                if end_i - start_i < tile_size:
                    start_i = end_i - tile_size
                if end_j - start_j < tile_size:
                    start_j = end_j - tile_size
                
                # Extract the tile from the original image
                tile = images[:, :, start_i:end_i, start_j:end_j]
    
                # Extract CLIP's features before token pooling
                #if backbone_type == 'clip-32' or backbone_type == 'clip-14':
                if 'georsclip' in backbone_type or 'remoteclip' in backbone_type or 'openclip' in backbone_type:
                    image_features = model(tile)[-1]
                elif 'clip-32' in backbone_type or 'clip-14' in backbone_type:
                    image_features = model(tile).last_hidden_state[:, 1:]
                else:
                    image_features = model(tile)[-1]

                _, K, D = image_features.shape
                p_w = p_h = int(K**0.5)
                image_features = image_features.reshape(B, p_h, p_w, -1)  # Reshape to 2D

                # Add features to their location in the original image and track counts per location
                output_features[:, start_i // patch_size:end_i // patch_size, start_j // patch_size:end_j // patch_size] += image_features
                count_tensor[:, start_i // patch_size:end_i // patch_size, start_j // patch_size:end_j // patch_size] += 1
    
    # Average the overlapping patches
    output_features /= count_tensor.unsqueeze(-1)
    
    return output_features, count_tensor

def extract_backbone_features(images, model, backbone_type, scale_factor=1):
    '''
    Extract features from a pre-trained backbone for any of the supported backbones.

    Args:
        images (torch.Tensor): Input tensor with shape (B, C, H, W)
        model (torch.nn.Module): Backbone model
        backbone_type (str): Backbone type
        scale_factor (int): Scale factor for the input images. Set to 1 for no scaling.
    '''
    images = F.interpolate(images, scale_factor=scale_factor, mode='bicubic')

    if 'dinov2' in backbone_type:
        with torch.no_grad():
            feats = model.forward_features(images)['x_prenorm'][:, 1:]
    elif 'resnet' in backbone_type:
        with torch.no_grad():
            feats = model(images)
            print(len(feats))
            for fe in feats:
                print(fe.shape)
    elif 'clip' in backbone_type:
        feats, _ = extract_clip_features(images, model, backbone_type)
        feats = feats.view(feats.shape[0], -1, feats.shape[-1])
    else:
        raise NotImplementedError('Backbone {} not implemented'.format(backbone_type))

    return feats