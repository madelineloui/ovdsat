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
from CoOp.clip import clip

NEW_CNAMES = {
    "airplane": "airplane",
    "airport": "airport",
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
    "Large Civil Transport/Utility": "Large Civil Transport/Utility aircraft",
    "Medium Civil Transport/Utility": "Medium Civil Transport/Utility aircraft",
    "Military Fighter/Interceptor/Attack": "Military Fighter/Interceptor/Attack aircraft",
    "Military Trainer": "Military Trainer aircraft",
    "Military Transport/Utility/AWAC": "Military Transport/Utility/AWAC aircraft",
    "Small Civil Transport/Utility": "Small Civil Transport/Utility aircraft",
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
    "airplane": "flying vehicle with a central fuselage, wings, and tail",
    "airport": "large aviation facility with runways, taxiways, terminals, and aircraft parking areas",
    "baseballfield": "sports field with a diamond-shaped infield, bases, and surrounding outfield",
    "basketballcourt": "rectangular sports court with painted lines, center markings, and baskets at both ends",
    "bridge": "raised structure carrying a road or path across water, terrain, or another obstacle",
    "chimney": "tall narrow vertical structure used to release smoke or exhaust",
    "dam": "large barrier spanning a river or reservoir to control or store water",
    "Expressway-Service-area": "roadside complex along an expressway with parking, buildings, and vehicle access",
    "Expressway-toll-station": "roadway checkpoint with multiple lanes and booths for collecting tolls",
    "golffield": "large landscaped sports course with fairways, greens, sand traps, and surrounding vegetation",
    "groundtrackfield": "oval running track surrounding a central athletic field",
    "harbor": "sheltered waterfront area with docks, piers, boats, and port facilities",
    "overpass": "road structure elevated above another road or transportation route",
    "ship": "large watercraft with an elongated hull and visible deck structures",
    "stadium": "large sports venue with a central field and surrounding spectator seating",
    "storagetank": "large circular or cylindrical container used for storing liquids, gases, or bulk materials",
    "tenniscourt": "rectangular sports court divided by a central net with marked playing lines",
    "trainstation": "railway facility with tracks, passenger platforms, and station buildings",
    "vehicle": "road transport such as a car, truck, bus, or van",
    "windmill": "tall tower with large rotating blades that capture energy from wind",
    "airplane": "flying vehicle with a central fuselage, wings, and tail",
    "baseball diamond": "sports field with a diamond-shaped infield, four bases, and surrounding outfield",
    "basketball court": "rectangular sports court with painted lines, center markings, and baskets at both ends",
    "bridge": "raised structure carrying a road or path across water, terrain, or another obstacle",
    "ground track field": "oval running track surrounding a central athletic field",
    "harbor": "sheltered waterfront area with docks, piers, boats, and port facilities",
    "ship": "large watercraft with an elongated hull and visible deck structures",
    "storage tank": "large circular or cylindrical container used for storing liquids, gases, or bulk materials",
    "tennis court": "rectangular sports court divided by a central net with marked playing lines",
    "vehicle": "road transport such as a car, truck, bus, or van",
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
    "airplane": "an airplane, a flying vehicle with a central fuselage, wings, and tail",
    "airport": "an airport, a large aviation facility with runways, taxiways, terminals, and aircraft parking areas",
    "baseballfield": "a baseball field, a sports field with a diamond-shaped infield, bases, and surrounding outfield",
    "basketballcourt": "a basketball court, a rectangular sports court with painted lines, center markings, and baskets at both ends",
    "bridge": "a bridge, a raised structure carrying a road or path across water, terrain, or another obstacle",
    "chimney": "a chimney, a tall narrow vertical structure used to release smoke or exhaust",
    "dam": "a dam, a large barrier spanning a river or reservoir to control or store water",
    "Expressway-Service-area": "an expressway service area, a roadside complex with parking, buildings, and road access",
    "Expressway-toll-station": "an expressway toll station, a roadway checkpoint with multiple lanes and booths for collecting tolls",
    "golffield": "a golf field, a large landscaped sports course with fairways, greens, sand traps, and surrounding vegetation",
    "groundtrackfield": "a ground track field, an oval running track surrounding a central athletic field",
    "harbor": "a harbor, a sheltered waterfront area with docks, piers, boats, and port facilities",
    "overpass": "an overpass, a road structure elevated above another road or transportation route",
    "ship": "a ship, a large watercraft with an elongated hull and visible deck structures",
    "stadium": "a stadium, a large sports venue with a central field and surrounding spectator seating",
    "storagetank": "a storage tank, a large circular or cylindrical container used for storing liquids, gases, or bulk materials",
    "tenniscourt": "a tennis court, a rectangular sports court divided by a central net with marked playing lines",
    "trainstation": "a train station, a railway facility with tracks, passenger platforms, and station buildings",
    "vehicle": "a vehicle, road transport such as a car, truck, bus, or van",
    "windmill": "a windmill, a tall tower with large rotating blades that capture energy from wind",
    "airplane": "an airplane, a flying vehicle with a central fuselage, wings, and tail",
    "baseball diamond": "a baseball diamond, a sports field with a diamond-shaped infield, four bases, and surrounding outfield",
    "basketball court": "a basketball court, a rectangular sports court with painted lines, center markings, and baskets at both ends",
    "bridge": "a bridge, a raised structure carrying a road or path across water, terrain, or another obstacle",
    "ground track field": "a ground track field, an oval running track surrounding a central athletic field",
    "harbor": "a harbor, a sheltered waterfront area with docks, piers, boats, and port facilities",
    "ship": "a ship, a large watercraft with an elongated hull and visible deck structures",
    "storage tank": "a storage tank, a large circular or cylindrical container used for storing liquids, gases, or bulk materials",
    "tennis court": "a tennis court, a rectangular sports court divided by a central net with marked playing lines",
    "vehicle": "a vehicle, road transport such as a car, truck, bus, or van",
}

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
        #tokenized_bg_prompts = clip.tokenize(bg_prompts).to(device)
        bg_text_features = model.encode_text(tokenized_bg_prompts).to(device)
    elif 'customclip' in args.backbone_type:
        tokenized_bg_prompts = tokenizer(bg_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        bg_text_features = model.encode_text(tokenized_bg_prompts).to(device)
    else:
        tokenized_bg_prompts = tokenizer(bg_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        #tokenized_bg_prompts = clip.tokenize(bg_prompts).to(device)
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
    prompts = []
    if args.desc_type == 'label_only':
        for c in classes:
            if c in NEW_CNAMES:
                prompts.append(f'a satellite image of a {NEW_CNAMES[c]}')
            else:
                prompts.append(f'a satellite image of a {c}')
    elif args.desc_type == 'desc_only':
        for c in classes:
            if c in NEW_CNAMES_DESC:
                prompts.append(f'a satellite image of {NEW_CNAMES_DESC[c]}')
            else:
                print('ERROR!!!')
                print(c)
                prompts.append(f'a satellite image of a {c}')
    elif args.desc_type == 'label_and_desc':
        for c in classes:
            if c in NEW_CNAMES_LABEL_DESC:
                prompts.append(f'a satellite image of {NEW_CNAMES_LABEL_DESC[c]}')
            else:
                print('ERROR!!!')
                print(c)
                prompts.append(f'a satellite image of a {c}')
    print('PROMPTS:')
    print(prompts)
    
    # Tokenize and extract text features
    if any(b in args.backbone_type for b in ('openclip', 'remoteclip', 'georsclip')):
        tokenized_prompts = tokenizer(prompts).to(device)
        #tokenized_prompts = clip.tokenize(prompts).to(device)
        print('tokenized_prompts', tokenized_prompts.shape)
        text_features = model.encode_text(tokenized_prompts).to(device)
        print('text_features', text_features.shape)
    elif 'customclip' in args.backbone_type:
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = model.encode_text(tokenized_prompts).to(device)
    else:
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        #tokenized_prompts = clip.tokenize(prompts).to(device)
        text_features = model.get_text_features(**tokenized_prompts).to(device)
        # Pass through projection layer to match image embedding dimension
        # text_features = model.text_projection(text_features) # TODO I think model.get_text_features takes care of this?
    
    # Normalize
    norm_text_features = F.normalize(text_features, p=2, dim=-1)
    print('norm_text_features', norm_text_features.shape)

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

    # print('EMBEDDING')
    # print(obj_category_dict['prototypes'][0][:100])

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
    parser.add_argument('--save_dir', type=str, default='run/text_prototypes/boxes/dior')
    parser.add_argument('--backbone_type', type=str, default='clip-14')
    parser.add_argument('--labels_dir', type=str, default='/home/gridsan/manderson/ovdsat/data/text/dior_labels.txt')
    parser.add_argument('--bg_prompts', type=str, default=None)
    parser.add_argument('--desc_type', type=str, default="label_only") # options: label_only, desc_only, label_and_desc
    args = parser.parse_args()

    main(args)
