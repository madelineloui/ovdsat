import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import open_clip
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import json
import pandas as pd
from PIL import Image

PATH_CKPT_CLIP14 = 'weights/clip-vit-large-patch14'
PATH_CKPT_GEORSCLIP_14 = 'weights/RS5M_ViT-L-14.pt'
PATH_CKPT_REMOTECLIP_14 = 'weights/RemoteCLIP-ViT-L-14.pt'
PATH_CKPT_CLIP14_FMOW = '/home/gridsan/manderson/train-CLIP/run/fmow/fmow-test-4.pth'
PATH_CKPT_OPENCLIP14_FMOW = '/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-fmow-4.pt'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, choices=['RN50', 'ViT-B-32', 'ViT-L-14'], help="Backbone name")
    parser.add_argument("--dataset", default='eurosat', type=str, help="Name of dataset")
    parser.add_argument("--dataset-root", type=str, help="Path to dataset root")
    parser.add_argument("--clip-path", default=None, type=str, help="Path to CLIP weight")
    parser.add_argument("--backbone_name", default='openclip-fmow', type=str)
    parser.add_argument("--backbone", default=None, type=str, help="Either openclip or clip")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--num-shots", type=int, nargs='+', default=[5, 10, 30], help="Number of samples per class for linear probe")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")  # New argument
    parser.add_argument("--seed", type=int, default=1, help="Seed for splits")
    parser.add_argument("--mode", type=str, help="Either zero shot or linear probe")
    args = parser.parse_args()
    return args


def save_results_to_file(args, results):
    """Saves experiment parameters and results to a text file."""
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.backbone_name}_{args.dataset}_results-{args.mode}-{args.seed}.txt"
    results_file = os.path.join(args.output_dir, output_file)

    with open(results_file, "w") as f:
        f.write("=== Experiment Parameters ===\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        if args.mode == 'zero_shot':
            f.write("\n=== Zero-Shot Classification Results ===\n")
            f.write(f"Zero-Shot Accuracy: {results * 100:.2f}%\n")
        
        elif args.mode == 'linear_probe':
            f.write("\n=== Linear Probe Results ===\n")
            for k, v in results.items():
                f.write(f"{k}: {v * 100:.2f}%\n")

    print(f"Results saved to: {results_file}")
    
    
def get_model(args):
    if args.backbone == 'openclip':
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device=args.device)
        tokenize = open_clip.tokenize
        if args.clip_path is not None:
            checkpoint = torch.load(args.clip_path, map_location="cuda")
            msg = CLIP_model.load_state_dict(checkpoint)
            print(msg)
    elif args.backbone == 'clip':
        CLIP_model = CLIPModel.from_pretrained(PATH_CKPT_CLIP14).to(args.device)
        processor = CLIPProcessor.from_pretrained(PATH_CKPT_CLIP14)

        def preprocess_train(images):
            proc = processor(images=images, return_tensors="pt")["pixel_values"].squeeze(0)
            return proc

        def preprocess_val(images):
            proc = processor(images=images, return_tensors="pt")["pixel_values"].squeeze(0)
            return proc

        def tokenize(texts):
            return processor(text=texts, return_tensors="pt", padding=True, truncation=True)["input_ids"]

        if args.clip_path:
            checkpoint = torch.load(args.clip_path, map_location=args.device)
            msg = CLIP_model.load_state_dict(checkpoint, strict=False)
            print(msg)
            
    for name, parameter in CLIP_model.named_parameters():
        parameter.requires_grad = False

    return CLIP_model, preprocess_train, preprocess_val, tokenize


# # Custom Dataset Class
# class CsvDataset(Dataset):
#     def __init__(self, csv_file, transform=None):
#         self.data = pd.read_csv(csv_file)  # Load CSV
#         self.transform = transform  # Apply transformations

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # Get image path, label, and classname
#         img_path = self.data.iloc[idx]["Filename"]
#         label = int(self.data.iloc[idx]["Label"])  # Ensure it's an integer
        
#         # Load image
#         image = Image.open(img_path).convert("RGB")
        
#         # Apply transforms (if any)
#         if self.transform:
#             image = self.transform(image)

#         return image, label


class CsvDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # Load CSV file
        self.transform = transform  # Apply image transformations
        self.root_dir = root_dir

        # Create a class-to-index mapping (like ImageFolder)
        self.classes = sorted(self.data["ClassName"].unique())  # Unique class names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Map class names to numeric labels
        self.data["Label"] = self.data["ClassName"].map(self.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]["Filename"])
        label = int(self.data.iloc[idx]["Label"])  # Convert label to integer

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label
    
    
def get_csv_dataloader(dataset_root, csv_file, preprocess, batch_size, num_workers):
    # Load dataset
    dataset = CsvDataset(dataset_root, csv_file, transform=preprocess)

    print("Detected classes:", dataset.classes)  # Print detected class names

    if len(dataset) == 0:
        raise FileNotFoundError("No valid image files found in the CSV file.")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return dataset, dataloader
    

def get_dataloader(dataset_root, preprocess, batch_size, num_workers):
    def is_valid_file(path):
        return not any(part.startswith('.') for part in path.split(os.sep))

    # Custom function to get valid class directories
    def find_classes(directory):
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and not d.startswith('.')]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    # Custom ImageFolder class that ignores hidden directories
    class CustomImageFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            return find_classes(directory)
            
    # Load dataset with custom filtering
    dataset = CustomImageFolder(dataset_root, transform=preprocess, is_valid_file=is_valid_file)

    print("Detected classes:", dataset.classes)  # Verify hidden folders are ignored

    if len(dataset.samples) == 0:
        raise FileNotFoundError("No valid image files found in the dataset root.")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return dataset, dataloader


def zero_shot_classification(model, dataloader, class_names, tokenize, args):
    model.eval()
    
    # Convert class names into text prompts and encode them
    text_inputs = tokenize([f"A satellite image of {c}" for c in class_names]).to(args.device)
    
    with torch.no_grad():
        if args.backbone == 'openclip':
            text_features = model.encode_text(text_inputs)
        elif args.backbone == 'clip':
            text_features = model.get_text_features(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize text features

    correct = 0
    total = 0

    with torch.no_grad():
        i = 0
        for images, labels in tqdm(dataloader, desc="Zero-Shot Classification"):
            # if i > 5:
            #     break
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Compute image features
            if args.backbone == 'openclip':
                image_features = model.encode_image(images)
            elif args.backbone == 'clip':
                image_features = model.get_image_features(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize image features

            # Compute similarity scores (cosine similarity)
            similarity = image_features @ text_features.T  # Compute similarity scores
            predictions = similarity.argmax(dim=1)  # Get class with highest similarity
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            i += 1

    accuracy = correct / total
    return accuracy


def linear_probe(model, dataset, num_shots, preprocess, args):
    results = {}

    # Ensure class indices remain the same for reproducibility
    if args.dataset == 'eurosat':
        class_indices = {c: np.where(np.array(dataset.data["Label"]) == i)[0] for i, c in enumerate(dataset.classes)}
    else:
        class_indices = {c: np.where(np.array(dataset.targets) == i)[0] for i, c in enumerate(dataset.classes)}

    for shots in num_shots:
        print(f"Training linear probe with {shots} shots per class...")

        if args.dataset == 'eurosat':
            # Use the train and test splits created in CoOp
            train_file = f'{args.dataset_root}/split_fewshot/shot_{shots}-seed_{args.seed}-train.csv'
            print(train_file)
            train_data = CsvDataset(args.dataset_root, train_file, preprocess)
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        else:
            # Set the random seed before sampling to ensure consistency
            random.seed(args.seed)
            np.random.seed(args.seed)
            train_indices = [random.sample(list(class_indices[c]), min(shots, len(class_indices[c]))) for c in class_indices]
            train_indices = [i for sublist in train_indices for i in sublist]
            print('TRAIN INDICES')
            print(train_indices)

            train_subset = Subset(dataset, train_indices)
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

        X_train, Y_train = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(train_loader):
                images = images.to('cuda')

                if args.backbone == 'openclip':
                    features = model.encode_image(images).cpu().numpy()
                elif args.backbone == 'clip':
                    features = model.get_image_features(images).cpu().numpy()

                X_train.append(features)
                Y_train.append(labels.numpy())

        X_train = np.vstack(X_train)
        Y_train = np.concatenate(Y_train)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        clf = LogisticRegression(max_iter=1000, random_state=args.seed, C=0.316).fit(X_train, Y_train)

        #test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        if args.dataset == 'eurosat':
            # test_file = f'{args.dataset_root}/test.csv'
            # test_data = CsvDataset(test_file, preprocess)
            # test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
            test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        else:  
            # Get all indices and exclude training indices
            all_indices = set(range(len(dataset)))
            train_indices_set = set(train_indices)
            test_indices = list(all_indices - train_indices_set)  # Exclude training indices

            # Create the test subset
            test_subset = Subset(dataset, test_indices)
            test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

        X_test, Y_test = [], []
        with torch.no_grad():
            i = 0
            for images, labels in tqdm(test_loader):
                # if i > 5:
                #     break
                images = images.to('cuda')

                if args.backbone == 'openclip':
                    features = model.encode_image(images).cpu().numpy()
                elif args.backbone == 'clip':
                    features = model.get_image_features(images).cpu().numpy()

                X_test.append(features)
                Y_test.append(labels.numpy())
                
                i += 1

        X_test = np.vstack(X_test)
        Y_test = np.concatenate(Y_test)
        X_test = scaler.transform(X_test)
        Y_pred = clf.predict(X_test)

        acc = accuracy_score(Y_test, Y_pred)
        results[f"Linear probe {shots} shots"] = acc
        
#         # Perform logistic regression
#         classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
#         classifier.fit(train_features, train_labels)

#         # Evaluate using the logistic regression classifier
#         predictions = classifier.predict(test_features)
#         accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
#         print(f"Accuracy = {accuracy:.3f}")

    return results

#dataset, num_shots, preprocess, args
def coop_classification(model, dataloader, class_names, prototype_path, args):
    """
    Perform classification using learned text prototypes using CoOp.
    """
    model.eval()

    # Load learned class prototypes
    prototype_data = torch.load(prototype_path, map_location=args.device)
    class_prototypes = prototype_data["prototypes"].to(args.device)  # Shape: (num_classes, feature_dim)
    class_prototypes /= class_prototypes.norm(dim=-1, keepdim=True)  # Normalize prototypes
    
    # TODO append class name

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Prototype-Based Classification"):
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Compute image features
            if args.backbone == 'openclip':
                image_features = model.encode_image(images)
            elif args.backbone == 'clip':
                image_features = model.get_image_features(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize image features

            # Compute cosine similarity between image features and class prototypes
            similarity = image_features @ class_prototypes.T
            predictions = similarity.argmax(dim=1)  # Get the class with highest similarity

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy



if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda"
    
    if args.backbone_name == 'clip':
        args.backbone = 'clip'
        args.clip_path = None
    elif args.backbone_name == 'openclip':
        args.backbone = 'openclip'
        args.clip_path = None
    elif args.backbone_name == 'remoteclip':
        args.backbone = 'openclip'
        args.clip_path = PATH_CKPT_REMOTECLIP_14 
    elif args.backbone_name == 'georsclip':
        args.backbone = 'openclip'
        args.clip_path = PATH_CKPT_GEORSCLIP_14 
    elif args.backbone_name == 'clip-fmow':
        args.backbone = 'clip'
        args.clip_path = PATH_CKPT_CLIP14_FMOW 
    elif args.backbone_name == 'openclip-fmow':
        args.backbone = 'openclip'
        args.clip_path = PATH_CKPT_OPENCLIP14_FMOW 
        
    if args.dataset == 'eurosat':
        args.dataset_root = '/home/gridsan/manderson/ovdsat/data/eurosat/EuroSAT'
    elif args.dataset == 'resisc':
        args.dataset_root = '/home/gridsan/manderson/ovdsat/data/RESISC45/test'
    elif args.dataset == 'patternnet':
        args.dataset_root = '/home/gridsan/manderson/ovdsat/data/patternnet'
    elif args.dataset == 'aid':
        args.dataset_root = '/home/gridsan/manderson/ovdsat/data/AID'
    
    # Set a fixed seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    model, preprocess_train, preprocess_val, tokenize = get_model(args)
    model.eval()
    
    if args.dataset == 'eurosat': # override the dataloader
        # test_file = f'{args.dataset_root}/test.csv'
        # dataset = CsvDataset(test_file, preprocess)
        # dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        dataset, dataloader = get_csv_dataloader(args.dataset_root, f'{args.dataset_root}/test.csv', preprocess_val, args.batch_size, args.workers)
    else:
        dataset, dataloader = get_dataloader(args.dataset_root, preprocess_val, args.batch_size, args.workers)

    if args.mode == 'zero_shot':
        print("Performing Zero-Shot Classification...")
        zero_shot_acc = zero_shot_classification(model, dataloader, dataset.classes, tokenize, args)
        print(f"Zero-Shot Accuracy: {zero_shot_acc * 100:.2f}%")
        save_results_to_file(args, zero_shot_acc)   
    elif args.mode == 'linear_probe':
        print("Performing Linear Probing...")
        linear_probe_results = linear_probe(model, dataset, args.num_shots, preprocess_train, args)
        for k, v in linear_probe_results.items():
            print(f"{k}: {v * 100:.2f}%")
        save_results_to_file(args, linear_probe_results)
