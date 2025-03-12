import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import open_clip
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime

# Set a fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

PATH_CKPT_CLIP14 = 'weights/clip-vit-large-patch14'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, choices=['RN50', 'ViT-B-32', 'ViT-L-14'], help="Backbone name")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--clip-path", default=None, type=str, help="Path to CLIP weight")
    parser.add_argument("--backbone", default='openclip', type=str, help="Either openclip or clip")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--num-shots", type=int, nargs='+', default=[5, 10, 30], help="Number of samples per class for linear probe")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")  # New argument
    args = parser.parse_args()
    return args


def save_results_to_file(args, zero_shot_acc, linear_probe_results):
    """Saves experiment parameters and results to a text file."""
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"experiment_results_{timestamp}.txt")

    with open(results_file, "w") as f:
        f.write("=== Experiment Parameters ===\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        f.write("\n=== Zero-Shot Classification Results ===\n")
        f.write(f"Zero-Shot Accuracy: {zero_shot_acc * 100:.2f}%\n")

        f.write("\n=== Linear Probe Results ===\n")
        for k, v in linear_probe_results.items():
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

    return CLIP_model, preprocess_train, preprocess_val, tokenize


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
        for images, labels in tqdm(dataloader, desc="Zero-Shot Classification"):
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

    accuracy = correct / total
    return accuracy


def linear_probe(model, dataset, num_shots, preprocess, args):
    results = {}

    # Ensure class indices remain the same for reproducibility
    class_indices = {c: np.where(np.array(dataset.targets) == i)[0] for i, c in enumerate(dataset.classes)}

    for shots in num_shots:
        print(f"Training linear probe with {shots} shots per class...")

        # Set the random seed before sampling to ensure consistency
        random.seed(SEED)
        np.random.seed(SEED)
        train_indices = [random.sample(list(class_indices[c]), min(shots, len(class_indices[c]))) for c in class_indices]
        train_indices = [i for sublist in train_indices for i in sublist]
        
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

        clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train, Y_train)

        test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        X_test, Y_test = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to('cuda')

                if args.backbone == 'openclip':
                    features = model.encode_image(images).cpu().numpy()
                elif args.backbone == 'clip':
                    features = model.get_image_features(images).cpu().numpy()

                X_test.append(features)
                Y_test.append(labels.numpy())

        X_test = np.vstack(X_test)
        Y_test = np.concatenate(Y_test)
        X_test = scaler.transform(X_test)
        Y_pred = clf.predict(X_test)

        acc = accuracy_score(Y_test, Y_pred)
        results[f"Linear probe {shots} shots"] = acc

    return results


if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda"
    model, preprocess_train, preprocess_val, tokenize = get_model(args)
    dataset, dataloader = get_dataloader(args.dataset_root, preprocess_val, args.batch_size, args.workers)

    print("Performing Zero-Shot Classification...")
    zero_shot_acc = zero_shot_classification(model, dataloader, dataset.classes, tokenize, args)
    print(f"Zero-Shot Accuracy: {zero_shot_acc * 100:.2f}%")

    print("Performing Linear Probing...")
    linear_probe_results = linear_probe(model, dataset, args.num_shots, preprocess_train, args)
    for k, v in linear_probe_results.items():
        print(f"{k}: {v * 100:.2f}%")
        
    save_results_to_file(args, zero_shot_acc, linear_probe_results)
