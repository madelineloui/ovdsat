import torch
from torch.utils.data import Dataset
from torchvision import transforms
import rasterio as rio
import glob

class SegData(Dataset):
    def __init__(self, data_dir, crop_size=512):
        """
        Args:
            txt_file: path to the text file containing paths of images and masks
            crop_size: crop size
            eval: if in eval mode, will not apply random transforms to images
        """
        self.data_dir = data_dir
        self.mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        self.std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        
        img_paths = glob.glob(f'{self.data_dir}/images/**/*.tif', recursive=True)
        img_paths.sort()
        label_paths = glob.glob(f'{self.data_dir}/labels/**/*.tif', recursive=True)
        label_paths.sort()
        self.samples = list(zip(img_paths, label_paths))

        self.composed_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=crop_size, scale=(0.1, 1.0), ratio=(0.6, 1.4)),
        ])
        
        self.norm = transforms.Normalize(mean=self.mean, std=self.std)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label_path = self.samples[idx]

        # Read tifs to numpy with rasterio
        with rio.open(img_path) as src:
            im = torch.tensor(src.read()).float()
        with rio.open(label_path) as src:
            lb = torch.tensor((src.read() / 255).astype(int)).float()

        stacked = torch.cat((im, lb), dim=0)
        stacked = self.composed_transforms(stacked)
        image = stacked[:4,:,:]
        mask = stacked[4:,:,:]

        #image = self.norm(image)
        
        # Normalize image
        mean = torch.mean(image)
        std = torch.std(image)
        image = (image - mean) / std

        return image, mask

    def get_id(self, idx):
        return self.samples[idx]
    
    
class DynnetEval(Dataset):
    def __init__(self, txt_file, crop_size=512):
        self.txt_file = txt_file
        with open(self.txt_file, 'r') as file:
            lines = file.readlines()
            self.image_paths = [line.strip().split() for line in lines]
        self.crop_size = crop_size
        self.num_crops_per_image = 4  # Number of crops per image
        self.total_crops = len(self.image_paths) * self.num_crops_per_image

    def __len__(self):
        return self.total_crops

    def __getitem__(self, idx):
        image_idx = idx % len(self.image_paths)
        patch_idx = idx // len(self.image_paths)
        img_path, mask_path = self.image_paths[image_idx]

        # Read tifs to numpy with rasterio
        with rio.open(img_path) as src:
            im = torch.tensor(src.read()).float()
        with rio.open(mask_path) as src:
            lb = torch.tensor((src.read() / 255).astype(int)).float()

        # Calculate patch coordinates
        row = (patch_idx // 4) * self.crop_size
        col = (patch_idx % 4) * self.crop_size

        # Crop the patch from the image
        image = im[:, row:row + self.crop_size, col:col + self.crop_size]
        mask = lb[:, row:row + self.crop_size, col:col + self.crop_size]

        # Normalize image
        mean = torch.mean(image)
        std = torch.std(image)
        image = (image - mean) / std

        return image, mask

    def get_id(self, idx):
        return self.image_paths[idx % len(self.image_paths)]