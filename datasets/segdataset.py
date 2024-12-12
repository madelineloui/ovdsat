import torch
from torch.utils.data import Dataset
from torchvision import transforms
import rasterio as rio
import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DynnetDataModule(pl.LightningDataModule):
    def __init__(self, train_split, val_split, test_split, batch_size=32, crop_size=512, num_workers=4):
        super().__init__()
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.num_workers = num_workers
        self.time_series_length = DynnetEval(txt_file=self.test_split, crop_size=self.crop_size).num_crops_per_image[0]

    def setup(self, stage=None):
        # Setup train and validation datasets
        self.train_data = DynnetData(txt_file=self.train_split, crop_size=self.crop_size)
        self.val_data = DynnetData(txt_file=self.val_split, crop_size=self.crop_size)
        self.test_data = DynnetEval(txt_file=self.test_split, crop_size=self.crop_size)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=self.num_workers, shuffle=False)


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
        
        # Normalize image
        mean = torch.mean(image)
        std = torch.std(image)
        image = (image - mean) / std

        return image, mask

    def get_id(self, idx):
        return self.samples[idx]
    
class DynnetData(SegData):
     def __init__(self, txt_file, crop_size=512):
        super().__init__(data_dir=None, crop_size=crop_size)
        self.txt_file = txt_file
        with open(self.txt_file, 'r') as file:
            lines = file.readlines()
            self.samples = [line.strip().split() for line in lines]
    
'''    
class DynnetEval(DynnetData):
    def __init__(self, txt_file, crop_size=512):
        super().__init__(txt_file, crop_size=crop_size)
        self.crop_size = crop_size
        self.num_crops_per_image = 4  # Number of crops per image
        self.total_crops = len(self.samples) * self.num_crops_per_image

    def __getitem__(self, idx):
        image_idx = idx % len(self.samples)
        patch_idx = idx // len(self.samples)
        img_path, mask_path = self.samples[image_idx]

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
    
    def __len__(self):
        return self.total_crops
    
    def get_id(self, idx):
        patch_idx = idx // len(self.samples)
        return (self.samples[idx], f'crop {patch_idx}')
'''
    
    
class DynnetEval(DynnetData):
    def __init__(self, txt_file, crop_size=512):
        super().__init__(txt_file, crop_size=crop_size)
        self.crop_size = crop_size

        # Calculate the total number of crops based on image dimensions
        self.num_crops_per_image = []
        self.all_crops = []
        for img_path, mask_path in self.samples:
            with rio.open(img_path) as src:
                height, width = src.height, src.width
            num_rows = height // crop_size
            num_cols = width // crop_size
            for row in range(num_rows):
                for col in range(num_cols):
                    self.all_crops.append((img_path, mask_path, row, col))
            self.num_crops_per_image.append(num_rows*num_cols)

    def __len__(self):
        return len(self.all_crops)

    def __getitem__(self, idx):
        img_path, mask_path, row, col = self.all_crops[idx]

        # Read tifs to numpy with rasterio
        with rio.open(img_path) as src:
            im = torch.tensor(src.read()).float()
        with rio.open(mask_path) as src:
            lb = torch.tensor((src.read() / 255).astype(int)).float()

        # Calculate crop coordinates
        start_row = row * self.crop_size
        start_col = col * self.crop_size

        # Extract the crop
        image = im[:, start_row:start_row + self.crop_size, start_col:start_col + self.crop_size]
        mask = lb[:, start_row:start_row + self.crop_size, start_col:start_col + self.crop_size]

        # Normalize image
        mean = torch.mean(image)
        std = torch.std(image)
        image = (image - mean) / std

        return image, mask

    def get_id(self, idx):
        img_path, _, row, col = self.all_crops[idx]
        return f"{img_path}, crop ({row}, {col})"
