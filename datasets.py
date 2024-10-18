# datasets.py
import os
import random
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class custom_dataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_names = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).filter(ImageFilter.UnsharpMask(radius=3, percent=100, threshold=0))
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])
        mask = Image.open(mask_path)

        seed = random.randint(0, 2 ** 32)
        
        if self.image_transform:
            torch.manual_seed(seed)
            image = self.image_transform(image)

        if self.mask_transform:
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)

        return image, mask


# Define the transformations
def get_transforms():
    image_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(15)], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    mask_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(15)], p=0.8),
        transforms.ToTensor()
    ])

    val_image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_mask_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    return image_transforms, mask_transforms, val_image_transforms, val_mask_transforms
