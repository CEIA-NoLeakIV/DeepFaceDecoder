import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

class VGGFace2Dataset(Dataset):
    """VGGFace2 dataset for DFD training"""
    
    def __init__(self, data_root, split='train', transform=None, image_size=224):
        self.data_root = os.path.join(data_root, split)
        self.transform = transform
        self.image_size = image_size
        
        # Collect all image paths
        self.image_paths = []
        self.identities = []
        
        for identity_dir in os.listdir(self.data_root):
            identity_path = os.path.join(self.data_root, identity_dir)
            if os.path.isdir(identity_path):
                for img_name in os.listdir(identity_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(identity_path, img_name))
                        self.identities.append(identity_dir)
        
        print(f"Loaded {len(self.image_paths)} images from {split} set")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        identity = self.identities[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'identity': identity,
            'path': img_path
        }

def get_data_transforms(image_size=224, augment=True):
    """Get data transforms for training and validation"""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_data_loaders(config):
    """Create data loaders for training and validation"""
    
    train_transform, val_transform = get_data_transforms(
        config.IMAGE_SIZE, 
        config.USE_AUGMENTATION
    )
    
    # Create datasets
    train_dataset = VGGFace2Dataset(
        config.DATA_ROOT, 
        split='train', 
        transform=train_transform,
        image_size=config.IMAGE_SIZE
    )
    
    val_dataset = VGGFace2Dataset(
        config.DATA_ROOT, 
        split='test', 
        transform=val_transform,
        image_size=config.IMAGE_SIZE
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader