# %%
import os
import torch
from torch.utils.data import TensorDataset, DataLoader , Dataset , random_split
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# %%
class FishDataset(Dataset):
    def __init__(self, base_path, transform=None, mask_transform=None):
        self.base_path = base_path
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get all fish categories
        self.fish_categories = [d for d in os.listdir(base_path) 
                               if os.path.isdir(os.path.join(base_path, d))]
        
        # Create a list of all image paths and their corresponding info
        self.samples = []
        
        for idx, category in enumerate(self.fish_categories):
            img_path = os.path.join(base_path, category, category)
            mask_path = os.path.join(base_path, category, f"{category} GT")
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue
                
            image_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
            image_files.sort(key=lambda x: int(x.split('.')[0]))
            
            for img_file in image_files:
                img_full_path = os.path.join(img_path, img_file)
                mask_full_path = os.path.join(mask_path, img_file)
                
                if os.path.exists(mask_full_path):
                    self.samples.append({
                        'image_path': img_full_path,
                        'mask_path': mask_full_path,
                        'label': idx,
                        'category': category
                    })
        
        # print(f"Found {len(self.samples)} samples across {len(self.fish_categories)} categories")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert("L")
        if self.transform:
            image = self.transform(image)
        
        # Load mask
        mask = Image.open(sample['mask_path']).convert("1")
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # Return image, mask, and label
        return image, mask, torch.tensor(sample['label'])

class TestFishDataset(Dataset):
    """Dataset for test images without masks"""
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.transform = transform
        
        # Get all fish categories
        self.fish_categories = [d for d in os.listdir(base_path) 
                               if os.path.isdir(os.path.join(base_path, d))]
        
        # Create category to index mapping
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.fish_categories)}
        
        # Create a list of all image paths
        self.samples = []
        
        for category in self.fish_categories:
            category_path = os.path.join(base_path, category)
            
            if not os.path.exists(category_path):
                continue
                
            image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_full_path = os.path.join(category_path, img_file)
                self.samples.append({
                    'image_path': img_full_path,
                    'label': self.category_to_idx[category],
                    'category': category
                })
        
        # print(f"Found {len(self.samples)} test samples across {len(self.fish_categories)} categories")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert("L")
        if self.transform:
            image = self.transform(image)
        
        # Return image and label (no mask for test data)
        return image, torch.tensor(sample['label'])


# Alternative function if you want separate train/val creation
def create_fish_dataloaders_separate(train_batch_size=64, val_batch_size=64, val_split=0.2):
    """Create train and validation dataloaders separately"""
    
    base_path = "/home/krrish/home/desktop/fish_repo/fish_project/src/data/raw/2/2/Fish_Dataset/Fish_Dataset"
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    
    # Create full dataset
    full_dataset = FishDataset(base_path, transform, mask_transform)
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, val_dataloader, full_dataset.fish_categories

def create_test_dataloader(batch_size=32):
    """Create test dataloader separately"""
    
    test_base_path = "/home/krrish/home/desktop/fish_repo/fish_project/src/data/raw/2/2/NA_Fish_Dataset"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_dataset = TestFishDataset(test_base_path, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return test_dataloader, test_dataset.fish_categories


    
  


# %%
