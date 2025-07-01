# %%
import os
from torch.utils.data import DataLoader, Dataset ,random_split
from PIL import Image
import torchvision.transforms as transforms

# %%
class FishSegmentationDataset(Dataset):
    def __init__(self, base_path, transform=None, mask_transform=None):
        self.base_path = base_path
        self.transform = transform
        self.mask_transform = mask_transform
        self.fish_categories = [d for d in os.listdir(base_path) 
                               if os.path.isdir(os.path.join(base_path, d))]
        
        
        self.samples = []
        
        for category in self.fish_categories:
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
                        'mask_path': mask_full_path
                    })
    
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
        
        # Return only image and mask for segmentation
        return image, mask

class TestSegmentationDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.transform = transform
        self.fish_categories = [d for d in os.listdir(base_path) 
                               if os.path.isdir(os.path.join(base_path, d))]
        
       
        self.samples = []
        
        for category in self.fish_categories:
            category_path = os.path.join(base_path, category)
            
            if not os.path.exists(category_path):
                continue
                
            image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_full_path = os.path.join(category_path, img_file)
                self.samples.append({
                    'image_path': img_full_path
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert("L")
        if self.transform:
            image = self.transform(image)
        
        return image

def TrainSegmentationDataloader(train_batch_size=96, val_batch_size=96, val_split=0.2):
    base_path = "../../src/data/Fish_Dataset/A_Fish_Dataset"
    
    # transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    
    # full dataset
    full_dataset = FishSegmentationDataset(base_path, transform, mask_transform)
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # training dataloaders and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, val_dataloader

def TestSegmentationDataloader(batch_size=32):
    test_base_path = "../../src/data/Fish_Dataset/NA_Fish_Dataset"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_dataset = TestSegmentationDataset(test_base_path, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return test_dataloader

def check_dataloaders():
    """Simple check of dataloader dimensions and pixel ranges"""
    
    # Get dataloaders
    train_dl, val_dl = TrainSegmentationDataloader()
    test_dl = TestSegmentationDataloader()
    
    print("TRAIN DATALOADER:")
    images, masks = next(iter(train_dl))
    print(f"Images: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Masks: {masks.shape}, range: [{masks.min():.3f}, {masks.max():.3f}]")
    
    print("\nVALIDATION DATALOADER:")
    images, masks = next(iter(val_dl))
    print(f"Images: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Masks: {masks.shape}, range: [{masks.min():.3f}, {masks.max():.3f}]")
    
    print("\nTEST DATALOADER:")
    images = next(iter(test_dl))
    print(f"Images: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")

# Run check
check_dataloaders()
# %%
