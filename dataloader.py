import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

class SRGANDataset(Dataset):
    def __init__(self, hr_dir, hr_size=96, scale=4, mode='train'):
        self.hr_dir = hr_dir
        self.crop_size = hr_size
        self.scale = scale
        self.mode = mode
        
        # Get all image files
        self.image_files = [f for f in os.listdir(hr_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {hr_dir}")
        print(f"Found {len(self.image_files)} images")

        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.hr_dir, self.image_files[idx])
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a different image if this one fails
            return self.__getitem__((idx + 1) % len(self))
        
        if self.mode == 'train':
            w, h = img.size
            if w < self.crop_size or h < self.crop_size:
                return self.__getitem__((idx + 1) % len(self))
            
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            hr_img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
            
            # Random Augmentation
            ran = random.random()
            if ran < 0.4:
                pass
            elif ran < 0.7:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
            elif ran < 0.9:
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                hr_img = hr_img.rotate(90)
        
        else:
            w, h = img.size
            if w < self.crop_size or h < self.crop_size:
                # If image is too small, resize it
                scale_factor = max(self.crop_size / w, self.crop_size / h)
                new_w = int(w * scale_factor) + 1
                new_h = int(h * scale_factor) + 1
                img = img.resize((new_w, new_h), Image.BICUBIC)
                w, h = img.size
            
            # Center crop for consistent validation
            x = (w - self.crop_size) // 2
            y = (h - self.crop_size) // 2
            hr_img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        
        # Convert to tensor
        hr = self.to_tensor(hr_img)
        
        # Create LR
        lr_img = hr_img.resize(
            (self.crop_size // self.scale, self.crop_size // self.scale),
            Image.BICUBIC
        )
        lr = self.to_tensor(lr_img)
        
        # HR in range [-1 1]
        hr = hr * 2.0 - 1.0      

        return lr, hr

# Create Dataloader
# [shuffle=True] to prevent model from memorizing, start index from 0
def DataloaderInit(hr_dir, batch_size=16, crop_size=96, scale=4,
                   num_workers=4, shuffle=True, mode='train'):
    
    dataset = SRGANDataset(hr_dir, crop_size, scale, mode)
    
    if mode == 'val':
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,      # parallel loading
        pin_memory=True,              # faster GPU transfer
        drop_last=(mode == 'train')   # Drop last incomplete batch
    )
    
    return dataloader
