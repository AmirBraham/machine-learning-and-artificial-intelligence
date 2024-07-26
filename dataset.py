import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class PhotoMonetDataset(Dataset):
    def __init__(self,root_monet,root_photo,transform=None) -> None:
        super().__init__()
        #root_monet is the folder path to monet style images
        #root_photo is the path to normal photos
        self.root_monet = root_monet
        self.root_photo = root_photo
        self.monet_images = os.listdir(self.root_monet)
        self.photo_images = os.listdir(self.root_photo)
        self.monet_len = len(self.monet_images)
        self.photo_len = len(self.photo_images)
        self.dataset_length = max(self.monet_len,self.photo_len)
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index):
        monet_img = self.monet_images[index % self.dataset_length]
        photo_img = self.photo_images[index % self.dataset_length]
        
        monet_path = os.path.join(self.root_monet,monet_img)
        photo_path = os.path.join(self.root_photo,photo_img)
        
        monet_img = np.array(Image.open(monet_path).convert("RGB"))
        photo_img = np.array(Image.open(photo_path).convert("RGB"))
        if(self.transform):
            augmentations = self.transform(image=photo_img, image0=monet_img)
            photo_img = augmentations["image"]
            monet_img = augmentations["image0"]
        return monet_img,photo_img
        
        