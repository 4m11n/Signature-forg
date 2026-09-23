import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import albumentations as A

class SignaturePairDataset(Dataset):
    def __init__(self, csv_path, img_size=105, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.augment = augment
        self.resize = T.Resize((img_size, img_size))
        if augment:
            self.aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.0, p=0.3),
            ])

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        img = Image.open(path).convert("L")
        img = self.resize(img)
        if self.augment:
            img_np = np.array(img)
            img_np = self.aug(image=img_np)["image"]
            tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0
        else:
            tensor = T.ToTensor()(img)
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_a = self._load(row["img_a"])
        img_b = self._load(row["img_b"])
        label = row["label"]
        return img_a, img_b, label