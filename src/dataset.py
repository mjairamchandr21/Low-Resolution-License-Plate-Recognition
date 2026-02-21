import torch
from torch.utils.data import Dataset
import cv2
import os
import pandas as pd
import numpy as np
import random

class LPRDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=(110, 40)):
        self.data = pd.read_csv(csv_file)
        self.root_dir = os.path.abspath(root_dir)
        self.target_size = target_size 

    def __len__(self):
        return len(self.data)

    def _safe_read(self, path):
        if not path or pd.isna(path):
            return None
            
        full_path = os.path.normpath(os.path.join(self.root_dir, path))
        
        # Check if file exists before even trying to read
        if not os.path.exists(full_path):
            return None

        img = cv2.imread(full_path)
        if img is None or img.size == 0:
            return None
        
        img = cv2.resize(img, self.target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            
            # Try to load LR
            lr_img = self._safe_read(row['lr_full_path'])
            if lr_img is None:
                raise ValueError(f"LR Image missing: {row['lr_full_path']}")

            lr_tensor = torch.from_numpy(lr_img)

            # Scenario-A: Load HR if path exists
            if pd.notna(row['hr_full_path']):
                hr_img = self._safe_read(row['hr_full_path'])
                if hr_img is not None:
                    return lr_tensor, torch.from_numpy(hr_img)
            
            # Scenario-B: Return Text
            return lr_tensor, str(row['plate_text'])
            
        except Exception as e:
            # SILENT FAIL & RETRY: If an image is broken, pick a random new index
            # This prevents the "cvtColor" crash from stopping your training
            new_idx = random.randint(0, len(self.data) - 1)
            return self.__getitem__(new_idx)