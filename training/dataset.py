import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


# Character set
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for CTC blank
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for blank


class LPRDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []
        self.data_root = data_root
        self._load_samples()

    def _load_samples(self):
        for scenario in os.listdir(self.data_root):
            scenario_path = os.path.join(self.data_root, scenario)
            if not os.path.isdir(scenario_path):
                continue

            for layout in os.listdir(scenario_path):
                layout_path = os.path.join(scenario_path, layout)
                if not os.path.isdir(layout_path):
                    continue

                for track in os.listdir(layout_path):
                    track_path = os.path.join(layout_path, track)
                    if not os.path.isdir(track_path):
                        continue

                    ann_path = os.path.join(track_path, "annotations.json")
                    if not os.path.exists(ann_path):
                        continue

                    with open(ann_path, "r") as f:
                        ann = json.load(f)
                        plate_text = ann["plate_text"].upper().replace(" ", "")

                    for file in os.listdir(track_path):
                        if file.startswith("lr-") and file.endswith(".png"):
                            img_path = os.path.join(track_path, file)
                            self.samples.append((img_path, plate_text))

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        encoded = []
        for c in text:
            if c in char_to_idx:
                encoded.append(char_to_idx[c])
        # skip unknown chars silently
        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        
        img_path, text = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to fixed size
        h, w = img.shape
        new_h = 32
        scale = new_h / h
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h))

        # pad to width 128
        if new_w < 128:
            pad = 128 - new_w
            img = np.pad(img, ((0,0),(0,pad)), mode='constant', constant_values=0)
        else:
            img = cv2.resize(img, (128, 32))

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W)

        img = torch.tensor(img, dtype=torch.float32)

        text_encoded = self.encode_text(text)
        if len(text_encoded) == 0:
            raise ValueError(f"Empty label after encoding: {text}")
        return img, text_encoded, len(text_encoded)