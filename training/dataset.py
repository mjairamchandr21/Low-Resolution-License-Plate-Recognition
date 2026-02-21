import os
import json
import cv2
import random
import torch
from torch.utils.data import Dataset
import numpy as np


# Character set
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for CTC blank
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for blank


class LPRDataset(Dataset):
    def __init__(self, data_root, augment=True, scenario_filter=None):
        """
        Args:
            data_root       : path to train/ folder
            augment         : if True, apply random augmentations during __getitem__
            scenario_filter : if set (e.g. "Scenario-B"), only load that scenario
        """
        self.samples         = []
        self.data_root       = data_root
        self.augment         = augment
        self.scenario_filter = scenario_filter
        self._load_samples()

    def _load_samples(self):
        for scenario in os.listdir(self.data_root):
            scenario_path = os.path.join(self.data_root, scenario)
            if not os.path.isdir(scenario_path):
                continue
            # Optional domain filter
            if self.scenario_filter and scenario != self.scenario_filter:
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

                    IMAGE_EXTS = (".png", ".jpg", ".jpeg")
                    for file in os.listdir(track_path):
                        if file.startswith("lr-") and \
                                os.path.splitext(file)[1].lower() in IMAGE_EXTS:
                            img_path = os.path.join(track_path, file)
                            self.samples.append((img_path, plate_text))

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        encoded = []
        for c in text:
            if c in char_to_idx:
                encoded.append(char_to_idx[c])
        return torch.tensor(encoded, dtype=torch.long)

    # ── Augmentations ────────────────────────────────────────────────────────

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a uint8 grayscale image."""

        # 1. JPEG compression simulation (mimics test-set domain)
        if random.random() < 0.5:
            quality = random.randint(40, 85)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode(".jpg", img, encode_param)
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

        # 2. Random brightness / contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.6, 1.4)   # contrast
            beta  = random.randint(-30, 30)     # brightness
            img   = np.clip(img.astype(np.int32) * alpha + beta, 0, 255).astype(np.uint8)

        # 3. Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(2, 8), img.shape).astype(np.int32)
            img   = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

        # 4. Mild Gaussian blur (simulates motion / focus blur)
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            img   = cv2.GaussianBlur(img, (ksize, ksize), 0)

        return img

    # ── __getitem__ ──────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Aspect-ratio-preserving resize to H=32, pad W to 128
        h, w   = img.shape
        new_h  = 32
        new_w  = int(w * (new_h / h))
        img    = cv2.resize(img, (new_w, new_h))

        if new_w < 128:
            img = np.pad(img, ((0, 0), (0, 128 - new_w)),
                         mode='constant', constant_values=0)
        else:
            img = cv2.resize(img, (128, 32))

        # Apply augmentations (training only)
        if self.augment:
            img = self._augment(img)

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)      # (1, H, W)
        img = torch.tensor(img, dtype=torch.float32)

        text_encoded = self.encode_text(text)
        if len(text_encoded) == 0:
            raise ValueError(f"Empty label after encoding: {text}")
        return img, text_encoded, len(text_encoded)