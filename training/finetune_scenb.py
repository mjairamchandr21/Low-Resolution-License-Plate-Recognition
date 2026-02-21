"""
Phase 2 fine-tuning on Scenario-B only.
Run after Phase 1 training is complete.

Usage:
    !python training/finetune_scenb.py
"""

import os
import torch
from torch.utils.data import DataLoader, random_split

from training.dataset import LPRDataset, NUM_CLASSES
from training.train import run_training, collate_fn, DATA_PATH, SAVE_DIR, DEVICE
from models.crnn import CRNN


# Auto-detect Scenario-B folder name
scenarios = [
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
]
print(f"Available scenarios: {scenarios}")

scen_b_name = None
for s in scenarios:
    if "b" in s.lower():
        scen_b_name = s
        break

if scen_b_name is None:
    print(f"ERROR: No Scenario-B found. Scenarios available: {scenarios}")
    exit(1)

print(f"Using: '{scen_b_name}'")

# Build datasets
train_ds = LPRDataset(DATA_PATH, augment=True,  scenario_filter=scen_b_name)
val_ds   = LPRDataset(DATA_PATH, augment=False, scenario_filter=scen_b_name)
print(f"Scenario-B samples: {len(train_ds)}")

n_train  = int(0.9 * len(train_ds))
n_val    = len(train_ds) - n_train
tr_split, va_split = random_split(
    train_ds, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
va_clean = torch.utils.data.Subset(val_ds, va_split.indices)

train_loader = DataLoader(tr_split, batch_size=64, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
val_loader   = DataLoader(va_clean,  batch_size=64, shuffle=False,
                          collate_fn=collate_fn, num_workers=2)

# Load best Phase-1 model
model = CRNN(NUM_CLASSES, dropout=0.3).to(DEVICE)
checkpoint = f"{SAVE_DIR}/crnn_best_main.pth"
model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
print(f"Loaded checkpoint: {checkpoint}")

# Fine-tune
run_training(model, train_loader, val_loader, epochs=5, lr=1e-4, tag="scenB")

print(f"\nDone. Best Scenario-B model -> {SAVE_DIR}/crnn_best_scenB.pth")
print("Use crnn_best_scenB.pth for submission.")
