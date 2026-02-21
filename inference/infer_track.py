"""
Track-level inference script.

Usage (local, Windows):
    python -m inference.infer_track \
        --model_path models/crnn_best.pth \
        --data_path  data/raw/wYe7pBJ7-train/train \
        --num_tracks 200

Usage (Colab):
    !python -m inference.infer_track \
        --model_path /content/drive/MyDrive/LPR_Project/crnn_best.pth \
        --data_path  /content/raw/wYe7pBJ7-train/train
"""

import argparse
import os
import cv2
import numpy as np
import torch

from models.crnn import CRNN
from training.dataset import NUM_CLASSES, idx_to_char
from utils.data_loader import get_all_tracks, load_track
from utils.aggregator import aggregate_track


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Image preprocessing ─────────────────────────────────────────────────────

def preprocess_image(img_path: str) -> torch.Tensor:
    """
    Load a single LR image and return a (1, 1, 32, 128) tensor
    using the same aspect-ratio-preserving resize + pad logic as dataset.py.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img.shape
    new_h = 32
    new_w = int(w * (new_h / h))
    img = cv2.resize(img, (new_w, new_h))

    if new_w < 128:
        pad = 128 - new_w
        img = np.pad(img, ((0, 0), (0, pad)), mode="constant", constant_values=0)
    else:
        img = cv2.resize(img, (128, 32))

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)          # (1, H, W)
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 1, H, W)


# ── Inference helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def get_logits(model: CRNN, img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run model on a single image tensor (1, 1, 32, 128).
    Returns logits of shape (T, num_classes).
    """
    img_tensor = img_tensor.to(DEVICE)
    logits = model(img_tensor)          # (1, T, C)
    return logits.squeeze(0)            # (T, C)


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_inference(model_path: str, data_path: str, num_tracks: int = None):
    # Load model
    print(f"Loading model from: {model_path}")
    model = CRNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Discover tracks
    all_tracks = get_all_tracks(data_path)
    if num_tracks:
        all_tracks = all_tracks[:num_tracks]

    print(f"Evaluating on {len(all_tracks)} tracks  |  device={DEVICE}\n")
    print(f"{'Track ID':<20} {'GT':<12} {'Pred':<12} {'Conf':>6}  Match")
    print("-" * 60)

    correct   = 0
    total     = 0
    skipped   = 0

    for track_path in all_tracks:
        sample = load_track(track_path)

        gt_text  = (sample["plate_text"] or "").upper().replace(" ", "")
        images   = sample["images"]

        if not images or not gt_text:
            skipped += 1
            continue

        # Run model on every frame
        logits_list = []
        for img_path in images:
            try:
                img_tensor = preprocess_image(img_path)
                logits     = get_logits(model, img_tensor)
                logits_list.append(logits)
            except Exception:
                pass   # skip corrupt frames silently

        if not logits_list:
            skipped += 1
            continue

        # Aggregate across frames
        pred_text, confidence = aggregate_track(logits_list, idx_to_char)

        match = pred_text == gt_text
        if match:
            correct += 1
        total += 1

        status = "✅" if match else "❌"
        print(f"{sample['track_id']:<20} {gt_text:<12} {pred_text:<12} {confidence:>6.3f}  {status}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Tracks evaluated : {total}")
    print(f"Tracks skipped   : {skipped}")
    if total:
        print(f"Track-level Exact Match Accuracy: {correct / total:.4f}  ({correct}/{total})")
    print("=" * 60)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track-level LPR inference")
    parser.add_argument(
        "--model_path", type=str,
        default="/content/drive/MyDrive/LPR_Project/crnn_best.pth",
        help="Path to saved model checkpoint (.pth)"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="/content/raw/wYe7pBJ7-train/train",
        help="Path to train data root"
    )
    parser.add_argument(
        "--num_tracks", type=int, default=None,
        help="Limit evaluation to first N tracks (default: all)"
    )
    args = parser.parse_args()

    run_inference(args.model_path, args.data_path, args.num_tracks)
