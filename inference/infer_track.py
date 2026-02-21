"""
Track-level inference script with CTC beam search + length enforcement.

Usage (Colab):
    !python -m inference.infer_track \
        --model_path /content/drive/MyDrive/LPR_Project/crnn_best.pth \
        --data_path  /content/raw/wYe7pBJ7-train/train \
        --num_tracks 200 \
        --beam_width 10
"""

import argparse
import os
import cv2
import numpy as np
import torch

from models.crnn import CRNN
from training.dataset import NUM_CLASSES, idx_to_char
from utils.data_loader import get_all_tracks, load_track
from utils.aggregator import aggregate_track, PLATE_LENGTH


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_image(img_path: str) -> torch.Tensor:
    """
    Aspect-ratio-preserving resize to height=32, pad width to 128.
    Returns a (1, 1, 32, 128) tensor.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img.shape
    new_h = 32
    new_w = int(w * (new_h / h))
    img   = cv2.resize(img, (new_w, new_h))

    if new_w < 128:
        pad = 128 - new_w
        img = np.pad(img, ((0, 0), (0, pad)), mode="constant", constant_values=0)
    else:
        img = cv2.resize(img, (128, 32))

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)                          # (1, H, W)
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0) # (1, 1, H, W)


# ── Model forward ────────────────────────────────────────────────────────────

@torch.no_grad()
def get_logits(model: CRNN, img_tensor: torch.Tensor) -> torch.Tensor:
    """Returns logits shape (T, num_classes)."""
    img_tensor = img_tensor.to(DEVICE)
    logits = model(img_tensor)   # (1, T, C)
    return logits.squeeze(0)     # (T, C)


# ── Main evaluation ──────────────────────────────────────────────────────────

def run_inference(model_path: str, data_path: str,
                  num_tracks: int = None, beam_width: int = 10):

    print(f"Loading model from: {model_path}")
    model = CRNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_tracks = get_all_tracks(data_path)
    if num_tracks:
        all_tracks = all_tracks[:num_tracks]

    print(f"Evaluating {len(all_tracks)} tracks  |  "
          f"device={DEVICE}  |  beam_width={beam_width}\n")
    print(f"{'Track ID':<20} {'GT':<12} {'Pred':<12} {'Conf':>6}  {'Len':>3}  Match")
    print("-" * 65)

    correct         = 0
    total           = 0
    skipped         = 0
    len_enforced    = 0  # how many times length-enforcement changed the pick
    wrong_len_pred  = 0  # how many preds still have wrong length after enforcement

    for track_path in all_tracks:
        sample  = load_track(track_path)
        gt_text = (sample["plate_text"] or "").upper().replace(" ", "")
        images  = sample["images"]

        if not images or not gt_text:
            skipped += 1
            continue

        # Per-frame logits
        logits_list = []
        for img_path in images:
            try:
                img_tensor = preprocess_image(img_path)
                logits_list.append(get_logits(model, img_tensor))
            except Exception:
                pass

        if not logits_list:
            skipped += 1
            continue

        # Aggregate with beam search + length enforcement
        pred_text, confidence, was_length_enforced = aggregate_track(
            logits_list, idx_to_char,
            beam_width=beam_width,
            expected_length=PLATE_LENGTH,
        )

        match = pred_text == gt_text
        if match:
            correct += 1
        if was_length_enforced:
            len_enforced += 1
        if len(pred_text) != PLATE_LENGTH:
            wrong_len_pred += 1

        total += 1

        len_flag = "✓" if was_length_enforced else " "
        status   = "✅" if match else "❌"
        print(f"{sample['track_id']:<20} {gt_text:<12} {pred_text:<12} "
              f"{confidence:>6.3f}  {len(pred_text):>3}{len_flag}  {status}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Tracks evaluated       : {total}")
    print(f"Tracks skipped         : {skipped}")
    if total:
        acc = correct / total
        print(f"Track-level Exact Match: {acc:.4f}  ({correct}/{total})")
        print(f"Length-enforced picks  : {len_enforced}  "
              f"({100*len_enforced/total:.1f}% of tracks)")
        print(f"Still wrong length     : {wrong_len_pred}  "
              f"({100*wrong_len_pred/total:.1f}% of tracks)")
    print("=" * 65)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track-level LPR inference")
    parser.add_argument(
        "--model_path", type=str,
        default="/content/drive/MyDrive/LPR_Project/crnn_best.pth",
    )
    parser.add_argument(
        "--data_path", type=str,
        default="/content/raw/wYe7pBJ7-train/train",
    )
    parser.add_argument("--num_tracks", type=int, default=None)
    parser.add_argument("--beam_width",  type=int, default=10,
                        help="CTC beam search width (default 10)")
    args = parser.parse_args()

    run_inference(args.model_path, args.data_path,
                  args.num_tracks, args.beam_width)
