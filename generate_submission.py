"""
Generate competition submission CSV from the public test set.

The public test set has a flat structure (no scenario/layout hierarchy):
    Pa7a3Hin-test-public/
        track_10005/
            lr-001.png
            lr-002.png
            ...

Usage (Colab):
    # 1. Unzip the test data
    !unzip /content/drive/MyDrive/LPR_Project/Pa7a3Hin-test-public.zip \
           -d /content/test_data

    # 2. Run submission generator
    !python generate_submission.py \
        --model_path /content/drive/MyDrive/LPR_Project/crnn_best.pth \
        --test_path  /content/test_data/Pa7a3Hin-test-public \
        --output     /content/drive/MyDrive/LPR_Project/submission.csv \
        --beam_width 10
"""

import argparse
import os
import zipfile
import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.crnn import CRNN
from training.dataset import NUM_CLASSES, idx_to_char
from utils.aggregator import aggregate_track, PLATE_LENGTH


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_image(img_path: str) -> torch.Tensor:
    """Aspect-ratio resize to H=32, pad W to 128.
    Applies JPEG denoising for .jpg inputs to reduce compression artefacts.
    Returns (1,1,32,128).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    # Denoise JPEG artefacts before resizing (helps model confidence)
    if img_path.lower().endswith((".jpg", ".jpeg")):
        img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
        # Mild sharpening to recover edges lost to compression
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img    = cv2.filter2D(img, -1, kernel)
        img    = np.clip(img, 0, 255).astype(np.uint8)

    h, w   = img.shape
    new_h  = 32
    new_w  = int(w * (new_h / h))
    img    = cv2.resize(img, (new_w, new_h))

    if new_w < 128:
        img = np.pad(img, ((0, 0), (0, 128 - new_w)),
                     mode="constant", constant_values=0)
    else:
        img = cv2.resize(img, (128, 32))

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)                           # (1,H,W)
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,1,H,W)



# ── Model forward ────────────────────────────────────────────────────────────

@torch.no_grad()
def get_logits(model: CRNN, img_tensor: torch.Tensor) -> torch.Tensor:
    return model(img_tensor.to(DEVICE)).squeeze(0)   # (T, C)


# ── Test track discovery ─────────────────────────────────────────────────────

def get_test_tracks(test_root: str):
    """
    Discover all track folders in the flat test structure.
    Returns sorted list of (track_id, [image_paths]).
    Accepts any .png or .jpg image inside each track folder.
    """
    IMAGE_EXTS = (".png", ".jpg", ".jpeg")
    tracks = []
    entries = sorted(os.listdir(test_root))

    # Print first-track contents for diagnostics
    for entry in entries:
        track_path = os.path.join(test_root, entry)
        if os.path.isdir(track_path):
            print(f"[DEBUG] First track '{entry}' files: {os.listdir(track_path)[:8]}")
            break

    for entry in entries:
        track_path = os.path.join(test_root, entry)
        if not os.path.isdir(track_path):
            continue

        images = sorted([
            os.path.join(track_path, f)
            for f in os.listdir(track_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])

        if images:
            tracks.append((entry, images))

    return tracks


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_submission(model_path: str, test_path: str,
                        output_path: str, beam_width: int = 10):

    print(f"Loading model : {model_path}")
    model = CRNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    tracks = get_test_tracks(test_path)
    print(f"Found {len(tracks)} test tracks  |  device={DEVICE}  |  "
          f"beam_width={beam_width}\n")

    if len(tracks) == 0:
        print(f"ERROR: No tracks found at: {test_path}")
        print(f"Directory contents: {os.listdir(test_path) if os.path.exists(test_path) else 'PATH DOES NOT EXIST'}")
        return

    rows          = []
    wrong_len     = 0
    len_enforced  = 0

    for track_id, images in tqdm(tracks, desc="Predicting"):
        logits_list = []
        for img_path in images:
            try:
                tensor = preprocess_image(img_path)
                logits_list.append(get_logits(model, tensor))
            except Exception:
                pass

        if not logits_list:
            pred_text, confidence = "UNKNOWN", 0.0
            enforced = False
        else:
            pred_text, confidence, enforced = aggregate_track(
                logits_list, idx_to_char,
                beam_width=beam_width,
                expected_length=PLATE_LENGTH,
            )

        if enforced:
            len_enforced += 1
        if len(pred_text) != PLATE_LENGTH:
            wrong_len += 1

        rows.append({
            "track_id":   track_id,
            "plate_text": pred_text,
            "confidence": f"{confidence:.4f}",
        })

    # Write predictions.txt in competition format:
    # track_id,plate_text;confidence   (no header)
    out_dir     = os.path.dirname(os.path.abspath(output_path))
    txt_path    = os.path.join(out_dir, "predictions.txt")
    zip_path    = os.path.join(out_dir, "submission.zip")

    os.makedirs(out_dir, exist_ok=True)

    with open(txt_path, "w") as f:
        for r in rows:
            f.write(f"{r['track_id']},{r['plate_text']};{r['confidence']}\n")

    # Zip predictions.txt directly (not inside a folder)
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname="predictions.txt")

    print(f"\n{'='*55}")
    print(f"predictions.txt : {txt_path}")
    print(f"submission.zip  : {zip_path}")
    print(f"Total tracks    : {len(rows)}")
    if len(rows) > 0:
        print(f"Length-enforced : {len_enforced} ({100*len_enforced/len(rows):.1f}%)")
        print(f"Wrong length    : {wrong_len}  ({100*wrong_len/len(rows):.1f}%)")
    print(f"{'='*55}")
    print("\nFirst 5 predictions:")
    for r in rows[:5]:
        print(f"  {r['track_id']},{r['plate_text']};{r['confidence']}")



# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPR competition submission generator")
    parser.add_argument(
        "--model_path", type=str,
        default="/content/drive/MyDrive/LPR_Project/crnn_best.pth",
    )
    parser.add_argument(
        "--test_path", type=str,
        default="/content/test/Pa7a3Hin-test-public",
        help="Path to extracted test folder containing track_XXXXX subdirs",
    )
    parser.add_argument(
        "--output", type=str,
        default="/content/drive/MyDrive/LPR_Project/submission.csv",
        help="Path to write submission CSV",
    )
    parser.add_argument(
        "--beam_width", type=int, default=10,
    )
    args = parser.parse_args()

    generate_submission(args.model_path, args.test_path,
                        args.output, args.beam_width)
