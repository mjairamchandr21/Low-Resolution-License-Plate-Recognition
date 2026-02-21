import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from training.dataset import LPRDataset, NUM_CLASSES, idx_to_char
from models.crnn import CRNN
import os
from collections import defaultdict, Counter


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    targets = torch.cat(labels)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    return images, targets, target_lengths


def greedy_decode_with_confidence(logits):
    """
    logits: (T, B, C)
    Returns:
        decoded_strings
        confidences
    """
    probs = F.softmax(logits, dim=2)
    max_probs, preds = torch.max(probs, dim=2)

    decoded = []
    confidences = []

    for b in range(preds.size(1)):
        prev = -1
        seq = []
        conf = []

        for t in range(preds.size(0)):
            val = preds[t, b].item()
            if val != prev and val != 0:
                seq.append(val)
                conf.append(max_probs[t, b].item())
            prev = val

        text = "".join([idx_to_char[i] for i in seq if i in idx_to_char])
        confidence = sum(conf) / len(conf) if len(conf) > 0 else 0.0

        decoded.append(text)
        confidences.append(confidence)

    return decoded, confidences


def evaluate():
    DATA_PATH = "/content/raw/wYe7pBJ7-train/train"

    dataset = LPRDataset(DATA_PATH)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_ds = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_ds, batch_size=64,
                            shuffle=False, collate_fn=collate_fn)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(
        torch.load("/content/drive/MyDrive/LPR_Project/crnn_best.pth")
    )
    model.eval()

    track_predictions = defaultdict(list)
    track_gt = {}

    with torch.no_grad():
        idx_global = 0

        for images, targets, target_lengths in val_loader:
            images = images.to(DEVICE)

            logits = model(images)
            logits = logits.permute(1, 0, 2)

            decoded, confidences = greedy_decode_with_confidence(logits)

            idx_local = 0
            for i in range(len(decoded)):
                gt = targets[idx_local:idx_local+target_lengths[i]].tolist()
                gt_text = "".join([idx_to_char[j] for j in gt if j in idx_to_char])

                # Recover track_id from dataset
                img_path, _ = val_ds.dataset.samples[val_ds.indices[idx_global]]
                track_id = os.path.basename(os.path.dirname(img_path))

                track_predictions[track_id].append(
                    (decoded[i], confidences[i])
                )
                track_gt[track_id] = gt_text

                idx_local += target_lengths[i]
                idx_global += 1

    # Aggregate per track
    correct = 0
    total = 0

    for track_id, preds in track_predictions.items():
        texts = [p[0] for p in preds]
        most_common = Counter(texts).most_common(1)[0][0]

        if most_common == track_gt[track_id]:
            correct += 1

        total += 1

    print(f"\nTrack-level Exact Match Accuracy: {correct/total:.4f}")


if __name__ == "__main__":
    evaluate()