import torch
import torch.nn.functional as F
import numpy as np


def greedy_decode_from_probs(avg_probs, idx_to_char):
    """
    Greedy decode from averaged frame probabilities.

    Args:
        avg_probs: Tensor of shape (T, num_classes) — already softmax-averaged
        idx_to_char: dict mapping index -> character

    Returns:
        text (str), confidence (float)
    """
    # Pick best class at each timestep
    best_indices = torch.argmax(avg_probs, dim=1)  # (T,)
    best_probs   = avg_probs[torch.arange(len(best_indices)), best_indices]  # (T,)

    # CTC collapse: remove blanks (idx=0) and repeated chars
    prev = -1
    chars = []
    char_probs = []

    for t in range(len(best_indices)):
        idx = best_indices[t].item()
        if idx != prev and idx != 0:
            chars.append(idx)
            char_probs.append(best_probs[t].item())
        prev = idx

    text = "".join([idx_to_char.get(i, "") for i in chars])
    confidence = float(np.mean(char_probs)) if char_probs else 0.0

    return text, confidence


def aggregate_track(logits_list, idx_to_char):
    """
    Aggregate predictions from multiple frames of the same track.

    Strategy:
        1. Apply softmax to each frame's logits  → probability distribution
        2. Average probabilities across all frames → smoother signal
        3. Greedy decode on the averaged probabilities

    Args:
        logits_list: list of Tensors, each shape (T, num_classes)
                     T = time steps (width of feature map = 32)
        idx_to_char: dict mapping index -> character

    Returns:
        text (str), confidence (float)
    """
    if not logits_list:
        return "", 0.0

    # Softmax each frame's logits
    probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]

    # Stack and average: (num_frames, T, num_classes) -> (T, num_classes)
    stacked = torch.stack(probs_list, dim=0)   # (F, T, C)
    avg_probs = stacked.mean(dim=0)            # (T, C)

    return greedy_decode_from_probs(avg_probs, idx_to_char)
