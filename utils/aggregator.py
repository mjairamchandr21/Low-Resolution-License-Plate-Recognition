import math
import torch
import torch.nn.functional as F
import numpy as np


# Brazilian plates are always 3 letters + 4 digits = 7 characters
PLATE_LENGTH = 7


# ── Numerical helpers ────────────────────────────────────────────────────────

def _log_sum_exp(a, b):
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


# ── CTC Beam Search ──────────────────────────────────────────────────────────

def ctc_beam_search(log_probs, beam_width=10):
    """
    Pure-PyTorch CTC beam search — no external deps required.

    Args:
        log_probs : Tensor (T, num_classes) — log-softmax probabilities
                    index 0 = CTC blank
        beam_width: number of beams to keep at each timestep

    Returns:
        List of (sequence_tuple, score_float) sorted by descending score.
        sequence_tuple contains raw class indices (no blanks, collapsed).
    """
    NEG_INF = float('-inf')
    T = log_probs.shape[0]

    # beams: dict { sequence_tuple -> (log_prob_ending_in_blank,
    #                                   log_prob_ending_in_non_blank) }
    beams = {(): (0.0, NEG_INF)}

    for t in range(T):
        new_beams = {}

        for seq, (log_pb, log_pnb) in beams.items():
            log_p_total = _log_sum_exp(log_pb, log_pnb)

            # ── Extend with blank ─────────────────────────────────────────
            log_p_blank = log_probs[t, 0].item()
            new_log_pb  = log_p_total + log_p_blank
            if seq in new_beams:
                ob, onb = new_beams[seq]
                new_beams[seq] = (_log_sum_exp(ob, new_log_pb), onb)
            else:
                new_beams[seq] = (new_log_pb, NEG_INF)

            # ── Extend with each non-blank character ──────────────────────
            for c in range(1, log_probs.shape[1]):
                log_p_c  = log_probs[t, c].item()
                new_seq  = seq + (c,)

                # If last char == c, only extend from blank (CTC rule)
                if seq and seq[-1] == c:
                    new_log_pnb = log_pb + log_p_c
                else:
                    new_log_pnb = log_p_total + log_p_c

                if new_seq in new_beams:
                    ob, onb = new_beams[new_seq]
                    new_beams[new_seq] = (ob, _log_sum_exp(onb, new_log_pnb))
                else:
                    new_beams[new_seq] = (NEG_INF, new_log_pnb)

        # Prune to top beam_width
        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda x: _log_sum_exp(x[1][0], x[1][1]),
                reverse=True,
            )[:beam_width]
        )

    # Gather results
    results = [
        (seq, _log_sum_exp(log_pb, log_pnb))
        for seq, (log_pb, log_pnb) in beams.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ── Greedy fallback ──────────────────────────────────────────────────────────

def greedy_decode_from_probs(avg_probs, idx_to_char):
    """
    Greedy decode from averaged frame probabilities.

    Args:
        avg_probs  : Tensor (T, num_classes) — softmax-averaged
        idx_to_char: dict {index -> char}

    Returns:
        text (str), confidence (float)
    """
    best_indices = torch.argmax(avg_probs, dim=1)
    best_probs   = avg_probs[torch.arange(len(best_indices)), best_indices]

    prev = -1
    chars, char_probs = [], []
    for t in range(len(best_indices)):
        idx = best_indices[t].item()
        if idx != prev and idx != 0:
            chars.append(idx)
            char_probs.append(best_probs[t].item())
        prev = idx

    text       = "".join(idx_to_char.get(i, "") for i in chars)
    confidence = float(np.mean(char_probs)) if char_probs else 0.0
    return text, confidence


# ── Length-enforced pick ─────────────────────────────────────────────────────

def _pick_with_length(beam_results, idx_to_char, expected_length=PLATE_LENGTH):
    """
    From beam search results, prefer a beam whose decoded length equals
    `expected_length`.  Falls back to the top beam if none match.

    Returns:
        text (str), confidence (float), length_enforced (bool)
    """
    def _decode(seq):
        return "".join(idx_to_char.get(i, "") for i in seq)

    # Try to find the highest-scoring beam with the right length
    for seq, score in beam_results:
        text = _decode(seq)
        if len(text) == expected_length:
            # Normalise log-score to a 0–1 confidence proxy
            confidence = min(1.0, max(0.0, math.exp(score / max(len(seq), 1))))
            return text, confidence, True

    # No beam matched — use top beam anyway
    if beam_results:
        seq, score = beam_results[0]
        text = _decode(seq)
        confidence = min(1.0, max(0.0, math.exp(score / max(len(seq), 1))))
        return text, confidence, False

    return "", 0.0, False


# ── Public API ───────────────────────────────────────────────────────────────

def aggregate_track(logits_list, idx_to_char,
                    beam_width=10, expected_length=PLATE_LENGTH):
    """
    Aggregate predictions from multiple frames of the same track.

    Pipeline:
        1. Softmax each frame's logits       → per-frame probability distributions
        2. Average probabilities across frames → smoother signal
        3. Log-softmax the averaged probs     → log-probs for beam search
        4. CTC beam search                    → ranked candidate sequences
        5. Length enforcement                 → prefer `expected_length`-char pred

    Args:
        logits_list    : list of Tensors, each (T, num_classes)
        idx_to_char    : dict {index -> char}
        beam_width     : number of beams in CTC search (default 10)
        expected_length: expected plate character count (default 7)

    Returns:
        text            (str)   — predicted plate string
        confidence      (float) — normalised confidence score
        length_enforced (bool)  — True if the result matched expected_length
    """
    if not logits_list:
        return "", 0.0, False

    # 1 & 2 — Softmax + average across frames
    probs_list = [F.softmax(logits.float(), dim=-1) for logits in logits_list]
    avg_probs  = torch.stack(probs_list, dim=0).mean(dim=0)   # (T, C)

    # 3 — Log for beam search (add tiny epsilon to avoid log(0))
    log_probs = torch.log(avg_probs + 1e-8)                   # (T, C)

    # 4 — Beam search
    beam_results = ctc_beam_search(log_probs, beam_width=beam_width)

    # 5 — Length enforcement
    return _pick_with_length(beam_results, idx_to_char, expected_length)
