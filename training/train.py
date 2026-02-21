import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from training.dataset import LPRDataset, NUM_CLASSES, idx_to_char
from models.crnn import CRNN
import torch.optim as optim
from tqdm import tqdm


DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH  = "/content/raw/wYe7pBJ7-train/train"
SAVE_DIR   = "/content/drive/MyDrive/LPR_Project"


# ── Helpers ──────────────────────────────────────────────────────────────────

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images         = torch.stack(images)
    targets        = torch.cat(labels)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    return images, targets, target_lengths


def greedy_decoder(output):
    """output: (T, B, C)"""
    output = torch.argmax(output, dim=2)
    decoded = []
    for b in range(output.size(1)):
        prev, seq = -1, []
        for t in range(output.size(0)):
            val = output[t, b].item()
            if val != prev and val != 0:
                seq.append(val)
            prev = val
        decoded.append(seq)
    return decoded


def decode_to_string(seq):
    return "".join(idx_to_char[i] for i in seq if i in idx_to_char)


def validate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(DEVICE)
            logits = model(images).permute(1, 0, 2)
            decoded = greedy_decoder(logits)
            idx = 0
            for i, seq in enumerate(decoded):
                gt   = decode_to_string(targets[idx:idx+target_lengths[i]].tolist())
                pred = decode_to_string(seq)
                if gt == pred:
                    correct += 1
                total += 1
                idx += target_lengths[i]
    return correct / total


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(model, train_loader, val_loader,
                 epochs, lr, tag="main", start_epoch=0):
    """
    Generic training loop reused for main training and fine-tune phase.
    """
    criterion  = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val   = 0.0

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch}")

        for images, targets, target_lengths in loop:
            images         = images.to(DEVICE)
            targets        = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            logits = model(images).permute(1, 0, 2)

            input_lengths = torch.full(
                (logits.size(1),), logits.size(0), dtype=torch.long
            ).to(DEVICE)

            loss = criterion(logits, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_acc = validate(model, val_loader)
        scheduler.step(val_acc)

        avg_loss = total_loss / len(train_loader)
        lr_now   = optimizer.param_groups[0]["lr"]
        print(f"\n[{tag}] Epoch {epoch} | Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | LR: {lr_now:.6f}")

        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            path = f"{SAVE_DIR}/crnn_best_{tag}.pth"
            torch.save(model.state_dict(), path)
            print(f"  ✅ Best {tag} model saved → {path}  (val={val_acc:.4f})")

        # Always save latest
        torch.save(model.state_dict(),
                   f"{SAVE_DIR}/crnn_latest_{tag}.pth")

    return best_val


# ── Main ─────────────────────────────────────────────────────────────────────

def train():
    # ── Phase 1: Full dataset training ──────────────────────────────────────
    print("=" * 60)
    print("PHASE 1: Full dataset training (Scenario-A + B, 30 epochs)")
    print("=" * 60)

    full_dataset = LPRDataset(DATA_PATH, augment=True)
    train_size   = int(0.9 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Val set should NOT be augmented — use a clean copy of the dataset
    val_dataset = LPRDataset(DATA_PATH, augment=False)
    val_ds_clean = torch.utils.data.Subset(val_dataset, val_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds_clean, batch_size=64, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    model = CRNN(NUM_CLASSES, dropout=0.3).to(DEVICE)

    run_training(model, train_loader, val_loader,
                 epochs=30, lr=1e-3, tag="main")

    # ── Phase 2: Scenario-B fine-tuning ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Scenario-B fine-tune (5 epochs, lower LR)")
    print("=" * 60)

    # Auto-detect the correct Scenario-B folder name
    available_scenarios = [
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    ]
    print(f"Available scenarios: {available_scenarios}")

    scen_b_name = None
    for s in available_scenarios:
        if "b" in s.lower() or "B" in s:
            scen_b_name = s
            break

    if scen_b_name is None:
        print("⚠️  No Scenario-B folder found — skipping fine-tune phase.")
        print(f"   Scenarios found: {available_scenarios}")
    else:
        print(f"Using scenario: '{scen_b_name}'")

        scen_b_train = LPRDataset(DATA_PATH, augment=True,  scenario_filter=scen_b_name)
        scen_b_val   = LPRDataset(DATA_PATH, augment=False, scenario_filter=scen_b_name)

        print(f"Scenario-B samples: {len(scen_b_train)}")

        if len(scen_b_train) == 0:
            print("⚠️  Scenario-B dataset is empty — check DATA_PATH and folder names.")
        else:
            sb_train_size = int(0.9 * len(scen_b_train))
            sb_val_size   = len(scen_b_train) - sb_train_size
            sb_train_ds, sb_val_idx = random_split(
                scen_b_train, [sb_train_size, sb_val_size],
                generator=torch.Generator().manual_seed(42)
            )
            sb_val_clean = torch.utils.data.Subset(scen_b_val, sb_val_idx.indices)

            sb_train_loader = DataLoader(sb_train_ds, batch_size=64, shuffle=True,
                                         collate_fn=collate_fn, num_workers=2)
            sb_val_loader   = DataLoader(sb_val_clean, batch_size=64, shuffle=False,
                                         collate_fn=collate_fn, num_workers=2)

            # Load best phase-1 weights before fine-tuning
            model.load_state_dict(
                torch.load(f"{SAVE_DIR}/crnn_best_main.pth", map_location=DEVICE)
            )

            run_training(model, sb_train_loader, sb_val_loader,
                         epochs=5, lr=1e-4, tag="scenB")

    print("\n🏁 Training complete.")
    print(f"   Best main model  → {SAVE_DIR}/crnn_best_main.pth")
    print(f"   Best ScenB model → {SAVE_DIR}/crnn_best_scenB.pth")
    print("   Use crnn_best_scenB.pth for submission (or crnn_best_main.pth if scenB skipped).")


if __name__ == "__main__":
    train()