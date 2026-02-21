import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from training.dataset import LPRDataset, NUM_CLASSES, idx_to_char
from models.crnn import CRNN
import torch.optim as optim
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    targets = torch.cat(labels)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    return images, targets, target_lengths


def greedy_decoder(output):
    output = torch.argmax(output, dim=2)

    decoded = []
    for b in range(output.size(1)):
        prev = -1
        seq = []
        for t in range(output.size(0)):
            val = output[t, b].item()
            if val != prev and val != 0:
                seq.append(val)
            prev = val
        decoded.append(seq)
    return decoded


def decode_to_string(seq):
    return "".join([idx_to_char[i] for i in seq if i in idx_to_char])


def validate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(DEVICE)
            logits = model(images)
            logits = logits.permute(1, 0, 2)

            decoded = greedy_decoder(logits)

            idx = 0
            for i in range(len(decoded)):
                gt = targets[idx:idx+target_lengths[i]].tolist()
                gt_text = decode_to_string(gt)
                pred_text = decode_to_string(decoded[i])

                if gt_text == pred_text:
                    correct += 1

                total += 1
                idx += target_lengths[i]

    return correct / total


def train():
    DATA_PATH = "/content/raw/wYe7pBJ7-train/train"

    dataset = LPRDataset(DATA_PATH)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)

    val_loader = DataLoader(val_ds, batch_size=64,
                            shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader)

        for images, targets, target_lengths in loop:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            logits = model(images)
            logits = logits.permute(1, 0, 2)

            input_lengths = torch.full(
                size=(logits.size(1),),
                fill_value=logits.size(0),
                dtype=torch.long
            ).to(DEVICE)

            loss = criterion(logits, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        val_acc = validate(model, val_loader)

        print(f"\nEpoch {epoch} Loss: {total_loss/len(train_loader)}")
        print(f"Validation Exact Match Accuracy: {val_acc:.4f}\n")

        torch.save(model.state_dict(),
                   f"/content/drive/MyDrive/LPR_Project/crnn_epoch_{epoch}.pth")


if __name__ == "__main__":
    train()