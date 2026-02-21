import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.dataset import LPRDataset, NUM_CLASSES
from models.crnn import CRNN
import torch.optim as optim
from tqdm import tqdm
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    images, labels, lengths = zip(*batch)

    images = torch.stack(images)

    targets = torch.cat(labels)
    target_lengths = torch.tensor(lengths, dtype=torch.long)

    return images, targets, target_lengths


def greedy_decoder(output):
    # output: (T, B, C)
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


def train():
    DATA_PATH = "/mnt/c/Users/Vikram Kumar/lpr_project/data/raw/wYe7pBJ7-train/train"

    dataset = LPRDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        collate_fn=collate_fn, num_workers=4)

    model = CRNN(NUM_CLASSES).to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0

        loop = tqdm(loader)

        for images, targets, target_lengths in loop:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            logits = model(images)
            logits = logits.permute(1, 0, 2)  # (T, B, C)

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

        print(f"Epoch {epoch} Loss: {total_loss / len(loader)}")

        torch.save(model.state_dict(), f"crnn_epoch_{epoch}.pth")


if __name__ == "__main__":
    train()