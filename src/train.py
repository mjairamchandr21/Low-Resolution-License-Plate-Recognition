import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import LPRDataset
from src.model import SRCNN
import os

def train():
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Data Loading
    # Point to the root directory since paths in CSV are 'data/raw/...'
    dataset = LPRDataset(csv_file='data/training_manifest.csv', root_dir='.')
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. Model, Loss, and Optimizer
    model = SRCNN().to(device)
    criterion = nn.MSELoss() # Measures how blurry the output is compared to HR
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Adjusts weights

    # 4. The Training Loop
    epochs = 10 # Start small for testing
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            # We only train on batches that have HR images (Scenario-A)
            if not isinstance(batch[1], torch.Tensor):
                continue
            
            lr_imgs, hr_imgs = batch[0].to(device), batch[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: Blurry -> Model -> Predicted Clear
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)

            # Backward pass: Calculate error and update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Complete. Average Loss: {running_loss/len(train_loader):.4f}")

    # 5. Save the Brain
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/srcnn_lpr.pth')
    print("Training finished! Model saved to models/srcnn_lpr.pth")

if __name__ == "__main__":
    train()