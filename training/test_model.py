import torch
from models.crnn import CRNN
from training.dataset import NUM_CLASSES

model = CRNN(NUM_CLASSES)

dummy = torch.randn(4, 1, 32, 128)

out = model(dummy)

print("Output shape:", out.shape)