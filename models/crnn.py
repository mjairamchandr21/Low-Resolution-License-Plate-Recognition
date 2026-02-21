import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32x128 -> 16x64

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 16x64 -> 8x32

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d((2, 1)),  # 8x32 -> 4x32

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d((2, 1))   # 4x32 -> 2x32
        )

        self.rnn = nn.LSTM(
            input_size=512 * 2,   # channels * height
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, 1, 32, 128)

        features = self.cnn(x)
        # shape: (B, 512, 2, 32)

        b, c, h, w = features.size()

        # Collapse height dimension
        features = features.permute(0, 3, 1, 2)  # (B, W, C, H)
        features = features.reshape(b, w, c * h)  # (B, W, 512*2)

        rnn_out, _ = self.rnn(features)  # (B, W, 512)

        logits = self.fc(rnn_out)  # (B, W, num_classes)

        return logits