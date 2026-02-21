import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        # FIXED: Removed the colon and added .__init__()
        super(SRCNN, self).__init__()
        
        # Layer 1: Patch extraction (looks for edges/shapes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        
        # Layer 2: Non-linear mapping (identifies character features)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        
        # Layer 3: Reconstruction (sharpens the image)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x