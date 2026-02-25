# 🖥️ VSCode Local Development Setup

## System Requirements

**Minimum:**

- Windows 10/11
- 8GB RAM
- 50GB storage (for dataset)
- CPU: i5 or better

**Recommended for GPU Training:**

- NVIDIA GPU with CUDA support (RTX 2060+)
- 16GB RAM
- Latest NVIDIA drivers
- CUDA 11.8+
- cuDNN 8.0+

---

## 1️⃣ Installation Steps

### Step 1: Install Python & VSCode

```bash
# Install Python 3.9+ (if not already)
# Download from: python.org

# Verify Python installation
python --version

# Install VSCode
# Download from: code.visualstudio.com
```

### Step 2: Install Required Extensions in VSCode

1. Open VSCode
2. Go to Extensions (Ctrl+Shift+X)
3. Install these extensions:
   - **Python** (Microsoft)
   - **Pylance** (Microsoft)
   - **Jupyter** (Microsoft)
   - **PyTorch** (inferential)
   - **Thunder Indent** (optional, improves readability)

### Step 3: Clone/Navigate to Project

```bash
# Navigate to project directory
cd "c:\Users\Vikram Kumar\lpr_project"

# Open in VSCode
code .
```

### Step 4: Create Virtual Environment

```bash
# In VSCode terminal (Ctrl+`)

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate on Windows (CMD)
.venv\Scripts\activate

# Verify activation (should show .venv prefix)
```

### Step 5: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA not available, install CPU version (for testing only):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 6: Select Python Interpreter in VSCode

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose ".venv (./venv/bin/python)" or similar

---

## 2️⃣ Project Organization

### Recommended Folder Structure

```
lpr_project/
├── .venv/                     # Virtual environment (ignore)
├── data/
│   ├── raw/                   # Original dataset
│   ├── processed/             # Preprocessed data (optional)
│   └── training_manifest.csv
├── models/
│   ├── __init__.py
│   ├── crnn.py               # ⭐ Main model
│   ├── ocr_model.py
│   └── __pycache__/
├── training/
│   ├── __init__.py
│   ├── train.py              # ⭐ Training script
│   ├── dataset.py            # ⭐ Dataset class
│   ├── test_dataset.py
│   └── __pycache__/
├── inference/
│   ├── infer_track.py        # ⭐ Inference script
│   └── __pycache__/
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── aggregator.py
│   └── __pycache__/
├── notebooks/
│   ├── exploration.ipynb     # For EDA
│   └── COLAB_Training.ipynb  # ⭐ Main training notebook
├── configs/                  # Configuration files
├── outputs/                  # Results & checkpoints
│   ├── checkpoints/          # Saved models
│   └── submissions/          # Final submissions
├── evaluation/
│   └── evaluate_track.py
├── .gitignore
├── requirements.txt
├── README.md
├── QUICKSTART.md
├── PROJECT_PLAN.md
└── generate_submission.py    # ⭐ Create submission.zip
```

---

## 3️⃣ Local Development Workflow

### Workflow 1: Data Exploration (Local)

```python
# Create new file: explore_data.py

import os
import json
from pathlib import Path
from PIL import Image

# Your exploration code here
DATA_PATH = "data/raw/wYe7pBJ7-train/train"

for scenario in os.listdir(DATA_PATH):
    print(f"Scenario: {scenario}")
    # ... your analysis
```

**Run in VSCode:**

```bash
python explore_data.py
```

### Workflow 2: Test Model Locally

```python
# File: test_model_local.py

import torch
from models.crnn import CRNN

# Create model
model = CRNN(num_classes=37)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
dummy_input = torch.randn(4, 1, 32, 128)  # Batch of 4 images
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
```

**Run:**

```bash
python test_model_local.py
```

### Workflow 3: Train on Small Dataset (Local Testing)

```python
# File: train_small_local.py

import torch
from training.dataset import LPRDataset
from training.train import run_training
from models.crnn import CRNN

# Create small dataset
dataset = LPRDataset("data/raw/wYe7pBJ7-train/train", augment=True)
print(f"Dataset size: {len(dataset)}")

# Test with first 100 samples only
small_dataset = torch.utils.data.Subset(dataset, range(min(100, len(dataset))))

# Create dataloader
loader = torch.utils.data.DataLoader(small_dataset, batch_size=8, shuffle=True)

# Create model
model = CRNN(num_classes=37).to('cuda' if torch.cuda.is_available() else 'cpu')

# Test one batch
for images, labels, lengths, texts in loader:
    print(f"Batch shape: {images.shape}")
    print(f"Sample texts: {texts[:3]}")
    break
```

---

## 4️⃣ Debugging in VSCode

### Debug Setup

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Python: Module",
      "type": "python",
      "request": "launch",
      "module": "training.train",
      "console": "integratedTerminal"
    }
  ]
}
```

### Using Debugger

1. Set breakpoint (click on line number)
2. Press F5 or click "Run" button
3. Step through code (F10 = step over, F11 = step into)
4. Inspect variables in left panel

---

## 5️⃣ Running Training Locally (CPU or GPU)

### Quick Test Training (CPU, 1 epoch)

```python
# File: train_test.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.dataset import LPRDataset, NUM_CLASSES, collate_fn
from models.crnn import CRNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# Load small dataset
dataset = LPRDataset("data/raw/wYe7pBJ7-train/train", augment=True)
subset = torch.utils.data.Subset(dataset, range(min(500, len(dataset))))

loader = DataLoader(subset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Model
model = CRNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train 1 epoch
model.train()
for batch_idx, (images, targets, target_lengths) in enumerate(loader):
    images = images.to(DEVICE)
    targets = targets.to(DEVICE)
    target_lengths = target_lengths.to(DEVICE)

    logits = model(images).permute(1, 0, 2)
    input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long).to(DEVICE)

    loss = criterion(logits, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

print("✅ Test training completed!")
```

**Run:**

```bash
python train_test.py
```

---

## 6️⃣ Transferring to Colab for Full Training

### When to Move to Colab:

- Local testing complete ✓
- Model working on small data ✓
- Ready for full dataset training

### Transfer Process:

1. **Save your development code:**

   ```bash
   git add .
   git commit -m "Local testing complete"
   ```

2. **Copy notebook to Colab:**
   - Download `notebooks/COLAB_Training.ipynb`
   - Upload to Google Drive
   - Open in Colab

3. **In Colab:**

   ```python
   # Mount drive
   from google.colab import drive
   drive.mount('/content/drive')

   # Copy project to Colab
   !cp -r /content/drive/MyDrive/LPR_Project .
   ```

---

## 7️⃣ Local Utils for Data Prep

### Create Data Preprocessing Script

```python
# File: utils/preprocess_data.py

import cv2
import json
from pathlib import Path

def preprocess_images(data_path, output_path, target_size=(128, 32)):
    """Preprocess images for faster loading"""

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_file in Path(data_path).rglob('*.png'):
        # Load and resize
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, target_size)

        # Save
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), img_resized)
        count += 1

    print(f"✅ Preprocessed {count} images")

# Usage
# preprocess_images("data/raw/wYe7pBJ7-train/train", "data/processed")
```

---

## 8️⃣ GPU Setup (Optional, for Training Locally)

### Check CUDA Availability

```bash
# CMD
nvidia-smi

# Should show GPU info if drivers installed
```

### Install GPU Support

```bash
# Uninstall CPU PyTorch
pip uninstall torch torchvision -y

# Install GPU PyTorch
pip install torch torchvision torcuda --index-url https://download.pytorch.org/whl/cu118

# Or use conda (easier for GPU)
conda install pytorch::pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Monitor GPU Usage During Training

```bash
# In separate terminal
while True:
    nvidia-smi
    timeout /t 2
done
```

---

## 9️⃣ Common Issues & Solutions

| Issue                | Solution                                           |
| -------------------- | -------------------------------------------------- |
| Module not found     | Check sys.path, verify .venv is activated          |
| CUDA out of memory   | Reduce batch size: `batch_size = 32`               |
| Slow training        | Use GPU, reduce image size, reduce epochs          |
| Data loading slow    | Preprocess images, use DataLoader with num_workers |
| Model won't converge | Check learning rate, data augmentation, batch norm |

---

## 📚 Recommended Workflow

1. **Day 1-2**: Local setup & exploration
   - Install dependencies
   - Load and visualize data
   - Test model on small subset

2. **Day 3-4**: Colab training
   - Run full training on GPU
   - Monitor leaderboard
   - Download results

3. **Day 5+**: Iteration & improvement
   - Analyze errors locally
   - Try different architectures
   - Retrain on Colab
   - Submit improvements

---

## 🎯 Next: Start the Project!

```bash
# 1. Terminal: Activate environment
.\.venv\Scripts\activate

# 2. VS Code: Open integrated terminal (Ctrl+`)

# 3. Run exploration:
python notebooks/COLAB_Training.ipynb

# 4. Or start coding:
code training/train.py    # Edit training script
# Make changes and save (Ctrl+S)
# Run: python train_test.py
```

Happy coding! 🚀
