# 🚀 LPR Project - Quick Start Guide

## Platform Choice: **Hybrid Approach Recommended**

### Best Setup:

```
VSCode (Local Development + Testing) + Google Colab (GPU Training)
```

---

## ⚡ Quick Start (Choose One)

### Option 1: Google Colab (⭐ Recommended for Quick Start)

**Pros:**

- ✅ Free GPU access (T4/V100)
- ✅ Pre-installed libraries (mostly)
- ✅ Direct Google Drive integration
- ✅ No local setup required
- ✅ Perfect for training

**Cons:**

- ❌ Limited local development experience
- ❌ Runtime resets after 12 hours
- ❌ Slower for iterative development

**Steps:**

1. Open Google Colab: https://colab.research.google.com/
2. Upload notebook: `notebooks/COLAB_Training.ipynb`
3. Run cells sequentially
4. Download `submission.zip` when done

**Estimated Time:**

- Setup: 5 minutes
- Training (25 epochs): 2-3 hours
- Inference: 30 minutes
- **Total: ~3 hours**

---

### Option 2: VSCode + Local Development

**Pros:**

- ✅ Full IDE features
- ✅ Better code organization
- ✅ Easier debugging
- ✅ No runtime limits
- ✅ Better for learning

**Cons:**

- ❌ Requires GPU locally (or use WSL + GPU)
- ❌ Longer setup time
- ❌ Need to manage dependencies

**Steps:**

#### 2.1 Clone & Setup Locally

```bash
# Navigate to project
cd c:\Users\Vikram Kumar\lpr_project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (CMD):
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'GPU available: {torch.cuda.is_available()}')"
```

#### 2.2 Local Training (to GPU on Colab later)

**Workflow:**

```
1. Develop locally (VSCode)
2. Test with small dataset
3. Push to Colab for full training
4. Download results
```

#### 2.3 Training on Colab

```python
# At the top of COLAB_Training.ipynb, configure:
DATA_PATH = '/content/drive/MyDrive/LPR_Project/data/raw/wYe7pBJ7-train/train'
SAVE_DIR = '/content/drive/MyDrive/LPR_Project'
```

---

## 📊 Project Structure

```
lpr_project/
├── notebooks/
│   └── COLAB_Training.ipynb          # ⭐ START HERE for Colab
├── models/
│   └── crnn.py                       # CRNN architecture
├── training/
│   ├── train.py                      # Training script
│   ├── dataset.py                    # Dataset class
│   └── test_dataset.py               # Dataset tests
├── inference/
│   └── infer_track.py                # Inference pipeline
├── utils/
│   ├── data_loader.py                # Data utilities
│   └── aggregator.py                 # Prediction aggregation
├── generate_submission.py            # Create submission.zip
├── evaluate.py                       # Evaluate model
├── infer.py                          # Run inference
├── requirements.txt                  # Dependencies
├── README.md                         # Project documentation
└── PROJECT_PLAN.md                   # Detailed roadmap
```

---

## 🎯 Current Status

✅ **Completed:**

- CRNN model architecture
- Data loading pipeline
- Training framework
- Inference system
- Submission generator

⚠️ **Configuration Needed:**

- Update paths in code (adjust to your setup)
- Verify Google Drive mount
- Check GPU availability

---

## 💡 Key Concepts to Understand

### 1. **CRNN Architecture**

```
Input Image (32×128)
    ↓
CNN Feature Extractor (4 layers)
- Conv2d + ReLU + MaxPool
- Outputs: 512 channels, 2×32 feature map
    ↓
Bidirectional LSTM (2 layers)
- Processes 32 time steps
- Hidden size: 256 (per direction)
    ↓
Fully Connected Classifier
- Outputs probabilities for 36 classes (0-9, A-Z)
    ↓
CTC Decoder
- Handles variable-length sequences
- Greedy decoding for inference
```

### 2. **Training Strategy**

```
Phase 1: Train on combined Scenario-A + B
- Aggressive augmentation
- Full learning rate
- ~25 epochs

Phase 2: Fine-tune on Scenario-B
- Lighter augmentation
- Lower learning rate (1e-4)
- Domain adaptation
```

### 3. **Inference Pipeline**

```
Per Track (5 LR images):
1. Load all 5 images
2. Model prediction for each → 5 predictions
3. Aggregate using confidence voting
4. Output: track_id, plate_text, confidence
```

### 4. **Aggregation Methods**

```
Option 1: Majority Voting (+ Max Confidence)
- Takes most common prediction
- Uses average confidence

Option 2: Confidence Weighted
- Scores each prediction by confidence
- Weighted voting

Option 3: Temporal Consistency
- Uses frame sequence information
- More complex but potentially better
```

---

## 📈 Expected Performance

| Stage                     | Accuracy | Status                         |
| ------------------------- | -------- | ------------------------------ |
| Baseline (Scenario-A)     | 40-50%   | Achievable with basic training |
| Intermediate (Scenario-B) | 50-60%   | With fine-tuning               |
| Competition Target        | >60%     | With optimization + ensemble   |
| State-of-the-art          | ~70%+    | With advanced techniques       |

---

## 🔧 Troubleshooting

### Issue: GPU Memory Error

```python
# Reduce batch size in training
batch_size = 32  # Instead of 64
```

### Issue: Slow Training on Colab

```
- Use Colab Pro for faster GPUs
- Reduce image resolution (currently 32×128)
- Use smaller model
```

### Issue: Low Accuracy

```
- Ensure data augmentation is enabled
- Increase training epochs
- Fine-tune learning rate
- Check data loading (verify annotations)
```

### Issue: Import Errors

```bash
# Reinstall packages
pip install --upgrade torch torchvision opencv-python
```

---

## 📝 Submission Checklist

Before uploading to CodaBench:

- [ ] Model trained to >40% accuracy
- [ ] Predictions.txt generated (format: `track_id,plate_text;confidence`)
- [ ] Submission.zip created (contains only predictions.txt)
- [ ] Sample predictions verified
- [ ] All 1000+ test tracks have predictions (no missing)
- [ ] Confidence scores between 0-1
- [ ] No extra files in ZIP

---

## 🎓 Learning Resources

### Understanding the Competition

- Read: `../data/training_manifest.csv`
- Check: [ICPR 2026 Handbook](competition-link)
- Understand: Scenario A vs B differences

### Deep Learning Concepts

- **CTC Loss**: Handles alignment without character-level labels
- **LSTM**: Models sequential dependencies (character transitions)
- **CNN**: Extracts visual features from images
- **Batch Normalization**: Stabilizes training
- **Dropout**: Prevents overfitting

### Code Structure

- Dataset class handles data loading + augmentation
- Model class defines CRNN architecture
- Training loop implements CTC loss optimization
- Inference applies greedy decoding
- Aggregator combines multi-frame predictions

---

## 🚀 Next Steps After First Submission

1. **Analyze Results**
   - Check public leaderboard performance
   - Identify weak predictions
2. **Iterate & Improve**
   - Increase model capacity
   - Try ensemble methods
   - Optimize augmentation
   - Fine-tune hyperparameters
3. **Advanced Techniques**
   - Super-resolution preprocessing
   - Beam search decoding
   - Transfer learning
   - Multi-model ensemble

4. **Optimize for Final Submission**
   - Tune aggregation strategy
   - Balance ensemble weights
   - Maximize confidence calibration

---

## 💬 Quick Help

**Q: Should I use Colab or VSCode?**

- A: Colab for first submission (fastest), then VSCode for iteration

**Q: What GPU do I need?**

- A: 2GB+ (Colab T4 is sufficient)

**Q: How long does training take?**

- A: ~2-3 hours on Colab with full dataset

**Q: Can I train on full dataset locally?**

- A: Only if you have RTX/A100 GPU (RTX 2060+ works but slow)

**Q: How to improve accuracy?**

- A: More data, better augmentation, ensemble models, hyperparameter tuning

---

## 📞 Support

- Check `PROJECT_PLAN.md` for detailed roadmap
- Review code comments in notebook
- Refer to PyTorch documentation for API references
- Check stack overflow for common PyTorch issues

Good luck with the competition! 🎯
