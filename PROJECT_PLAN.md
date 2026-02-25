# LPR Project - Complete Roadmap

## Phase 1: Setup & Environment (Week 1)

- [ ] Install dependencies locally
- [ ] Setup Python virtual environment
- [ ] Configure Google Colab notebook template
- [ ] Test data loading pipeline
- [ ] Verify model architecture compiles

## Phase 2: Data Pipeline (Week 1-2)

- [ ] Implement robust data loader with all augmentations
- [ ] Handle both Scenario-A and Scenario-B data
- [ ] Create train/val splits
- [ ] Verify annotation parsing (JSON structure)
- [ ] Test image preprocessing and augmentation
- [ ] Generate sample batches for debugging

## Phase 3: Model Training (Week 2-3)

### 3.1 Phase 1 Training (Main)

- Train on combined Scenario-A + Scenario-B data
- Use aggressive augmentation to simulate low-resolution
- Monitor CTC loss and character accuracy
- Save best checkpoints

### 3.2 Phase 2 Fine-tuning

- Fine-tune on Scenario-B domain
- Lower learning rate (1e-4)
- Reduce augmentation intensity
- Monitor for overfitting

## Phase 4: Inference & Aggregation (Week 3)

- [ ] Implement track-level aggregation (majority voting + confidence)
- [ ] Test on public test set samples
- [ ] Generate predictions in correct format
- [ ] Verify submission format compliance

## Phase 5: Optimization & Submission (Week 4)

- [ ] Hyperparameter tuning
- [ ] Ensemble methods (if time permits)
- [ ] Final evaluation on public test set
- [ ] Create submission.zip
- [ ] Submit to CodaBench

---

## Key Implementation Details

### Data Format

```
data/raw/wYe7pBJ7-train/train/
├── Scenario-A/
│   ├── Brazilian/
│   │   ├── track_00001/
│   │   │   ├── lr-001.png to lr-005.png
│   │   │   ├── hr-001.png to hr-005.png
│   │   │   └── annotations.json (with corners)
│   │   └── ...
│   └── Mercosur/
└── Scenario-B/
    ├── Brazilian/
    └── Mercosur/ (NO corner annotations)
```

### Model Architecture

- **Input**: 1-channel, 32×128 grayscale images
- **CNN**: 4-layer conv stack → 512 channels, 2×32 feature map
- **RNN**: 2-layer Bi-LSTM (256 hidden units each direction)
- **Output**: CTC-logits for 36 characters (0-9, A-Z) + 1 blank

### Training Strategy

1. **Phase 1**: All data, aggressive augmentation
2. **Phase 2**: Scenario-B only, lighter augmentation
3. **CTC Loss**: Handles variable-length sequences without character alignment

### Aggregation Strategy

- For each track: 5 LR images → 5 predictions
- Methods:
  - Majority voting + max confidence
  - Weighted averaging by confidence
  - Temporal consistency (if applicable)

---

## Dependencies Required

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
opencv-python>=4.5.0
Pillow>=8.0.0
tqdm>=4.50.0
scikit-image>=0.18.0
```

---

## Current Status

✅ Model architecture (CRNN) - defined
✅ Training loop structure - in place
✅ Dataset class - started
⚠️ Data augmentation - needs completion
⚠️ CTC decoding - needs Beam Search implementation
⚠️ Inference pipeline - needs full implementation
⚠️ Aggregation logic - needs implementation
⚠️ Submission generator - needs refinement

---

## Colab Workflow

1. Mount Google Drive: `drive.mount('/content/drive')`
2. Copy data: `!cp -r /content/drive/MyDrive/lpr_data .`
3. Run training cells
4. Save checkpoints to Drive
5. Download predictions.txt

---

## Success Metrics

- **Target Accuracy (Phase 1)**: >40% character accuracy on Scenario-A
- **Target Accuracy (Phase 2)**: >50% character accuracy on Scenario-B
- **Final Goal**: Top 50% on leaderboard (>60% accuracy)
