# License Plate Recognition (LPR) for Low-Quality Surveillance

Automated recognition of license plates in surveillance contexts is challenging due to distortion, motion blur, sensor noise, and symbols blending into complex backgrounds. This project implements a robust, two-phase Deep Learning approach designed to maintain high accuracy even under these adversarial conditions.

## 🚀 Key Features

- **Deep CRNN Architecture**: Combines a Convolutional Neural Network (CNN) for spatial feature extraction with a Bidirectional LSTM (Bi-LSTM) for sequential modeling.
- **CTC Alignment**: Utilizes Connectionist Temporal Classification (CTC) loss to handle character alignment without requiring per-character bounding box annotations.
- **Two-Phase Fine-Tuning**: A domain-adaptation strategy that trains on a broad dataset (Scenario-A) before fine-tuning on highly specific, low-quality surveillance data (Scenario-B).
- **Robust Augmentation Pipeline**: Simulates real-world surveillance degradations including JPEG artifacts, focus blur, sensor noise, and varying lighting conditions.
- **Advanced Decoding**: Implements Beam Search decoding with track-level sequence aggregation to maximize prediction confidence across multiple video frames.

---

## 🏗️ Core Methodology

### Model Architecture (`CRNN`)
The core model is a Convolutional Recurrent Neural Network (CRNN):
1.  **CNN Backbone**: 4-layer convolutional stack with Batch Normalization and Max-Pooling to extract high-level visual features.
2.  **Recurrent Layers**: A 2-layer Bidirectional LSTM that models the horizontal dependencies between character segments.
3.  **Head**: A fully connected classifier mapping recurrence outputs to character probabilities (36 classes: 0-9, A-Z + 1 blank).

### Training Strategy
- **Phase 1 (Main)**: Training on the full dataset with aggressive augmentation to build a generalized feature extractor.
- **Phase 2 (Fine-tune)**: Targeted fine-tuning on "Scenario-B" (public test domain) with a lower learning rate (1e-4) to close the domain gap.
- **Augmentations**: To bridge the gap between clean training data and "low-res" (LR) test data, we simulate JPEG compression (40-85 quality), Gaussian noise, and blur directly in the data loader.

---

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-repo/lpr-project.git
    cd lpr-project
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📖 Usage Guide

### 1. Main Training
Train the model on the full dataset (Scenario-A + B):
```bash
python -m training.train
```

### 2. Scenario-B Fine-Tuning
Refine the model for the target surveillance domain:
```bash
python training/finetune_scenb.py
```

### 3. Generate Submission
Process a test dataset and generate the `predictions.txt` and `submission.zip`:
```bash
python generate_submission.py --model_path models/crnn_best_scenB.pth --test_path path/to/public_test --output submissions/submission.csv
```

---

## 📂 Project Structure

- `models/`: Model definitions (CRNN architecture and EasyOCR wrapper).
- `training/`: Training scripts, dataset loaders, and fine-tuning logic.
- `src/`: Core preprocessing and dataset utilities.
- `utils/`: Aggregation and post-processing logic (Beam Search).
- `generate_submission.py`: Main script for competition inference.
- `evaluate.py`: Local evaluation metrics.

## 🧠 Advanced Inference Techniques

- **Beam Search**: Instead of simple greedy decoding, we explore the top $N$ most likely character sequences.
- **Track Aggregation**: For surveillance video tracks, we average the logits across multiple frames before decoding, significantly reducing the impact of single-frame "glitches."
- **Denoising**: The inference pipeline includes Non-Local Means Denoising to filter out heavy compression artifacts from `.jpg` inputs.
