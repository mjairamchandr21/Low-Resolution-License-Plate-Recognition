Multi-Frame LPR: Super-Resolution & OCR Pipeline
This repository contains an end-to-end Deep Learning pipeline designed for License Plate Recognition (LPR) in challenging conditions. The project utilizes a Multi-Frame Super-Resolution (SR) model to enhance low-resolution video tracks and a Transformer-based OCR model for character recognition.

Features
Multi-Frame Fusion: Aggregates information from 5 consecutive frames to reconstruct a high-quality license plate image.

Joint Training: Optimized using a combined loss function (MSE for SR + CTC for OCR).

Transformer-Based OCR: Leverages attention mechanisms to handle sequence modeling of characters.

Competition Ready: Includes a script to generate submissions in the track_id,plate_text;confidence format.

Model Architecture
1. Multi-Frame Super-Resolution (SR)
The SR model takes a tensor of shape [Batch, 5, 3, 64, 128] and collapses the temporal dimension into the channel dimension. It uses a series of convolutions and a PixelShuffle layer to upscale the image to [256, 128].

2. Transformer-OCR
The OCR model consists of:

CNN Backbone: Extracts spatial features from the SR output.

Adaptive Height Pooling: Collapses the height dimension to create a 1D sequence.

Transformer Encoder: Processes the sequence to understand character dependencies.

CTC Decoding: Translates the sequence into a final string.

Metrics
MSE (Mean Squared Error): Measures the reconstruction quality of the SR model.
CTC Loss: Optimizes character alignment and recognition.
Accuracy: Percentage of tracks where the entire plate string is predicted exactly correct.

Submission Format
The output submission.txt follows the requirement:
track_id,plate_text;confidence
Example:
Plaintext
track_00001,ABC1234;0.9876
track_00002,XYZ9012;0.8543

Contributing
Feel free to open issues or submit pull requests to improve the SR quality or OCR accuracy!
