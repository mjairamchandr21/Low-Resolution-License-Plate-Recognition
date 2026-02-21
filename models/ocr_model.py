import easyocr
import cv2
import numpy as np


class EasyOCRModel:
    def __init__(self, device="cpu"):
        self.reader = easyocr.Reader(
            ['en'],
            gpu=(device == "cuda")
        )

    def preprocess(self, image_path):
        img = cv2.imread(image_path)

        # Resize (important for low-res)
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Optional: slight sharpening
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)

        return sharp

    def predict(self, image_path):
        img = self.preprocess(image_path)

        results = self.reader.readtext(img)

        if len(results) == 0:
            return "", 0.0

        # Pick highest confidence prediction
        best = max(results, key=lambda x: x[2])

        text = best[1].upper().replace(" ", "")
        confidence = float(best[2])

        return text, confidence