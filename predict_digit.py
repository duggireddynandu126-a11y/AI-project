"""Predict a digit from an 8x8 grayscale image array saved as a .npy file.

Example:
    python predict_digit.py sample_digit.npy

The .npy file must contain an 8x8 numeric array with pixel values similar to the
scikit-learn digits dataset (typically 0..16). This script does not process camera
photos or arbitrary image files.
"""
import sys
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_digit_classifier.joblib"

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict_digit.py path_to_8x8_array.npy")
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run `python main.py` first to train and save it.")
    image = np.load(sys.argv[1])
    if image.shape != (8, 8):
        raise SystemExit(f"Expected an 8x8 array; received shape {image.shape}.")
    model = joblib.load(MODEL_PATH)
    features = image.astype(float).reshape(1, 64)
    prediction = int(model.predict(features)[0])
    print(f"Predicted digit: {prediction}")

if __name__ == "__main__":
    main()
