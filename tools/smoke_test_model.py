"""Smoke test script for MobileNetV2 PlantVillage model loader."""

import sys
from PIL import Image
import numpy as np

# Add project root to path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.model_loader import load_model_and_processor, predict_image
from app.config import MODEL_ID


def main():
    print(f"Testing model loading for: {MODEL_ID}")
    processor, model = load_model_and_processor(MODEL_ID)
    print("Model and processor successfully loaded!")
    print(f"Total model classes: {len(model.config.id2label)}")

    # Create synthetic test leaf (green canvas with texture)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :, 1] = 160  # Green channel
    arr[:, :, 0] = 40   # Slight Red
    arr[:, :, 2] = 30   # Slight Blue
    dummy_img = Image.fromarray(arr)

    print("Running inference on dummy image...")
    result = predict_image(dummy_img, processor, model, top_k=3)

    print("\n--- Smoke Test Results ---")
    print(f"Highest Prediction: {result['highest']['label_clean']}")
    print(f"Confidence: {result['highest']['percentage']}%")
    print(f"Low Confidence Flag: {result['is_low_confidence']}")
    print("Top 3 Predictions:")
    for i, pred in enumerate(result["top_predictions"], 1):
        print(f"  {i}. {pred['label_clean']} ({pred['percentage']}%) [Raw: {pred['label_raw']}]")

    print("\nSmoke test PASSED!")


if __name__ == "__main__":
    main()

