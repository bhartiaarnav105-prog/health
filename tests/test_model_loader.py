"""Automated tests for model loader, image preprocessing, and prediction logic."""

import os
import sys
import unittest
from PIL import Image
import numpy as np

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import CONFIDENCE_WARNING_THRESHOLD, MODEL_ID, TOP_K, format_label
from app.model_loader import load_model_and_processor, predict_image


class TestPlantHealthModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[Setup] Loading model and processor for test suite...")
        cls.processor, cls.model = load_model_and_processor(MODEL_ID)

    def test_01_label_formatter(self):
        """Verify format_label correctly parses PlantVillage labels."""
        self.assertEqual(format_label("Tomato___Early_blight"), "Tomato - Early Blight")
        self.assertEqual(format_label("Apple___healthy"), "Apple - Healthy")
        self.assertEqual(
            format_label("Corn_(maize)___Common_rust_"),
            "Corn (Maize) - Common Rust",
        )
        self.assertEqual(
            format_label("Pepper,_bell___Bacterial_spot"),
            "Bell Pepper - Bacterial Spot",
        )

    def test_02_valid_jpg_prediction(self):
        """Test inference with a valid JPG image."""
        img_path = os.path.join("samples", "test_leaf.jpg")
        self.assertTrue(os.path.exists(img_path))
        with Image.open(img_path) as img:
            res = predict_image(img, self.processor, self.model, top_k=TOP_K)
            self.assertIn("top_predictions", res)
            self.assertEqual(len(res["top_predictions"]), 3)
            self.assertIn("highest", res)
            self.assertGreaterEqual(res["highest"]["score"], 0.0)
            self.assertLessEqual(res["highest"]["score"], 1.0)
            print(f"JPG Top Prediction: {res['highest']['label_clean']} ({res['highest']['percentage']}%)")

    def test_03_valid_jpeg_prediction(self):
        """Test inference with a valid JPEG image."""
        img_path = os.path.join("samples", "test_leaf.jpeg")
        self.assertTrue(os.path.exists(img_path))
        with Image.open(img_path) as img:
            res = predict_image(img, self.processor, self.model, top_k=TOP_K)
            self.assertEqual(len(res["top_predictions"]), 3)
            print(f"JPEG Top Prediction: {res['highest']['label_clean']} ({res['highest']['percentage']}%)")

    def test_04_valid_png_rgba_prediction(self):
        """Test inference with a valid PNG image containing RGBA alpha channel."""
        img_path = os.path.join("samples", "test_leaf.png")
        self.assertTrue(os.path.exists(img_path))
        with Image.open(img_path) as img:
            self.assertEqual(img.mode, "RGBA")
            res = predict_image(img, self.processor, self.model, top_k=TOP_K)
            self.assertEqual(len(res["top_predictions"]), 3)
            print(f"PNG Top Prediction: {res['highest']['label_clean']} ({res['highest']['percentage']}%)")

    def test_05_invalid_input_type(self):
        """Test error handling when passing invalid non-image objects."""
        with self.assertRaises(TypeError):
            predict_image("not_an_image", self.processor, self.model)

    def test_06_top_3_probabilities_descending(self):
        """Verify Top-3 predictions are sorted in descending order of confidence."""
        img_path = os.path.join("samples", "test_leaf.jpg")
        with Image.open(img_path) as img:
            res = predict_image(img, self.processor, self.model, top_k=3)
            scores = [p["score"] for p in res["top_predictions"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_07_confidence_warning_trigger(self):
        """Verify low confidence warning flag (<60%) triggers when uncertainty is high."""
        img_path = os.path.join("samples", "low_confidence_leaf.jpg")
        with Image.open(img_path) as img:
            res = predict_image(img, self.processor, self.model, top_k=3)
            top_score = res["highest"]["score"]
            print(f"Low-confidence test sample top score: {top_score*100:.2f}%")
            self.assertLess(top_score, CONFIDENCE_WARNING_THRESHOLD)
            self.assertTrue(res["is_low_confidence"])


if __name__ == "__main__":
    unittest.main()
