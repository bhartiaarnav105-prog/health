"""Phase 3 Reliability and Edge Case Test Suite for AIoT Plant Health Monitoring System."""

import io
import os
import sys
import unittest
from PIL import Image

# Ensure app package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import (
    CONFIDENCE_WARNING_THRESHOLD,
    MAX_IMAGE_DIMENSION,
    MODEL_ID,
    TOP_K,
    format_label,
)
from app.model_loader import (
    load_model_and_processor,
    predict_image,
    validate_and_load_image,
)


class TestPhase3Reliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[Phase 3 Setup] Initializing cached model and processor on CPU...")
        cls.processor, cls.model = load_model_and_processor(MODEL_ID)

    def test_01_healthy_leaf_image(self):
        """Case 1: Test prediction on a healthy leaf image."""
        img_path = os.path.join("samples", "healthy_leaf.jpg")
        self.assertTrue(os.path.exists(img_path))
        with open(img_path, "rb") as f:
            res = predict_image(f.read(), self.processor, self.model, top_k=TOP_K)
            self.assertEqual(len(res["top_predictions"]), 3)
            self.assertIn("score", res["highest"])
            print(f"Case 1 (Healthy Leaf): Predicted '{res['highest']['label_clean']}' ({res['highest']['percentage']}%)")

    def test_02_diseased_leaf_image(self):
        """Case 2: Test prediction on a diseased leaf image."""
        img_path = os.path.join("samples", "diseased_leaf.jpg")
        self.assertTrue(os.path.exists(img_path))
        with open(img_path, "rb") as f:
            res = predict_image(f.read(), self.processor, self.model, top_k=TOP_K)
            self.assertEqual(len(res["top_predictions"]), 3)
            print(f"Case 2 (Diseased Leaf): Predicted '{res['highest']['label_clean']}' ({res['highest']['percentage']}%)")

    def test_03_different_plant_species_coverage(self):
        """Case 3: Verify the model covers diverse plant species from PlantVillage."""
        all_labels = list(self.model.config.id2label.values())
        self.assertEqual(len(all_labels), 38)

        # Verify species diversity in class dictionary
        species_keywords = ["Apple", "Corn", "Grape", "Peach", "Pepper", "Potato", "Tomato", "Strawberry", "Cherry"]
        for sp in species_keywords:
            matching = [lbl for lbl in all_labels if sp.lower() in lbl.lower()]
            self.assertGreater(len(matching), 0, f"Species {sp} must be present in labels")
        print(f"Case 3 (Species Coverage): Verified {len(species_keywords)} target crop species across 38 classes.")

    def test_04_high_confidence_prediction(self):
        """Case 4: Test a high-confidence prediction (> 60%)."""
        img_path = os.path.join("samples", "test_leaf.png")
        with open(img_path, "rb") as f:
            res = predict_image(f.read(), self.processor, self.model, top_k=TOP_K)
            self.assertGreater(res["highest"]["score"], CONFIDENCE_WARNING_THRESHOLD)
            self.assertFalse(res["is_low_confidence"])
            print(f"Case 4 (High Confidence): Score {res['highest']['percentage']}% > 60% threshold.")

    def test_05_low_confidence_prediction(self):
        """Case 5: Test that ambiguous input triggers low-confidence flag (< 60%)."""
        img_path = os.path.join("samples", "low_confidence_leaf.jpg")
        with open(img_path, "rb") as f:
            res = predict_image(f.read(), self.processor, self.model, top_k=TOP_K)
            self.assertLess(res["highest"]["score"], CONFIDENCE_WARNING_THRESHOLD)
            self.assertTrue(res["is_low_confidence"])
            print(f"Case 5 (Low Confidence): Score {res['highest']['percentage']}% < 60% -> warning triggered.")

    def test_06_invalid_image_data(self):
        """Case 6: Test invalid non-image file (plain text / invalid bytes)."""
        invalid_bytes = b"Hello world, this is a plain text file, not an image."
        with self.assertRaises(ValueError) as ctx:
            validate_and_load_image(invalid_bytes)
        self.assertIn("Corrupted or unsupported", str(ctx.exception))
        print("Case 6 (Invalid Image): Successfully caught ValueError on non-image bytes.")

    def test_07_corrupted_image(self):
        """Case 7: Test corrupted / truncated image bytes."""
        corrupted_path = os.path.join("samples", "corrupted_image.jpg")
        self.assertTrue(os.path.exists(corrupted_path))
        with open(corrupted_path, "rb") as f:
            corrupted_data = f.read()
        with self.assertRaises(ValueError) as ctx:
            validate_and_load_image(corrupted_data)
        self.assertIn("Corrupted or unsupported", str(ctx.exception))
        print("Case 7 (Corrupted Image): Successfully caught ValueError on truncated image.")

    def test_08_very_large_image(self):
        """Case 8: Test very large image (4000x3000, 12 MP) auto-downscaling."""
        large_path = os.path.join("samples", "very_large_leaf.jpg")
        self.assertTrue(os.path.exists(large_path))
        with open(large_path, "rb") as f:
            raw_bytes = f.read()
        # Validate that image is safely loaded and downscaled below MAX_IMAGE_DIMENSION
        sanitized = validate_and_load_image(raw_bytes)
        self.assertLessEqual(max(sanitized.width, sanitized.height), MAX_IMAGE_DIMENSION)
        # Ensure prediction runs smoothly without memory exhaustion
        res = predict_image(sanitized, self.processor, self.model, top_k=TOP_K)
        self.assertEqual(len(res["top_predictions"]), 3)
        print(f"Case 8 (Very Large Image): Successfully downscaled from 4000x3000 to {sanitized.size} and predicted.")

    def test_09_unsupported_image_format(self):
        """Case 9: Test detection of unsupported formats (e.g. .webp, .bmp not in ALLOWED_EXTENSIONS)."""
        from app.config import ALLOWED_EXTENSIONS
        for fmt in ["webp", "bmp", "tiff", "gif", "svg"]:
            self.assertNotIn(fmt, ALLOWED_EXTENSIONS)
        print("Case 9 (Unsupported Formats): Verified rejection policy for webp, bmp, tiff, gif.")

    def test_10_model_download_failure(self):
        """Case 10: Test graceful error handling when loading non-existent model ID."""
        with self.assertRaises(RuntimeError) as ctx:
            load_model_and_processor("nonexistent/invalid-model-id-999")
        self.assertIn("Failed to load model", str(ctx.exception))
        print("Case 10 (Model Download Failure): Caught RuntimeError with informative message.")

    def test_11_missing_dependencies_check(self):
        """Case 11: Verify all required production dependencies are present and importable."""
        modules = ["torch", "torchvision", "transformers", "PIL", "streamlit", "pandas"]
        for mod in modules:
            __import__(mod)
        print("Case 11 (Dependencies): Verified all 6 core libraries are installed and operational.")

    def test_12_empty_upload_handling(self):
        """Case 12: Verify empty bytes or empty buffer handling."""
        with self.assertRaises(ValueError) as ctx:
            validate_and_load_image(b"")
        self.assertIn("empty", str(ctx.exception).lower())
        print("Case 12 (Empty Upload): Successfully rejected empty 0-byte payload.")


if __name__ == "__main__":
    unittest.main()

