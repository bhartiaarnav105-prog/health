"""End-to-end tests for the Streamlit dashboard using AppTest."""

import os
import unittest
from streamlit.testing.v1 import AppTest


DASHBOARD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "dashboard.py"))


class TestStreamlitDashboard(unittest.TestCase):
    def test_01_initial_dashboard_render(self):
        """Test initial state without any file uploaded."""
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30)
        at.run()
        self.assertFalse(at.exception)

        # Check title and sidebar
        self.assertTrue(any("AIoT Plant Health Monitoring System" in title.value for title in at.title))
        
        # Verify predict button is initially disabled because no file is uploaded
        predict_button = next(b for b in at.button if "Run Disease Prediction" in b.label)
        self.assertTrue(predict_button.disabled)

        # Verify initial info prompt is present
        self.assertTrue(any("Upload a leaf photo" in info.value for info in at.info))

    def test_02_empty_upload(self):
        """Test dashboard behavior when no file is uploaded."""
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30)
        at.run()
        uploader = at.file_uploader[0]
        self.assertIsNone(uploader.value)
        predict_button = next(b for b in at.button if "Run Disease Prediction" in b.label)
        self.assertTrue(predict_button.disabled)

    def test_03_valid_jpg_upload_and_predict(self):
        """Test uploading a valid JPG file, clicking predict, and checking top-3 and history."""
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30)
        at.run()

        # Upload JPG file
        jpg_path = os.path.join("samples", "test_leaf.jpg")
        with open(jpg_path, "rb") as f:
            file_data = f.read()

        uploader = at.file_uploader[0]
        uploader.upload(filename="test_leaf.jpg", content=file_data, mime_type="image/jpeg")
        at.run()

        # Image preview should now be rendered
        self.assertTrue(len(at.image) > 0)

        # Predict button should now be enabled
        predict_button = next(b for b in at.button if "Run Disease Prediction" in b.label)
        self.assertFalse(predict_button.disabled)

        # Click predict button
        predict_button.click()
        at.run()

        self.assertFalse(at.exception)

        # Verify results container and top-3 differential predictions rendered
        markdown_text = " ".join([m.value for m in at.markdown])
        self.assertIn("Top-3 Differential Predictions", markdown_text)
        self.assertIn("Confidence Comparison", markdown_text)

        # Verify prediction history table is populated
        self.assertTrue(len(at.dataframe) > 0)
        self.assertEqual(len(at.session_state.prediction_history), 1)

    def test_04_valid_png_upload_and_predict(self):
        """Test uploading a PNG file and predicting."""
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30)
        at.run()

        png_path = os.path.join("samples", "test_leaf.png")
        with open(png_path, "rb") as f:
            file_data = f.read()

        at.file_uploader[0].upload(filename="test_leaf.png", content=file_data, mime_type="image/png")
        at.run()

        predict_button = next(b for b in at.button if "Run Disease Prediction" in b.label)
        predict_button.click()
        at.run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state.prediction_history), 1)

    def test_05_low_confidence_warning_trigger(self):
        """Test low confidence warning when uploading ambiguous image."""
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30)
        at.run()

        low_conf_path = os.path.join("samples", "low_confidence_leaf.jpg")
        with open(low_conf_path, "rb") as f:
            file_data = f.read()

        at.file_uploader[0].upload(filename="low_confidence_leaf.jpg", content=file_data, mime_type="image/jpeg")
        at.run()

        predict_button = next(b for b in at.button if "Run Disease Prediction" in b.label)
        predict_button.click()
        at.run()

        # Check for warning component
        warnings = [w.value for w in at.warning]
        self.assertTrue(any("Low Confidence Warning" in w for w in warnings))

    def test_06_reset_button(self):
        """Test reset button clears current analysis."""
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30)
        at.run()

        jpg_path = os.path.join("samples", "test_leaf.jpg")
        with open(jpg_path, "rb") as f:
            file_data = f.read()

        at.file_uploader[0].upload(filename="test_leaf.jpg", content=file_data, mime_type="image/jpeg")
        at.run()

        predict_button = next(b for b in at.button if "Run Disease Prediction" in b.label)
        predict_button.click()
        at.run()
        self.assertIsNotNone(at.session_state.current_result)

        # Click Reset
        reset_button = next(b for b in at.button if "Reset" in b.label and "Run" not in b.label)
        reset_button.click()
        at.run()

        self.assertIsNone(at.session_state.current_result)


if __name__ == "__main__":
    unittest.main()
