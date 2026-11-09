"""Unit tests for ESP32 IoT telemetry client (autonomous irrigation, no manual pump control)."""

import unittest
from unittest.mock import MagicMock, patch
import requests

from app.iot_client import fetch_esp32_telemetry, normalize_base_url


class TestESP32IoTClient(unittest.TestCase):
    def test_01_normalize_url(self):
        """Test URL normalization utility."""
        self.assertEqual(normalize_base_url("192.168.12.16"), "http://192.168.12.16")
        self.assertEqual(normalize_base_url("http://192.168.12.16/"), "http://192.168.12.16")
        self.assertEqual(normalize_base_url("https://esp32.local/"), "https://esp32.local")

    @patch("requests.get")
    def test_02_successful_telemetry_fetch(self, mock_get):
        """Test parsing valid JSON telemetry response from ESP32."""
        mock_sensor_resp = MagicMock()
        mock_sensor_resp.status_code = 200
        mock_sensor_resp.json.return_value = {
            "raw_value": 2480,
            "moisture": 32.5,
            "soil_status": "MODERATE",
            "pump_status": "OFF",
            "sensor_valid": True,
            "last_watering_duration": 5,
            "last_watering_elapsed_seconds": 120,
            "current_state": "IDLE",
            "shutoff_target": 35.0,
            "dry_threshold": 30.0,
            "watering_attempts": 0,
            "max_attempts": 3,
        }

        mock_status_resp = MagicMock()
        mock_status_resp.status_code = 200
        mock_status_resp.json.return_value = {
            "system_status": "ONLINE",
            "wifi_connected": True,
            "ip_address": "192.168.12.16",
            "uptime_seconds": 360,
            "wifi_rssi": -62,
        }

        mock_get.side_effect = [mock_sensor_resp, mock_status_resp]

        result = fetch_esp32_telemetry("http://192.168.12.16")
        self.assertTrue(result["online"])
        self.assertEqual(result["moisture"], 32.5)
        self.assertEqual(result["raw_value"], 2480)
        self.assertEqual(result["soil_status"], "MODERATE")
        self.assertEqual(result["pump_status"], "OFF")
        self.assertEqual(result["shutoff_target"], 35.0)
        self.assertEqual(result["dry_threshold"], 30.0)
        self.assertTrue(result["sensor_valid"])
        self.assertEqual(result["wifi_rssi"], -62)
        self.assertIsNone(result["error_message"])

    @patch("requests.get")
    def test_03_connection_timeout(self, mock_get):
        """Test handling connection timeout when ESP32 is offline."""
        mock_get.side_effect = requests.exceptions.ConnectTimeout("Timed out connecting")

        result = fetch_esp32_telemetry("http://192.168.12.16", timeout=1.0)
        self.assertFalse(result["online"])
        self.assertIsNone(result["moisture"])  # Must be None, NOT 0.0!
        self.assertIn("timed out", result["error_message"].lower())

    @patch("requests.get")
    def test_04_connection_error(self, mock_get):
        """Test handling connection refused / network error."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = fetch_esp32_telemetry("http://192.168.12.16")
        self.assertFalse(result["online"])
        self.assertIsNone(result["moisture"])
        self.assertIn("Failed to connect", result["error_message"])

    @patch("requests.get")
    def test_05_dry_soil_telemetry(self, mock_get):
        """Test telemetry parsing when ESP32 reports DRY condition and pump is active."""
        mock_sensor_resp = MagicMock()
        mock_sensor_resp.status_code = 200
        mock_sensor_resp.json.return_value = {
            "raw_value": 3100,
            "moisture": 18.0,
            "soil_status": "DRY",
            "pump_status": "ON",
            "sensor_valid": True,
            "last_watering_duration": 5,
            "last_watering_elapsed_seconds": 3,
            "current_state": "WATERING",
            "shutoff_target": 35.0,
            "dry_threshold": 30.0,
            "watering_attempts": 1,
            "max_attempts": 3,
        }
        mock_status_resp = MagicMock()
        mock_status_resp.status_code = 200
        mock_status_resp.json.return_value = {"uptime_seconds": 120, "wifi_rssi": -55}

        mock_get.side_effect = [mock_sensor_resp, mock_status_resp]

        result = fetch_esp32_telemetry("http://192.168.12.16")
        self.assertTrue(result["online"])
        self.assertEqual(result["soil_status"], "DRY")
        self.assertEqual(result["pump_status"], "ON")
        self.assertEqual(result["current_state"], "WATERING")
        self.assertEqual(result["moisture"], 18.0)
        self.assertLess(result["moisture"], result["dry_threshold"])

    @patch("requests.get")
    def test_06_cooldown_state_telemetry(self, mock_get):
        """Test telemetry when ESP32 is in soak cooldown after a 5-second pump cycle."""
        mock_sensor_resp = MagicMock()
        mock_sensor_resp.status_code = 200
        mock_sensor_resp.json.return_value = {
            "raw_value": 2800,
            "moisture": 24.0,
            "soil_status": "DRY",
            "pump_status": "OFF",
            "sensor_valid": True,
            "last_watering_duration": 5,
            "last_watering_elapsed_seconds": 2,
            "current_state": "COOLDOWN",
            "shutoff_target": 35.0,
            "dry_threshold": 30.0,
            "watering_attempts": 1,
            "max_attempts": 3,
        }
        mock_status_resp = MagicMock()
        mock_status_resp.status_code = 200
        mock_status_resp.json.return_value = {"uptime_seconds": 180, "wifi_rssi": -60}

        mock_get.side_effect = [mock_sensor_resp, mock_status_resp]

        result = fetch_esp32_telemetry("http://192.168.12.16")
        self.assertTrue(result["online"])
        self.assertEqual(result["current_state"], "COOLDOWN")
        self.assertEqual(result["pump_status"], "OFF")
        self.assertEqual(result["last_watering_duration"], 5)
        self.assertEqual(result["watering_attempts"], 1)

    @patch("requests.get")
    def test_07_lockout_state_telemetry(self, mock_get):
        """Test telemetry when ESP32 is in LOCKOUT after 3 failed watering attempts."""
        mock_sensor_resp = MagicMock()
        mock_sensor_resp.status_code = 200
        mock_sensor_resp.json.return_value = {
            "raw_value": 3200,
            "moisture": 12.0,
            "soil_status": "DRY",
            "pump_status": "OFF",
            "sensor_valid": True,
            "last_watering_duration": 5,
            "last_watering_elapsed_seconds": 30,
            "current_state": "LOCKOUT",
            "shutoff_target": 35.0,
            "dry_threshold": 30.0,
            "watering_attempts": 3,
            "max_attempts": 3,
        }
        mock_status_resp = MagicMock()
        mock_status_resp.status_code = 200
        mock_status_resp.json.return_value = {"uptime_seconds": 300, "wifi_rssi": -58}

        mock_get.side_effect = [mock_sensor_resp, mock_status_resp]

        result = fetch_esp32_telemetry("http://192.168.12.16")
        self.assertTrue(result["online"])
        self.assertEqual(result["current_state"], "LOCKOUT")
        self.assertEqual(result["pump_status"], "OFF")
        self.assertEqual(result["watering_attempts"], 3)
        self.assertEqual(result["max_attempts"], 3)
        # Pump should be OFF in lockout even though soil is dry
        self.assertFalse(result["sensor_valid"] and result["pump_status"] == "ON")

    @patch("requests.get")
    def test_08_sensor_invalid_telemetry(self, mock_get):
        """Test telemetry parsing when sensor_valid is False (disconnected sensor)."""
        mock_sensor_resp = MagicMock()
        mock_sensor_resp.status_code = 200
        mock_sensor_resp.json.return_value = {
            "raw_value": 0,
            "moisture": 0.0,
            "soil_status": "OUT_OF_BOUNDS",
            "pump_status": "OFF",
            "sensor_valid": False,
            "last_watering_duration": 0,
            "last_watering_elapsed_seconds": -1,
            "current_state": "LOCKOUT",
            "shutoff_target": 35.0,
            "dry_threshold": 30.0,
            "watering_attempts": 0,
            "max_attempts": 3,
        }
        mock_status_resp = MagicMock()
        mock_status_resp.status_code = 200
        mock_status_resp.json.return_value = {"uptime_seconds": 60, "wifi_rssi": -70}

        mock_get.side_effect = [mock_sensor_resp, mock_status_resp]

        result = fetch_esp32_telemetry("http://192.168.12.16")
        self.assertTrue(result["online"])
        self.assertFalse(result["sensor_valid"])
        self.assertEqual(result["pump_status"], "OFF")
        self.assertEqual(result["current_state"], "LOCKOUT")

    @patch("requests.get")
    def test_09_status_endpoint_failure_still_returns_sensor(self, mock_get):
        """Test that /sensor data is returned even if /status endpoint fails."""
        mock_sensor_resp = MagicMock()
        mock_sensor_resp.status_code = 200
        mock_sensor_resp.json.return_value = {
            "raw_value": 2200,
            "moisture": 45.0,
            "soil_status": "WET",
            "pump_status": "OFF",
            "sensor_valid": True,
            "last_watering_duration": 5,
            "last_watering_elapsed_seconds": 600,
            "current_state": "IDLE",
            "shutoff_target": 35.0,
            "dry_threshold": 30.0,
            "watering_attempts": 0,
            "max_attempts": 3,
        }

        mock_get.side_effect = [
            mock_sensor_resp,
            requests.exceptions.ConnectionError("Status endpoint unreachable"),
        ]

        result = fetch_esp32_telemetry("http://192.168.12.16")
        # /sensor succeeded so device is online
        self.assertTrue(result["online"])
        self.assertEqual(result["moisture"], 45.0)
        self.assertEqual(result["soil_status"], "WET")
        self.assertIsNone(result["error_message"])
        # /status failed so these are None
        self.assertIsNone(result["wifi_rssi"])
        self.assertIsNone(result["uptime_seconds"])

    @patch("requests.get")
    def test_10_invalid_json_response(self, mock_get):
        """Test graceful handling of non-JSON response from ESP32."""
        mock_sensor_resp = MagicMock()
        mock_sensor_resp.status_code = 200
        mock_sensor_resp.json.side_effect = ValueError("Not valid JSON")

        mock_get.return_value = mock_sensor_resp

        result = fetch_esp32_telemetry("http://192.168.12.16")
        self.assertFalse(result["online"])
        self.assertIsNotNone(result["error_message"])
        self.assertIn("JSON", result["error_message"])


if __name__ == "__main__":
    unittest.main()
