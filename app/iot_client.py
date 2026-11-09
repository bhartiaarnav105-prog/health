"""ESP32 IoT Client for telemetry retrieval from the autonomous irrigation REST API."""

from datetime import datetime
from typing import Any, Dict, Optional
import requests

from app.config import DEFAULT_ESP32_IP, IOT_REQUEST_TIMEOUT


def normalize_base_url(url: str) -> str:
    """Ensure URL has http:// scheme and no trailing slash."""
    clean_url = url.strip()
    if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
        clean_url = "http://" + clean_url
    return clean_url.rstrip("/")


def fetch_esp32_telemetry(
    base_url: str = DEFAULT_ESP32_IP,
    timeout: float = IOT_REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """Fetch live sensor telemetry and device status from ESP32 REST API.
    
    Args:
        base_url: Base URL of ESP32 (e.g. 'http://192.168.12.16').
        timeout: Network request timeout in seconds.
        
    Returns:
        Dictionary containing live telemetry or detailed error diagnosis.
    """
    clean_url = normalize_base_url(base_url)
    sensor_endpoint = f"{clean_url}/sensor"
    status_endpoint = f"{clean_url}/status"
    now_str = datetime.now().strftime("%H:%M:%S")

    result: Dict[str, Any] = {
        "online": False,
        "timestamp": now_str,
        "base_url": clean_url,
        "raw_value": None,
        "moisture": None,
        "soil_status": "UNAVAILABLE",
        "pump_status": "UNAVAILABLE",
        "sensor_valid": False,
        "last_watering_duration": None,
        "last_watering_elapsed_seconds": None,
        "current_state": "DISCONNECTED",
        "shutoff_target": 35.0,
        "dry_threshold": 30.0,
        "watering_attempts": None,
        "max_attempts": None,
        "uptime_seconds": None,
        "wifi_rssi": None,
        "ip_address": None,
        "error_message": None,
    }

    try:
        # 1. Fetch sensor telemetry
        sensor_resp = requests.get(sensor_endpoint, timeout=timeout)
        if sensor_resp.status_code == 200:
            sensor_data = sensor_resp.json()
            result["online"] = True
            result["raw_value"] = sensor_data.get("raw_value")
            result["moisture"] = sensor_data.get("moisture")
            result["soil_status"] = sensor_data.get("soil_status", "UNKNOWN")
            result["pump_status"] = sensor_data.get("pump_status", "UNKNOWN")
            result["sensor_valid"] = bool(sensor_data.get("sensor_valid", False))
            result["last_watering_duration"] = sensor_data.get("last_watering_duration")
            result["last_watering_elapsed_seconds"] = sensor_data.get("last_watering_elapsed_seconds")
            result["current_state"] = sensor_data.get("current_state", "UNKNOWN")
            result["shutoff_target"] = sensor_data.get("shutoff_target", 35.0)
            result["dry_threshold"] = sensor_data.get("dry_threshold", 30.0)
            result["watering_attempts"] = sensor_data.get("watering_attempts")
            result["max_attempts"] = sensor_data.get("max_attempts")
        else:
            result["error_message"] = f"ESP32 returned HTTP {sensor_resp.status_code} for /sensor."
            return result

        # 2. Fetch status telemetry (optional enrichment)
        try:
            status_resp = requests.get(status_endpoint, timeout=timeout)
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                result["uptime_seconds"] = status_data.get("uptime_seconds")
                result["wifi_rssi"] = status_data.get("wifi_rssi")
                result["ip_address"] = status_data.get("ip_address")
        except Exception:
            pass  # If /status fails but /sensor succeeded, telemetry is still valid

    except requests.exceptions.ConnectTimeout:
        result["error_message"] = (
            f"Connection timed out ({timeout}s). ESP32 at {clean_url} is unreachable. "
            "Please check power or Wi-Fi connection."
        )
    except requests.exceptions.ConnectionError:
        result["error_message"] = (
            f"Failed to connect to {clean_url}. Ensure your laptop is on the same Wi-Fi "
            "network and the IP address is correct."
        )
    except requests.exceptions.RequestException as req_err:
        result["error_message"] = f"Network error communicating with ESP32: {req_err}"
    except ValueError as json_err:
        result["error_message"] = f"Invalid JSON received from ESP32: {json_err}"
    except Exception as ex:
        result["error_message"] = f"Unexpected error: {ex}"

    return result
