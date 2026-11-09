"""Configuration settings, constants, and label utilities for the Plant Health Monitoring System."""

# Application Info
APP_TITLE = "AIoT Plant Health Monitoring System"
APP_SUBTITLE = "Integrated AI disease screening, soil moisture monitoring, and smart irrigation"

# Model Constants
MODEL_ID = "neghani/plantvillage-mobilenetv2"
TOP_K = 3
CONFIDENCE_WARNING_THRESHOLD = 0.60  # 60% threshold for low-confidence warnings

# Image Dimension & Size Limits (Protects CPU RAM from multi-megapixel mobile photos)
MAX_IMAGE_DIMENSION = 2048  # Downscale if width or height exceeds 2048px
MAX_FILE_SIZE_MB = 25       # Streamlit upload limit reference

# File Upload Settings
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

# ESP32 IoT & Irrigation Configuration
DEFAULT_ESP32_IP = "http://192.168.12.16"
IOT_REQUEST_TIMEOUT = 2.5       # Seconds before network timeout
IOT_DEFAULT_REFRESH_SEC = 5     # Default polling interval in seconds
DRY_WARNING_THRESHOLD = 30.0    # Soil moisture % below which dry warning & sound trigger
PUMP_SHUTOFF_TARGET = 35.0      # Target soil moisture % where pump automatically shuts off

# Mandatory Advisory Disclaimer
DISCLAIMER_TEXT = (
    "This model was trained on PlantVillage lab-style images. It may perform "
    "poorly on real-world photographs. Predictions are suggestions, not confirmed "
    "diagnoses. Results should be verified by an agricultural expert."
)


def format_label(raw_label: str) -> str:
    """Format a raw PlantVillage class label into a clean, human-readable title.
    
    Example:
        'Tomato___Early_blight' -> 'Tomato - Early Blight'
        'Corn_(maize)___Common_rust_' -> 'Corn (Maize) - Common Rust'
        'Apple___healthy' -> 'Apple - Healthy'
    """
    if not raw_label or not isinstance(raw_label, str):
        return "Unknown Condition"

    if "___" not in raw_label:
        return raw_label.replace("_", " ").title()

    crop_part, condition_part = raw_label.split("___", 1)

    # Format crop
    crop = crop_part.replace("_", " ").strip()
    if "(maize)" in crop.lower():
        crop = "Corn (Maize)"
    elif "(including_sour)" in crop_part.lower():
        crop = "Cherry (Including Sour)"
    elif "pepper,_bell" in crop_part.lower():
        crop = "Bell Pepper"
    else:
        crop = crop.title()

    # Format condition / disease
    condition = condition_part.replace("_", " ").strip()
    if condition.lower() == "healthy":
        condition = "Healthy"
    else:
        condition = condition.title()

    return f"{crop} - {condition}"
