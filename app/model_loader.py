"""Model loader, image validation, and inference module for Plant Health Detection."""

import io
import os
from typing import Any, Dict, List, Tuple, Union
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.config import (
    CONFIDENCE_WARNING_THRESHOLD,
    MAX_IMAGE_DIMENSION,
    MODEL_ID,
    TOP_K,
    format_label,
)


def validate_and_load_image(image_input: Union[bytes, io.BytesIO, Image.Image]) -> Image.Image:
    """Validate, sanitize, and load a leaf image.
    
    Protects against:
    - Corrupted image bytes or truncated headers
    - Non-image files (text, executables, random data)
    - Smartphone EXIF orientation flips
    - Excessive image resolution causing high memory usage
    
    Args:
        image_input: Raw image bytes, BytesIO buffer, or existing PIL Image.
        
    Returns:
        Sanitized 3-channel RGB PIL Image ready for model preprocessing.
        
    Raises:
        ValueError: If file is corrupted or not an image.
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        if len(image_input) == 0:
            raise ValueError("The provided image file is empty (0 bytes).")
        try:
            img = Image.open(io.BytesIO(image_input))
            # Force load image data to catch truncated or corrupted files immediately
            img.load()
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ValueError(f"Corrupted or unsupported image file: {e}") from e
    elif hasattr(image_input, "read"):
        try:
            data = image_input.read()
            if len(data) == 0:
                raise ValueError("The uploaded image file is empty (0 bytes).")
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ValueError(f"Corrupted or unreadable image file: {e}") from e
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Automatically correct image orientation based on EXIF metadata (common in smartphone photos)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass  # If EXIF reading fails, continue with original orientation

    # Ensure 3-channel RGB mode (converts RGBA, Grayscale, Palette, CMYK)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Downscale very large images (e.g. 12+ Megapixels) to protect CPU memory and speed up processing
    max_dim = max(img.width, img.height)
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / float(max_dim)
        new_width = max(1, int(img.width * scale))
        new_height = max(1, int(img.height * scale))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def load_model_and_processor(
    model_id: str = MODEL_ID,
) -> Tuple[Any, Any]:
    """Load pretrained image processor and MobileNetV2 classification model on CPU.
    
    Args:
        model_id: Hugging Face model repository identifier.
        
    Returns:
        Tuple of (processor, model) in evaluation mode on CPU device.
        
    Raises:
        RuntimeError: If model cannot be downloaded or initialized.
    """
    try:
        # Check for optional HF_TOKEN in Streamlit secrets to allow higher Hugging Face download rate limits
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
                os.environ.setdefault("HF_TOKEN", str(st.secrets["HF_TOKEN"]))
        except Exception:
            pass

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        
        # Explicit CPU device placement
        device = torch.device("cpu")
        model.to(device)
        model.eval()  # Set model to evaluation mode (disables dropout layers)
        return processor, model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{model_id}' from Hugging Face Hub: {exc}"
        ) from exc


def predict_image(
    image: Union[Image.Image, bytes, io.BytesIO],
    processor: Any,
    model: Any,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """Run model inference on a leaf image and return top-k predictions with confidence.
    
    Args:
        image: PIL Image object, raw bytes, or BytesIO buffer of the leaf.
        processor: Loaded AutoImageProcessor.
        model: Loaded AutoModelForImageClassification.
        top_k: Number of highest-confidence predictions to return.
        
    Returns:
        Dictionary containing:
            - top_predictions: List of dicts (label_raw, label_clean, score, percentage, is_healthy, class_id)
            - highest: The top-1 prediction dict
            - is_low_confidence: Boolean flag if top-1 score < threshold
    """
    # Validate and sanitize input image
    rgb_image = validate_and_load_image(image)

    # Apply preprocessing transformations (resize 256, center crop 224, rescale, normalize)
    inputs = processor(images=rgb_image, return_tensors="pt")

    # Forward pass on CPU without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Extract top-k probabilities and corresponding class indices
    num_classes = len(probabilities)
    k = min(top_k, num_classes)
    top_scores, top_indices = torch.topk(probabilities, k=k)

    top_predictions: List[Dict[str, Any]] = []
    for score_tensor, idx_tensor in zip(top_scores, top_indices):
        class_id = int(idx_tensor.item())
        score = float(score_tensor.item())
        raw_label = model.config.id2label[class_id]
        clean_label = format_label(raw_label)
        is_healthy = "healthy" in raw_label.lower()

        top_predictions.append({
            "class_id": class_id,
            "label_raw": raw_label,
            "label_clean": clean_label,
            "score": score,
            "percentage": round(score * 100.0, 2),
            "is_healthy": is_healthy,
        })

    highest = top_predictions[0]
    is_low_confidence = highest["score"] < CONFIDENCE_WARNING_THRESHOLD

    return {
        "top_predictions": top_predictions,
        "highest": highest,
        "is_low_confidence": is_low_confidence,
        "image_size": rgb_image.size,
    }
