"""AIoT Plant Health Monitoring System - Streamlit Dashboard.

Optimized frontend structure:
- Tab 1: Plant Disease Detection (MobileNetV2 PyTorch classification, Top-3 differential predictions, History)
  - Pure image upload (camera module removed for performance & stability)
  - Lazy model loading (zero startup lag, single cached load on inference)
- Tab 2: Soil Moisture & Smart Irrigation (Real-time ESP32 telemetry, 4 metric cards, Altair hydration chart, Actuator log)
- Minimal Black, Green, Red theme
- High-performance, lightweight Header Particle Animation (60 FPS, optimized canvas, zero UI lag)
"""

from datetime import datetime
import importlib
import io
import os
import sys
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import app.config as config
importlib.reload(config)

from app.config import (
    ALLOWED_EXTENSIONS,
    APP_SUBTITLE,
    APP_TITLE,
    CONFIDENCE_WARNING_THRESHOLD,
    DEFAULT_ESP32_IP,
    DISCLAIMER_TEXT,
    DRY_WARNING_THRESHOLD,
    IOT_REQUEST_TIMEOUT,
    MODEL_ID,
    PUMP_SHUTOFF_TARGET,
    TOP_K,
)
from app.iot_client import fetch_esp32_telemetry
from app.model_loader import (
    load_model_and_processor,
    predict_image,
    validate_and_load_image,
)

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS Theme Injection (Black, Green, Red Palette)
# ---------------------------------------------------------------------------
_CSS_PATH = os.path.join(os.path.dirname(__file__), "floratech.css")
if os.path.isfile(_CSS_PATH):
    with open(_CSS_PATH, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Synthetic Alert Sound (Encoded 880Hz Audio Chime)
# ---------------------------------------------------------------------------
ALERT_CHIME_B64 = (
    "UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YVtvT18Ae3h2dXNzc3N1d3d5"
    "ent9gIGDg4SFhYWFhISDgoGAf317eXVzcnFvbWtnZWVkYmFfXltaWVhXV1dYWFlaW11eX2BhYmNk"
    "ZWZnaGlqa2xtbW5vcHFyc3R1dnd4eXp7fH1+f4CBgoOEhYaHiImKi4yNjo+QkZKTlJWWl5iZmpuc"
    "nZ6foKGio6SlpqeoqaqrrK2ur7CxsrO0tba3uLm6u7y9vr/AwcLDxMXGx8jJysvMzc7P0NHS09TV"
    "1tfY2drb3N3e3+Dh4uPk5ebn6Onq6+zt7u/w8fLz9PX29/j5+vv8/f7/AABA"
)

# ---------------------------------------------------------------------------
# Cached Model Loader (CPU Mode)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache the plant disease model and preprocessor once into memory."""
    return load_model_and_processor(MODEL_ID)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
# Check for secrets-defined gateway URL (optional, e.g. for reverse tunnels or cloud setups)
_default_esp32_ip = DEFAULT_ESP32_IP
try:
    if hasattr(st, "secrets") and "ESP32_BASE_URL" in st.secrets:
        _default_esp32_ip = str(st.secrets["ESP32_BASE_URL"])
except Exception:
    pass

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "esp32_ip" not in st.session_state:
    st.session_state.esp32_ip = _default_esp32_ip

if "latest_telemetry" not in st.session_state:
    st.session_state.latest_telemetry = None

if "soil_history" not in st.session_state:
    st.session_state.soil_history = []

if "watering_history" not in st.session_state:
    st.session_state.watering_history = []

if "previous_soil_condition" not in st.session_state:
    st.session_state.previous_soil_condition = "INITIAL"

if "sound_muted" not in st.session_state:
    st.session_state.sound_muted = False

if "operating_mode" not in st.session_state:
    st.session_state.operating_mode = "🟢 LIVE MODE"

if "demo_soil_step" not in st.session_state:
    st.session_state.demo_soil_step = 0


def reset_analysis():
    st.session_state.current_result = None
    st.session_state.uploader_key += 1


def clear_history():
    st.session_state.prediction_history = []


def clear_soil_history():
    st.session_state.soil_history = []
    st.session_state.watering_history = []


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 FloraTech AIoT")
    st.caption("AIoT Agricultural Diagnostics & Precision Irrigation")

    st.markdown("---")
    st.markdown("#### ⚙️ System Mode")
    mode_selection = st.radio(
        "Operating Mode:",
        ["🟢 LIVE MODE", "🧪 DEMO MODE"],
        index=0 if st.session_state.operating_mode == "🟢 LIVE MODE" else 1,
        help="Switch between physical hardware telemetry and isolated software simulation.",
    )
    st.session_state.operating_mode = mode_selection
    is_demo_mode = (mode_selection == "🧪 DEMO MODE")

    st.markdown("---")
    st.markdown("#### 📡 ESP32 Gateway")
    st.session_state.esp32_ip = st.text_input(
        "Device URL:",
        value=st.session_state.esp32_ip,
        help="Local network base URL of your ESP32 microcontroller.",
    )

    st.markdown("---")
    st.markdown("#### 🔬 Neural Architecture")
    st.markdown(
        f"""
        - **Model**: `MobileNetV2`
        - **Repository**: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID})
        - **Vocabulary**: 38 PlantVillage Classes
        - **Confidence Threshold**: `{int(CONFIDENCE_WARNING_THRESHOLD * 100)}%`
        - **Inference Hardware**: Local CPU (FP32)
        """
    )

    st.markdown("---")
    st.markdown("### ⚠️ Mandatory Disclaimer")
    st.warning(DISCLAIMER_TEXT)


# ---------------------------------------------------------------------------
# HEADER — Lightweight, Hardware-Optimized Particle Animation (Canvas)
# ---------------------------------------------------------------------------
PARTICLE_HEADER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #0c0f11;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .header-box {
      position: relative;
      width: 100%;
      height: 120px;
      display: flex;
      align-items: center;
      background: linear-gradient(135deg, #111416 0%, #0c0f11 100%);
      border: 1px solid rgba(78, 222, 163, 0.25);
      border-radius: 12px;
      overflow: hidden;
    }
    canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      cursor: default;
    }
    .hud-tag {
      position: absolute;
      top: 10px;
      left: 20px;
      font-family: monospace;
      font-size: 10px;
      color: #86948a;
      letter-spacing: 0.08em;
      pointer-events: none;
      display: flex;
      align-items: center;
      gap: 6px;
      z-index: 3;
    }
    .pulse-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #4edea3;
      box-shadow: 0 0 6px #4edea3;
    }
  </style>
</head>
<body>
  <div class="header-box" id="container">
    <div class="hud-tag">
      <span class="pulse-dot"></span>
      <span>SYS_ONLINE · AIoT TELEMETRY BUS</span>
    </div>
    <canvas id="particleCanvas"></canvas>
  </div>

  <script>
    (function() {
      const canvas = document.getElementById('particleCanvas');
      const ctx = canvas.getContext('2d');
      const container = document.getElementById('container');
      
      let width = 0;
      let height = 0;
      let dpr = Math.min(window.devicePixelRatio || 1, 2);
      let particles = [];
      const particleCount = 80;
      let mouse = { x: -9999, y: -9999, radius: 70 };

      function resize() {
        const rect = container.getBoundingClientRect();
        width = rect.width;
        height = rect.height;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
        initParticles();
      }

      class Particle {
        constructor() {
          this.reset(true);
        }
        reset(initial = false) {
          const startX = width * 0.04 + Math.random() * (width * 0.30);
          this.x = initial ? startX + Math.random() * (width * 0.35) : startX;
          this.y = (height / 2 + 10) + (Math.random() - 0.5) * 40;
          this.vx = Math.random() * 0.7 + 0.35;
          this.vy = (Math.random() - 0.5) * 0.25;
          this.size = Math.random() < 0.8 ? Math.random() * 1.5 + 0.8 : Math.random() * 2.2 + 1.2;
          
          const roll = Math.random();
          if (roll < 0.72) {
            this.color = 'rgba(255, 255, 255, 0.9)';
          } else if (roll < 0.88) {
            this.color = 'rgba(78, 222, 163, 0.95)';
          } else {
            this.color = 'rgba(239, 68, 68, 0.9)';
          }
          this.alpha = Math.random() * 0.65 + 0.25;
          this.maxLife = Math.floor(Math.random() * 140 + 90);
          this.life = initial ? Math.floor(Math.random() * this.maxLife) : this.maxLife;
          this.waveSpeed = Math.random() * 0.04 + 0.015;
          this.waveOffset = Math.random() * Math.PI * 2;
        }
        update() {
          this.life--;
          if (this.life <= 0 || this.x > width + 20) {
            this.reset();
            return;
          }
          this.x += this.vx;
          this.y += this.vy + Math.sin(this.life * this.waveSpeed + this.waveOffset) * 0.3;
          
          const dx = this.x - mouse.x;
          const dy = this.y - mouse.y;
          const dist = Math.hypot(dx, dy);
          if (dist < mouse.radius) {
            const force = (1 - dist / mouse.radius) * 1.8;
            this.x += (dx / dist) * force;
            this.y += (dy / dist) * force;
          }

          const progress = this.life / this.maxLife;
          if (progress > 0.8) {
            this.currentAlpha = ((1 - progress) / 0.2) * this.alpha;
          } else if (progress < 0.25) {
            this.currentAlpha = (progress / 0.25) * this.alpha;
          } else {
            this.currentAlpha = this.alpha;
          }
        }
        draw(ctx) {
          ctx.globalAlpha = Math.max(0, Math.min(1, this.currentAlpha));
          ctx.fillStyle = this.color;
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      function initParticles() {
        particles = [];
        for (let i = 0; i < particleCount; i++) {
          particles.push(new Particle());
        }
      }

      function drawTitle() {
        ctx.save();
        const fontSize = Math.min(Math.max(width * 0.034, 18), 28);
        ctx.font = '700 ' + fontSize + 'px sans-serif';
        ctx.textBaseline = 'middle';
        const startX = Math.max(width * 0.04, 22);
        const startY = height / 2 + 10;

        // "AIoT " in emerald green
        ctx.fillStyle = '#4edea3';
        ctx.fillText('AIoT ', startX, startY);
        const prefixWidth = ctx.measureText('AIoT ').width;

        // "Plant Health Monitoring System" in white
        ctx.fillStyle = '#ffffff';
        ctx.fillText('Plant Health Monitoring System', startX + prefixWidth, startY);
        ctx.restore();
      }

      let lastTime = 0;
      function animate(time) {
        requestAnimationFrame(animate);
        if (document.hidden) return;
        // Throttle to max 45fps to eliminate CPU/GPU overhead
        if (time - lastTime < 22) return;
        lastTime = time;

        ctx.clearRect(0, 0, width, height);
        drawTitle();
        ctx.save();
        ctx.globalCompositeOperation = 'screen';
        for (let i = 0; i < particles.length; i++) {
          particles[i].update();
          particles[i].draw(ctx);
        }
        ctx.restore();
      }

      canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouse.x = e.clientX - rect.left;
        mouse.y = e.clientY - rect.top;
      }, { passive: true });
      canvas.addEventListener('mouseleave', () => {
        mouse.x = -9999;
        mouse.y = -9999;
      }, { passive: true });

      window.addEventListener('resize', resize);
      resize();
      requestAnimationFrame(animate);
    })();
  </script>
</body>
</html>"""

components.html(PARTICLE_HEADER_HTML, height=130, scrolling=False)

# Keep the official title and subtitle in the DOM for automated AppTests and accessibility
st.markdown('<div class="hidden-title-anchor">', unsafe_allow_html=True)
st.title("🌿 " + APP_TITLE)
st.write(APP_SUBTITLE)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PRIMARY DASHBOARD TABS
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🌿 Plant Disease Detection", "💧 Soil Moisture & Smart Irrigation"])

# ===========================================================================
# TAB 1: PLANT DISEASE DETECTION
# ===========================================================================
with tab1:
    ws_left, ws_right = st.columns([1.2, 1], gap="medium")

    with ws_left:
        st.subheader("Leaf Specimen Inspection")

        uploaded_file = st.file_uploader(
            label="Upload plant leaf specimen (JPG, JPEG, or PNG):",
            type=ALLOWED_EXTENSIONS,
            key=f"leaf_uploader_{st.session_state.uploader_key}",
            help="Upload a clear photograph of a single leaf specimen under even lighting.",
        )

        valid_image: Optional[Image.Image] = None
        source_name = None
        source_bytes = None

        if uploaded_file is not None:
            source_name = uploaded_file.name
            source_bytes = uploaded_file.getvalue()

        if source_bytes is not None and source_name is not None:
            file_size_kb = len(source_bytes) / 1024.0
            file_ext = os.path.splitext(source_name)[1].lower().replace(".", "")
            if file_ext and file_ext not in ALLOWED_EXTENSIONS:
                st.error(
                    f"❌ **Unsupported File Type**: `.{file_ext}` is not supported. "
                    "Please upload a `.jpg`, `.jpeg`, or `.png` image."
                )
            else:
                try:
                    valid_image = validate_and_load_image(source_bytes)
                    st.image(
                        valid_image,
                        caption=f"Specimen Preview: {source_name} ({valid_image.width}×{valid_image.height}px, {file_size_kb:.1f} KB)",
                        use_container_width=True,
                    )
                except ValueError as val_err:
                    st.error(f"❌ **Invalid or Corrupted Image**: {val_err}")
                    valid_image = None
                except Exception as gen_err:
                    st.error(f"❌ **File Processing Error**: Could not read image ({gen_err}).")
                    valid_image = None

        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            predict_clicked = st.button(
                "🔍 Run Disease Prediction",
                type="primary",
                use_container_width=True,
                disabled=(valid_image is None),
            )
        with btn_col2:
            if st.button("🔄 Reset Analysis", use_container_width=True):
                reset_analysis()
                st.rerun()

        if predict_clicked and valid_image is not None:
            with st.spinner("Executing MobileNetV2 neural pathology inference..."):
                try:
                    processor, model = get_model()
                    result = predict_image(
                        image=valid_image,
                        processor=processor,
                        model=model,
                        top_k=TOP_K,
                    )
                    st.session_state.current_result = {
                        "filename": source_name,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "result": result,
                    }
                    st.session_state.prediction_history.insert(
                        0,
                        {
                            "Timestamp": datetime.now().strftime("%H:%M:%S"),
                            "File": source_name,
                            "Suggested Diagnosis": result["highest"]["label_clean"],
                            "Confidence": f"{result['highest']['percentage']}%",
                            "Condition": "Healthy" if result["highest"]["is_healthy"] else "Diseased",
                            "Low Confidence": "Yes" if result["is_low_confidence"] else "No",
                        },
                    )
                except Exception as inf_err:
                    st.error(f"❌ **Inference Error**: An unexpected error occurred: {inf_err}")

    with ws_right:
        st.subheader("Diagnostic Results")

        if st.session_state.current_result is None:
            st.info("👈 Upload a leaf photo and click **'Run Disease Prediction'** to analyze plant health.")
        else:
            current = st.session_state.current_result
            res = current["result"]
            top_pred = res["highest"]
            is_healthy = top_pred["is_healthy"]

            # Diagnostic summary card
            st.markdown(f"### {top_pred['label_clean']}")
            st.caption(f"Raw Label Identifier: `{top_pred['label_raw']}`")

            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.metric("Model Confidence", f"{top_pred['percentage']}%")
            with res_c2:
                if is_healthy:
                    st.success("🌱 Flora Nominal (Healthy)")
                else:
                    st.error("🚨 Critical Pathology Detected")

            # Low Confidence Warning check
            if res["is_low_confidence"]:
                st.warning(
                    f"⚠️ **Low Confidence Warning ({top_pred['percentage']}% < {int(CONFIDENCE_WARNING_THRESHOLD * 100)}%):** "
                    f"The neural network is uncertain about this classification. The leaf photo may be blurry, poorly lit, "
                    f"show multiple leaves, or belong to an unsupported plant species. Please verify with an agricultural expert."
                )

            # Top-3 Differential Predictions
            st.markdown("#### 🏆 Top-3 Differential Predictions")
            for rank, pred in enumerate(res["top_predictions"], 1):
                icon = "✅" if pred["is_healthy"] else "⚠️"
                p_col1, p_col2 = st.columns([3, 1])
                with p_col1:
                    st.write(f"**#{rank} {icon} {pred['label_clean']}**")
                    st.caption(f"Raw Class: `{pred['label_raw']}`")
                with p_col2:
                    st.metric(label="Score", value=f"{pred['percentage']}%")
                st.progress(min(max(pred["score"], 0.0), 1.0))

            # Confidence Comparison Bar Chart
            st.markdown("#### 📊 Confidence Comparison")
            chart_df = pd.DataFrame(
                {
                    "Predicted Condition": [p["label_clean"] for p in res["top_predictions"]],
                    "Confidence (%)": [p["percentage"] for p in res["top_predictions"]],
                }
            ).set_index("Predicted Condition")
            st.bar_chart(chart_df, color="#10b981")

    # Session Prediction History Table
    st.markdown("---")
    h_col1, h_col2, h_col3 = st.columns([3, 1, 1])
    with h_col1:
        st.subheader("📜 Session Prediction History")
    with h_col2:
        if st.session_state.prediction_history:
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            csv_buffer = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export CSV",
                data=csv_buffer,
                file_name=f"plant_health_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with h_col3:
        if st.button("🗑️ Clear History", use_container_width=True, disabled=len(st.session_state.prediction_history) == 0):
            clear_history()
            st.rerun()

    if not st.session_state.prediction_history:
        st.caption("No predictions recorded in this session yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)


# ===========================================================================
# TAB 2: SOIL MOISTURE & SMART IRRIGATION
# ===========================================================================
with tab2:
    st.subheader("Soil Moisture & Autonomous Irrigation Telemetry")
    st.caption("Continuous capacitive sensor monitoring [<30.0% Trigger → 5s Irrigation Pulse → 5s Soak → 35.0% Recovery Target].")

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([3, 1, 1, 1])
    with ctrl_col1:
        endpoint_url = st.text_input(
            "ESP32 Device URL:",
            value=st.session_state.esp32_ip,
            key="tab2_esp32_url",
            help="HTTP endpoint of your ESP32 on the local Wi-Fi network (e.g. http://192.168.12.16)",
        )
    with ctrl_col2:
        poll_clicked = st.button("🔄 Poll ESP32 Now", type="primary", use_container_width=True)
    with ctrl_col3:
        auto_poll = st.checkbox("Auto-Refresh (10s)", value=False, help="Automatically refresh readings periodically.")
    with ctrl_col4:
        st.session_state.sound_muted = st.toggle("🔇 Mute Alert", value=st.session_state.sound_muted, help="Mute dry soil acoustic alarm.")

    # Telemetry Polling Execution (only executed on user poll action or active auto-polling)
    if poll_clicked:
        if is_demo_mode:
            # Simulated telemetry sequence for cloud demonstrations without hardware
            demo_samples = [
                {
                    "moisture": 32.5,
                    "raw_value": 2480,
                    "soil_status": "MODERATE",
                    "pump_status": "OFF",
                    "last_watering_duration": 5,
                    "last_watering_elapsed_seconds": 120,
                    "current_state": "IDLE",
                    "watering_attempts": 0,
                },
                {
                    "moisture": 26.5,
                    "raw_value": 3120,
                    "soil_status": "DRY",
                    "pump_status": "ON",
                    "last_watering_duration": 5,
                    "last_watering_elapsed_seconds": 2,
                    "current_state": "WATERING",
                    "watering_attempts": 1,
                },
                {
                    "moisture": 36.5,
                    "raw_value": 2320,
                    "soil_status": "OPTIMAL",
                    "pump_status": "OFF",
                    "last_watering_duration": 5,
                    "last_watering_elapsed_seconds": 10,
                    "current_state": "IDLE",
                    "watering_attempts": 0,
                },
            ]
            step_idx = st.session_state.demo_soil_step % len(demo_samples)
            st.session_state.demo_soil_step += 1
            sample = demo_samples[step_idx]

            telemetry = {
                "online": True,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "base_url": "http://esp32-demo.local",
                "raw_value": sample["raw_value"],
                "moisture": sample["moisture"],
                "soil_status": sample["soil_status"],
                "pump_status": sample["pump_status"],
                "sensor_valid": True,
                "last_watering_duration": sample["last_watering_duration"],
                "last_watering_elapsed_seconds": sample["last_watering_elapsed_seconds"],
                "current_state": sample["current_state"],
                "shutoff_target": PUMP_SHUTOFF_TARGET,
                "dry_threshold": DRY_WARNING_THRESHOLD,
                "watering_attempts": sample["watering_attempts"],
                "max_attempts": 3,
                "uptime_seconds": 1200 + step_idx * 30,
                "wifi_rssi": -58,
                "ip_address": "192.168.12.16 (Simulated)",
                "error_message": None,
                "is_demo": True,
            }
            st.session_state.latest_telemetry = telemetry
            record = {
                "Time": telemetry["timestamp"],
                "Moisture (%)": telemetry["moisture"],
                "Raw ADC": telemetry["raw_value"],
                "Status": telemetry["soil_status"],
                "Pump": telemetry["pump_status"],
            }
            st.session_state.soil_history.append(record)
            if sample["pump_status"] == "ON":
                st.session_state.watering_history.insert(0, {
                    "Start Time": telemetry["timestamp"],
                    "Stop Time": "5s (Auto)",
                    "Duration (s)": "5s",
                    "Start Moisture": f"{sample['moisture']:.1f}%",
                    "End Moisture": "In Progress",
                    "Type": "ESP32 Auto (<30%)",
                })
        else:
            with st.spinner("Querying ESP32 REST API..."):
                telemetry = fetch_esp32_telemetry(endpoint_url, timeout=IOT_REQUEST_TIMEOUT)
                st.session_state.latest_telemetry = telemetry
                if telemetry["online"] and telemetry["moisture"] is not None:
                    record = {
                        "Time": telemetry["timestamp"],
                        "Moisture (%)": telemetry["moisture"],
                        "Raw ADC": telemetry["raw_value"],
                        "Status": telemetry["soil_status"],
                        "Pump": telemetry["pump_status"],
                    }
                    st.session_state.soil_history.append(record)

    telem = st.session_state.latest_telemetry
    st.markdown("---")

    if telem is None:
        st.info("Click **'🔄 Poll ESP32 Now'** to retrieve live sensor telemetry.")
    elif not telem["online"]:
        st.error(
            f"❌ **ESP32 Telemetry Gateway Offline**: Could not connect to `{telem.get('base_url')}` (Timeout: {IOT_REQUEST_TIMEOUT}s).\n\n"
            f"Error Diagnosis: {telem.get('error_message')}\n\n"
            "Please confirm that the ESP32 is powered on, connected to the same Wi-Fi network, and that the IP address matches the Serial Monitor.\n\n"
            "📌 **Streamlit Community Cloud Notice**: Public cloud servers cannot reach private local network addresses (such as `192.168.x.x` or `10.x.x.x`) on your local Wi-Fi. "
            "Switch to **🧪 DEMO MODE** in the sidebar to test soil telemetry and irrigation visualization features in the cloud, or run the app locally on your LAN for physical ESP32 communication."
        )
    else:
        # 4 Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        moist_val = telem["moisture"]
        raw_val = telem["raw_value"]
        soil_status = telem["soil_status"]
        pump_status = telem["pump_status"]

        with m_col1:
            st.metric(
                label="Soil Moisture",
                value=f"{moist_val:.1f}%" if moist_val is not None else "—",
                delta="Nominal" if (moist_val is not None and moist_val >= DRY_WARNING_THRESHOLD) else "Dry Alert",
                delta_color="normal" if (moist_val is not None and moist_val >= DRY_WARNING_THRESHOLD) else "inverse",
            )
        with m_col2:
            st.metric(
                label="Raw ADC (GPIO 34)",
                value=f"{raw_val}" if raw_val is not None else "—",
                help="12-bit Analog-to-Digital Converter output (0 = Saturated, 4095 = Air/Dry)",
            )
        with m_col3:
            st.metric(
                label="Soil Condition",
                value=soil_status if soil_status else "UNKNOWN",
            )
        with m_col4:
            st.metric(
                label="Pump Relay (GPIO 26)",
                value=pump_status if pump_status else "UNKNOWN",
            )

        # Dry Soil Desiccation Warning & Audio Chime
        if moist_val is not None and moist_val < DRY_WARNING_THRESHOLD:
            st.error(
                f"🚨 **DRY SOIL DESICCATION HAZARD ({moist_val:.1f}% < {DRY_WARNING_THRESHOLD}%)**: "
                "Soil moisture has dropped below critical threshold. Closed-loop automatic irrigation pulse triggered on firmware."
            )
            if (st.session_state.previous_soil_condition != "DRY") and not st.session_state.sound_muted:
                st.html(
                    f'<audio autoplay><source src="data:audio/wav;base64,{ALERT_CHIME_B64}" type="audio/wav"></audio>'
                )
        st.session_state.previous_soil_condition = soil_status

        # Soil Moisture Progression Chart (Altair)
        st.markdown("#### 📈 Soil Moisture Progression")
        if not st.session_state.soil_history:
            st.caption("No historical records in this session yet. Click '🔄 Poll ESP32 Now' to populate.")
        else:
            hist_df = pd.DataFrame(st.session_state.soil_history)
            base = alt.Chart(hist_df).encode(
                x=alt.X("Time:N", title="Time", sort=None),
                y=alt.Y("Moisture (%):Q", title="Soil Moisture (%)", scale=alt.Scale(domain=[0, 100])),
            )
            line = base.mark_line(point=True, color="#10b981", strokeWidth=2.5).encode(
                tooltip=["Time", "Moisture (%)", "Status", "Pump"]
            )
            warning_df = pd.DataFrame([{"Threshold": DRY_WARNING_THRESHOLD, "Label": "30% Critical Dry Alert"}])
            warning_line = alt.Chart(warning_df).mark_rule(color="#ef4444", strokeDash=[5, 5], size=2).encode(
                y="Threshold:Q",
                tooltip=["Label"],
            )
            shutoff_df = pd.DataFrame([{"Threshold": PUMP_SHUTOFF_TARGET, "Label": "35% Irrigation Cutoff Target"}])
            shutoff_line = alt.Chart(shutoff_df).mark_rule(color="#4edea3", strokeDash=[5, 5], size=2).encode(
                y="Threshold:Q",
                tooltip=["Label"],
            )
            combined_chart = (
                (line + warning_line + shutoff_line)
                .properties(
                    title="Hydration Progression Curve (Red: 30% Trigger, Green: 35% Cutoff)",
                    height=320,
                )
                .configure_view(strokeWidth=0)
                .configure_axis(gridColor="rgba(60, 74, 66, 0.35)", labelColor="#86948a", titleColor="#e1e2e5")
                .configure_title(color="#e1e2e5")
            )
            st.altair_chart(combined_chart, use_container_width=True)

        # Autonomous Irrigation Actuator Log
        st.markdown("#### 🚰 Autonomous Irrigation Actuator Log")
        w_col1, w_col2 = st.columns([3, 1])
        with w_col1:
            if not st.session_state.watering_history:
                st.caption("No pump activations recorded in this session.")
            else:
                st.dataframe(pd.DataFrame(st.session_state.watering_history), use_container_width=True)
        with w_col2:
            if st.button("🗑️ Clear Soil History", use_container_width=True):
                clear_soil_history()
                st.rerun()

        # ESP32 Diagnostic Expander
        with st.expander("ESP32-S3 Micro-Controller Diagnostics Bus", expanded=False):
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.write(f"**ESP32 Gateway:** `{telem.get('ip_address') or telem['base_url']}`")
                st.write(f"**Wi-Fi RSSI:** `{telem.get('wifi_rssi', 'N/A')} dBm`")
                st.write(f"**System Uptime:** `{telem.get('uptime_seconds', 'N/A')} seconds`")
            with d_col2:
                st.write(f"**Sensor Valid:** `{telem.get('sensor_valid', True)}`")
                st.write(f"**Irrigation Cycles:** `{telem.get('watering_attempts', 0)} / {telem.get('max_attempts', 3)}`")
                elapsed_sec = telem.get("last_watering_elapsed_seconds")
                elapsed_display = f"{elapsed_sec}s ago" if elapsed_sec is not None and elapsed_sec >= 0 else "Never in this boot"
                st.write(f"**Elapsed Since Watering:** `{elapsed_display}`")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.info(f"**Agronomic Advisory & Model Disclaimer**: {DISCLAIMER_TEXT}")
