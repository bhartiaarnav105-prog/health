# AIoT Plant Health Monitoring System

An integrated agricultural monitoring and smart irrigation platform combining:
1. **Plant Leaf Disease Screening**: A lightweight pretrained `MobileNetV2` deep convolutional neural network fine-tuned on the PlantVillage dataset (38 classes, 14 crops).
2. **IoT Soil Moisture & Smart Irrigation**: Real-time telemetry and bi-directional pump relay control via an ESP32 microcontroller with a closed-loop **35.0% moisture auto-shutoff**.

---

## 🌟 Key Features

### 🌿 Plant Disease AI (Tab 1)
- **Manual Leaf Image Upload**: Accepts JPG, JPEG, and PNG formats (with automatic RGBA transparency handling).
- **Edge-Ready CPU Inference**: Runs locally via PyTorch with ~30–70 ms inference latency.
- **Single-Load Model Caching**: Uses `@st.cache_resource` so the ~14 MB model loads only once.
- **Top-3 Differential Predictions**: Displays candidate diagnoses with normalized confidence percentages and progress bars.
- **Low-Confidence Warning (< 60%)**: Protects against uncertain or out-of-distribution inputs.
- **Session Prediction History**: Tracks evaluations with one-click export to CSV.

### 💧 Soil Moisture & Smart Irrigation (Tab 2)
- **Real-Time Telemetry over Wi-Fi**: Periodically queries ESP32 `GET /sensor` and `GET /status`.
- **4 Live Metric Cards**: Displays Soil Moisture %, Raw ADC (0–4095 on GPIO 34), Soil Condition (`DRY`, `MODERATE`, `WET`), and Pump Relay Status (`ON`/`OFF`).
- **Bi-Directional Remote Pump Control**:
  - 🟢 **SWITCH PUMP ON**: Sends `POST /pump?action=ON`. Requires soil moisture $< 35\%$ and sensor health.
  - 🔴 **SWITCH PUMP OFF**: Sends `POST /pump?action=OFF`. Instantly de-energizes the relay.
- **Hardware-Enforced 35% Auto-Shutoff**: The ESP32 firmware continuously monitors moisture and automatically turns the pump OFF as soon as soil reaches **35.0%**, or after a hard safety timeout (15s max).
- **Dry Soil Warning & Sound Alert (< 30%)**:
  - Visual alert banner when moisture drops below 30%.
  - Self-contained HTML5 audio chime alert with a **Mute/Unmute** toggle.
  - Plays only on state transition into DRY (avoiding repeated sound on every refresh).
- **Interactive Altair Moisture Chart**: Displays real moisture readings over time with:
  - 🔴 **30% Red Dashed Line**: Dry Soil Warning Threshold
  - 🟢 **35% Green Dashed Line**: Pump Auto-Shutoff Target
- **Watering Event Log**: Records pump activations, duration, and moisture before and after irrigation.
- **Truthful Offline Isolation**: Never fabricates data; displays `OFFLINE` status when the board is unreachable instead of false 0% moisture.

---

## 📁 Project Structure

```text
plant-monitor/ (c:\Users\bhart\OneDrive\Desktop\helath)
│
├── README.md                      # Comprehensive user, architecture, and wiring guide
├── requirements.txt               # Pinned minimal Python dependencies
├── .gitignore                     # Git ignore rules for caches and artifacts
│
├── firmware/
│   └── esp32_plant_monitor/
│       └── esp32_plant_monitor.ino# ESP32 firmware: ADC, Relay, Wi-Fi REST API, 35% Shutoff
│
├── app/
│   ├── __init__.py                # Package declaration
│   ├── config.py                  # Thresholds, timeouts, and label utilities
│   ├── iot_client.py              # ESP32 REST client (telemetry & POST /pump)
│   ├── model_loader.py            # Pretrained MobileNetV2 loader & CPU inference
│   └── dashboard.py               # Streamlit web dashboard (Tab 1: AI, Tab 2: IoT)
│
├── tools/
│   └── smoke_test_model.py        # Terminal smoke test for AI inference
│
├── tests/
│   ├── test_iot_client.py         # Unit tests for IoT client & pump commands
│   ├── test_model_loader.py       # Core unit tests for preprocessing & predictions
│   ├── test_dashboard_ui.py       # Headless Streamlit AppTest UI test suite
│   └── test_phase3_reliability.py # 12-case reliability and edge-case test suite
│
└── samples/                       # Test assets (JPG, JPEG, PNG, corrupted, large, etc.)
```

---

## 🔌 Hardware Wiring Schematic

```
                      +-------------------+
                      |   ESP32 DevKit    |
                      |                   |
                      |           GPIO 34 |<--- [Analog OUT] Capacitive Soil Sensor
                      |           GPIO 26 |---> [IN / Signal] Relay Module
                      |               GND |---+
                      +-------------------+   |
                                              |
                      +-------------------+   |
                      |    Relay Module   |   |
                      |                   |   |
                      |               VCC |<--+-- [+5V] External 5V Power Supply
                      |               GND |<--+-- [GND] External 5V Power Supply
                      |                IN |<===== [GPIO 26]
                      |                   |
                      |          COM (in) |<----- [+5V] External 5V Power Supply
                      |         NO (out)  |-----> [+] 5V DC Water Pump
                      +-------------------+
                                                          |
                      +-------------------+               |
                      | 5V DC Water Pump  |               |
                      |               [-] |<--------------+-- [GND] External 5V Power Supply
                      +-------------------+
```

### Safety Rules:
1. **Never power the pump motor from ESP32 pins**: Motor stall currents can destroy the ESP32 regulator. Always use an external 5V power supply (1A+).
2. **Common Ground**: Tie the ESP32 `GND`, Relay `GND`, and external 5V power supply `GND` together.
3. **Sensor Power**: Power the capacitive sensor from the ESP32 **3.3V** pin so its analog voltage remains strictly within the 0V–3.3V ADC limit.

---

## 🚀 Running the System

### 1. Upload Firmware to ESP32
1. Open `firmware/esp32_plant_monitor/esp32_plant_monitor.ino` in Arduino IDE.
2. Verify Wi-Fi credentials at lines 33–34:
   ```cpp
   const char* WIFI_SSID     = "GUESTHOUSE";
   const char* WIFI_PASSWORD = "Guest#26";
   ```
3. Set **Tools $\rightarrow$ Board** to `ESP32 Dev Module` and select your COM port.
4. Click **Upload** (`Ctrl + U`).
5. Open Serial Monitor at **115200 baud** to see the assigned IP address (e.g. `http://192.168.12.16`).

### 2. Launch the Streamlit Dashboard
```powershell
streamlit run app/dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser:
- **Tab 1**: Upload plant leaf photos for disease screening.
- **Tab 2**: Monitor live soil moisture, receive audio alerts, view the Altair chart with 30%/35% target lines, and manually switch the pump ON/OFF.

### 3. Run Automated Tests
```powershell
python -m unittest discover tests
```

---

## ☁️ Deploying to Streamlit Community Cloud

Follow these steps to deploy this project to [Streamlit Community Cloud](https://share.streamlit.io):

### Deployment Flow
```text
GitHub repository
        ↓
Streamlit Community Cloud
        ↓
app/dashboard.py
        ↓
public Streamlit URL
```

### Step-by-Step Instructions
1. Push your project to a GitHub repository (e.g., `your-username/helath`).
2. Log into [Streamlit Community Cloud](https://share.streamlit.io).
3. Click **New app** and select your repository and branch.
4. Set the **Main file path** to:
   ```text
   app/dashboard.py
   ```
5. *(Optional)* Under **Advanced settings → Secrets**, you can provide an optional Hugging Face access token to increase model download rate limits:
   ```toml
   HF_TOKEN = "hf_your_personal_access_token_here"
   ```
6. Click **Deploy!**

### ⚠️ Hardware Communication Notice (ESP32 in the Cloud)
- **Private Network Limitation**: Streamlit Community Cloud operates from public cloud servers. Public cloud containers **cannot directly reach private local network IP addresses** (such as `http://192.168.12.16` or `10.x.x.x`) on your home or greenhouse Wi-Fi.
- **Using the Cloud Deployment**:
  - **🌿 Plant Disease AI (Tab 1)**: Fully functional in the cloud. Leaf photos can be uploaded and analyzed by the PyTorch model with full confidence metrics and CSV history export.
  - **💧 Soil Moisture & Irrigation (Tab 2)**: Switch to **🧪 DEMO MODE** in the sidebar to simulate live capacitive sensor readings, 30% desiccation hazard alerts, acoustic warnings, Altair threshold graphs, and actuator event logs.
  - **Physical ESP32 Live Telemetry**: To communicate directly with your physical ESP32 microcontroller, run the dashboard locally on the same Wi-Fi network (`streamlit run app/dashboard.py`).

---

## ⚠️ Advisory & Disclaimer

> "This model was trained on PlantVillage lab-style images. It may perform poorly on real-world photographs. Predictions are suggestions, not confirmed diagnoses. Results should be verified by an agricultural expert."
