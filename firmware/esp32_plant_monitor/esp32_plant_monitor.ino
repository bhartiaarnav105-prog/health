/*
 * =====================================================================================
 * Project: AIoT Plant Health Monitoring System - Soil Moisture, Irrigation & Wi-Fi API
 * Phase: Phase 4 - Streamlit Soil Monitoring & Autonomous ESP32 Irrigation (<30% -> 5s -> 5s soak -> 35%)
 * Board: ESP32 DevKit
 * 
 * Hardware Connections:
 *   - Capacitive Soil Sensor: Analog OUT -> GPIO 34 (ADC1_CH6, 3.3V recommended)
 *   - Relay Module Signal:    IN         -> GPIO 26
 *   - 5V DC Water Pump:       V+ through Relay COM and NO terminals
 *   - Power:                  External 5V supply for Pump, Common GND with ESP32
 * 
 * Autonomous Irrigation Logic:
 *   - Automatic Trigger: Starts pump when soil moisture < 30.0%.
 *   - Duration: Runs pump for 5 seconds.
 *   - Soak Cooldown: 5 seconds soak pause before taking a fresh sensor reading.
 *   - Repeat Cycle: If still < 30.0%, repeats up to 3 consecutive attempts.
 *   - Automatic Shutoff Target: Stops immediately and clears cycle once moisture reaches 35.0%.
 *   - Safety Lockout: After 3 attempts without recovery or on sensor anomaly, pump locks OFF.
 *   - Non-blocking Wi-Fi & Web Server on port 80 (GET /sensor, GET /status).
 * =====================================================================================
 */

#include <WiFi.h>
#include <WebServer.h>
#include <esp_system.h>

// =====================================================================================
// 1. WI-FI CREDENTIALS & SERVER CONFIGURATION
// =====================================================================================
const char* WIFI_SSID     = "GUESTHOUSE";
const char* WIFI_PASSWORD = "Guest#26";

// Non-blocking Wi-Fi reconnect timer
const unsigned long WIFI_RECONNECT_INTERVAL_MS = 15000; // Check every 15s if disconnected
unsigned long previousWifiCheckMillis = 0;
bool wasWifiConnected = false;
bool isServerRunning = false;

// Embedded HTTP Web Server on standard port 80
WebServer server(80);


// =====================================================================================
// 2. PIN DEFINITIONS & HARDWARE CONFIGURATION
// =====================================================================================
const int SOIL_SENSOR_PIN = 34; // Analog input from capacitive soil sensor (ADC1)
const int RELAY_PIN       = 26; // Digital output to relay module controlling pump

// RELAY ACTIVE LOGIC:
// Most 5V single-channel relay modules are Active-LOW (pulling pin LOW triggers relay).
// Set to false if your relay module is Active-HIGH.
const bool RELAY_ACTIVE_LOW = true;


// =====================================================================================
// 3. USER CONFIGURABLE CALIBRATION CONSTANTS
// =====================================================================================
int DRY_VALUE = 3200;  // Measured raw ADC in completely dry soil
int WET_VALUE = 1500;  // Measured raw ADC in saturated/watered soil

// Sensor hardware sanity boundaries (12-bit ADC range: 0 - 4095)
const int MIN_VALID_ADC = 100;
const int MAX_VALID_ADC = 4050;


// =====================================================================================
// 4. IRRIGATION & SHUTOFF SETTINGS
// =====================================================================================
const float DRY_THRESHOLD = 30.0;             // Automatically starts pump below 30.0%
const float SHUTOFF_THRESHOLD_PERCENT = 35.0; // Automatic shutoff target at 35.0%
const float HYSTERESIS_PERCENT = 5.0;          // Deadband between trigger and shutoff

const unsigned long PUMP_DURATION_MS = 5000;            // Runs pump for 5 seconds
const unsigned long COOLDOWN_DURATION_MS = 5000;        // 5 seconds soak cooldown before re-reading
const int MAX_WATERING_ATTEMPTS = 3;                    // Max consecutive cycles before safety lockout

const unsigned long SAMPLING_INTERVAL_MS = 1000; // 1 second reading interval
const int NUM_SAMPLES = 5;                       // 5 samples per cycle
const int SAMPLE_DELAY_MICROS = 400;             // 400us between samples (~2ms total)


// =====================================================================================
// 5. IRRIGATION STATE MACHINE DEFINITION
// =====================================================================================
enum IrrigationState {
  STATE_IDLE,      // Normal soil monitoring; moisture >= 30.0%
  STATE_WATERING,  // Pump actively running (5s duration)
  STATE_COOLDOWN,  // Pump OFF; 5s soaking period before re-sampling
  STATE_LOCKOUT    // Max attempts exceeded (3) or sensor fault; pump locked OFF
};

IrrigationState currentState = STATE_IDLE;


// =====================================================================================
// 6. GLOBAL VARIABLES
// =====================================================================================
unsigned long previousSampleMillis = 0;
unsigned long pumpStartMillis = 0;
unsigned long cooldownStartMillis = 0;
unsigned long lastWateringFinishMillis = 0; // Timestamp when last cycle finished
unsigned long lastWateringDurationActualSec = 0;

int wateringAttemptCount = 0;
int lastRawADC = 0;
float lastMoisturePercent = 0.0;
String lastCondition = "INITIALIZING";
bool isSensorValid = false;
bool isPumpRunning = false;


// =====================================================================================
// 7. HARDWARE HELPER FUNCTIONS
// =====================================================================================

void setPump(bool activate) {
  if (activate) {
    digitalWrite(RELAY_PIN, RELAY_ACTIVE_LOW ? LOW : HIGH);
    isPumpRunning = true;
  } else {
    digitalWrite(RELAY_PIN, RELAY_ACTIVE_LOW ? HIGH : LOW);
    if (isPumpRunning && pumpStartMillis > 0) {
      lastWateringDurationActualSec = (millis() - pumpStartMillis) / 1000;
    }
    isPumpRunning = false;
  }
}

/**
 * Fast, non-blocking 5-sample ADC read using microsecond spacing (~2ms total).
 */
int readAveragedADC(int pin, int samples) {
  long sum = 0;
  for (int i = 0; i < samples; i++) {
    sum += analogRead(pin);
    delayMicroseconds(SAMPLE_DELAY_MICROS);
  }
  return (int)(sum / samples);
}

float calculateMoisturePercent(int rawADC, int dryVal, int wetVal) {
  if (dryVal == wetVal) return 0.0;

  float percent = ((float)(dryVal - rawADC) / (float)(dryVal - wetVal)) * 100.0;
  if (percent < 0.0) percent = 0.0;
  if (percent > 100.0) percent = 100.0;
  return percent;
}

String getSoilCondition(float percent) {
  if (percent < DRY_THRESHOLD) {
    return "DRY";
  } else if (percent > (DRY_THRESHOLD + 40.0)) {
    return "WET";
  } else {
    return "MODERATE";
  }
}

String getStateString(IrrigationState state) {
  switch (state) {
    case STATE_IDLE:     return "IDLE";
    case STATE_WATERING: return "WATERING";
    case STATE_COOLDOWN: return "COOLDOWN";
    case STATE_LOCKOUT:  return "LOCKOUT";
    default:             return "UNKNOWN";
  }
}

const char* getResetReasonString(esp_reset_reason_t reason) {
  switch (reason) {
    case ESP_RST_POWERON:   return "Power-on Event";
    case ESP_RST_EXT:       return "External Pin Reset";
    case ESP_RST_SW:        return "Software Reset (esp_restart)";
    case ESP_RST_PANIC:     return "Software Exception / Panic";
    case ESP_RST_INT_WDT:   return "Interrupt Watchdog Reset";
    case ESP_RST_TASK_WDT:  return "Task Watchdog Reset";
    case ESP_RST_WDT:       return "Other Watchdog Reset";
    case ESP_RST_DEEPSLEEP: return "Deep-Sleep Wakeup";
    case ESP_RST_BROWNOUT:  return "Brownout Reset (Voltage Dip)";
    case ESP_RST_SDIO:      return "SDIO Reset";
    default:                return "General Reset";
  }
}


// =====================================================================================
// 8. HTTP REQUEST HANDLERS (REST API)
// =====================================================================================

/**
 * GET /sensor
 * Returns live soil telemetry, pump status, and shutoff metrics in JSON.
 */
void handleGetSensor() {
  long lastWateringElapsedSeconds = -1;
  if (lastWateringFinishMillis > 0) {
    lastWateringElapsedSeconds = (long)((millis() - lastWateringFinishMillis) / 1000);
  }

  String json;
  json.reserve(360);
  json = "{";
  json += "\"raw_value\":" + String(lastRawADC) + ",";
  json += "\"moisture\":" + String(lastMoisturePercent, 1) + ",";
  json += "\"soil_status\":\"" + lastCondition + "\",";
  json += "\"pump_status\":\"" + String(isPumpRunning ? "ON" : "OFF") + "\",";
  json += "\"sensor_valid\":" + String(isSensorValid ? "true" : "false") + ",";
  json += "\"last_watering_duration\":" + String(lastWateringDurationActualSec > 0 ? lastWateringDurationActualSec : (PUMP_DURATION_MS / 1000)) + ",";
  json += "\"last_watering_elapsed_seconds\":" + String(lastWateringElapsedSeconds) + ",";
  json += "\"current_state\":\"" + getStateString(currentState) + "\",";
  json += "\"shutoff_target\":" + String(SHUTOFF_THRESHOLD_PERCENT, 1) + ",";
  json += "\"dry_threshold\":" + String(DRY_THRESHOLD, 1) + ",";
  json += "\"watering_attempts\":" + String(wateringAttemptCount) + ",";
  json += "\"max_attempts\":" + String(MAX_WATERING_ATTEMPTS);
  json += "}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

/**
 * GET /status
 * Returns system health, Wi-Fi connectivity, local IP, and irrigation state.
 */
void handleGetStatus() {
  unsigned long uptimeSeconds = millis() / 1000;
  bool isConnected = (WiFi.status() == WL_CONNECTED);
  String ipStr = isConnected ? WiFi.localIP().toString() : "DISCONNECTED";

  String json;
  json.reserve(300);
  json = "{";
  json += "\"system_status\":\"ONLINE\",";
  json += "\"wifi_connected\":" + String(isConnected ? "true" : "false") + ",";
  json += "\"ip_address\":\"" + ipStr + "\",";
  json += "\"sensor_valid\":" + String(isSensorValid ? "true" : "false") + ",";
  json += "\"current_state\":\"" + getStateString(currentState) + "\",";
  json += "\"pump_status\":\"" + String(isPumpRunning ? "ON" : "OFF") + "\",";
  json += "\"uptime_seconds\":" + String(uptimeSeconds) + ",";
  json += "\"wifi_rssi\":" + String(isConnected ? WiFi.RSSI() : 0) + ",";
  json += "\"free_heap_bytes\":" + String(ESP.getFreeHeap());
  json += "}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

/**
 * 404 Handler for invalid routes.
 */
void handleNotFound() {
  String json = "{\"error\":\"Endpoint not found\",\"valid_endpoints\":[\"/sensor\",\"/status\"]}";
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(404, "application/json", json);
}


// =====================================================================================
// 9. SETUP
// =====================================================================================
void setup() {
  // 1. BOOT SAFETY FIRST: Force relay pin OFF before declaring output
  digitalWrite(RELAY_PIN, RELAY_ACTIVE_LOW ? HIGH : LOW);
  pinMode(RELAY_PIN, OUTPUT);
  setPump(false);

  // 2. Initialize Serial Monitor
  Serial.begin(115200);

  // 3. Boot Diagnostics
  esp_reset_reason_t reason = esp_reset_reason();
  Serial.println();
  Serial.println("==========================================================");
  Serial.println("  AIoT Plant Health - ESP32 Auto-Irrigation & REST API    ");
  Serial.print("  Hardware Reset Reason: ");
  Serial.print(getResetReasonString(reason));
  Serial.print(" (Code: 0x");
  Serial.print((int)reason, HEX);
  Serial.println(")");
  Serial.println("==========================================================");

  // 4. Configure ADC
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  // 5. Initial Sensor Reading
  lastRawADC = readAveragedADC(SOIL_SENSOR_PIN, NUM_SAMPLES);
  if (lastRawADC >= MIN_VALID_ADC && lastRawADC <= MAX_VALID_ADC) {
    isSensorValid = true;
    lastMoisturePercent = calculateMoisturePercent(lastRawADC, DRY_VALUE, WET_VALUE);
    lastCondition = getSoilCondition(lastMoisturePercent);
  } else {
    isSensorValid = false;
    lastCondition = "INITIALIZING";
  }

  // 6. Register HTTP Routes
  server.on("/sensor", HTTP_GET, handleGetSensor);
  server.on("/status", HTTP_GET, handleGetStatus);
  server.onNotFound(handleNotFound);

  // 7. Initiate Background Wi-Fi (Non-blocking)
  Serial.print("[Wi-Fi] Starting background connection to SSID: ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.println("[System] Sensor monitoring and automatic irrigation ACTIVE.\n");
}


// =====================================================================================
// 10. MAIN LOOP (4 Non-Blocking Tasks with 35% Shutoff Safety)
// =====================================================================================
void loop() {
  unsigned long currentMillis = millis();

  // -----------------------------------------------------------------------------------
  // TASK 1: WI-FI STATE TRACKER & SAFE SERVER LIFECYCLE
  // -----------------------------------------------------------------------------------
  bool isConnectedNow = (WiFi.status() == WL_CONNECTED);

  if (isConnectedNow) {
    if (!wasWifiConnected) {
      wasWifiConnected = true;

      if (!isServerRunning) {
        server.begin();
        isServerRunning = true;
      }

      Serial.println("\n==========================================================");
      Serial.println("  [Wi-Fi] Connected Successfully in Background!           ");
      Serial.print("  ESP32 Local IP Address: http://");
      Serial.println(WiFi.localIP());
      Serial.println("  Available Endpoints:                                    ");
      Serial.print("    -> Telemetry:    http://"); Serial.print(WiFi.localIP()); Serial.println("/sensor");
      Serial.print("    -> Status:       http://"); Serial.print(WiFi.localIP()); Serial.println("/status");
      Serial.println("==========================================================\n");
    }
  } else {
    if (wasWifiConnected) {
      wasWifiConnected = false;
      Serial.println("\n[Wi-Fi] Connection lost! Irrigation continues offline.");
    }

    if (currentMillis - previousWifiCheckMillis >= WIFI_RECONNECT_INTERVAL_MS) {
      previousWifiCheckMillis = currentMillis;
      Serial.println("[Wi-Fi] Background reconnect attempt...");
      WiFi.reconnect();
    }
  }

  // -----------------------------------------------------------------------------------
  // TASK 2: SERVICE INCOMING HTTP REQUESTS
  // -----------------------------------------------------------------------------------
  if (isServerRunning && isConnectedNow) {
    server.handleClient();
  }

  // -----------------------------------------------------------------------------------
  // TASK 3: NON-BLOCKING SENSOR SAMPLING (Runs every 1000ms)
  // -----------------------------------------------------------------------------------
  if (currentMillis - previousSampleMillis >= SAMPLING_INTERVAL_MS) {
    previousSampleMillis = currentMillis;

    lastRawADC = readAveragedADC(SOIL_SENSOR_PIN, NUM_SAMPLES);

    if (lastRawADC < MIN_VALID_ADC || lastRawADC > MAX_VALID_ADC) {
      isSensorValid = false;
      lastCondition = "OUT_OF_BOUNDS";
      lastMoisturePercent = 0.0;

      // Emergency Fail-Safe: Force pump OFF immediately on sensor anomaly
      if (isPumpRunning) {
        setPump(false);
        Serial.println("[CRITICAL] Sensor out-of-bounds while pump active! Pump forced OFF.");
      }
      currentState = STATE_LOCKOUT;
      Serial.print("[WARNING] Abnormal ADC Reading: ");
      Serial.println(lastRawADC);
    } else {
      isSensorValid = true;
      lastMoisturePercent = calculateMoisturePercent(lastRawADC, DRY_VALUE, WET_VALUE);
      lastCondition = getSoilCondition(lastMoisturePercent);

      Serial.print("[Sensor] Raw: ");
      Serial.print(lastRawADC);
      Serial.print(" | Moisture: ");
      Serial.print(lastMoisturePercent, 1);
      Serial.print("% (");
      Serial.print(lastCondition);
      Serial.print(") | State: ");
      Serial.print(getStateString(currentState));
      Serial.print(" | Pump: ");
      Serial.print(isPumpRunning ? "ON" : "OFF");
      Serial.print(" | Wi-Fi: ");
      Serial.println(isConnectedNow ? "Connected" : "Connecting...");
    }
  }

  // -----------------------------------------------------------------------------------
  // TASK 4: IRRIGATION STATE MACHINE (<30% -> 5s PUMP -> 5s SOAK -> 35% SHUTOFF)
  // -----------------------------------------------------------------------------------
  switch (currentState) {

    case STATE_IDLE:
      // Auto-watering triggered when moisture drops below 30%
      if (isSensorValid && (lastMoisturePercent < DRY_THRESHOLD)) {
        Serial.println("\n>>> [IRRIGATION TRIGGER] Moisture below 30.0% dry threshold!");
        Serial.println(">>> Starting pump for 5 seconds...");
        setPump(true);
        pumpStartMillis = currentMillis;
        currentState = STATE_WATERING;
      }
      break;

    case STATE_WATERING:
      // 1. AUTOMATIC SHUTOFF: Stop pump immediately when soil reaches 35.0% target
      if (isSensorValid && (lastMoisturePercent >= SHUTOFF_THRESHOLD_PERCENT)) {
        setPump(false);
        wateringAttemptCount = 0;
        lastWateringFinishMillis = currentMillis;

        Serial.println("\n>>> [AUTO-SHUTOFF] Target 35.0% moisture reached! Pump stopped early.");
        Serial.print(">>> Entering soak cooldown for ");
        Serial.print(COOLDOWN_DURATION_MS / 1000);
        Serial.println(" seconds.\n");

        cooldownStartMillis = currentMillis;
        currentState = STATE_COOLDOWN;
        break;
      }

      // 2. TIMEOUT SHUTOFF: Run pump for 5 seconds, then turn it OFF
      if (currentMillis - pumpStartMillis >= PUMP_DURATION_MS) {
        setPump(false);
        wateringAttemptCount++;
        lastWateringFinishMillis = currentMillis;

        Serial.println("\n>>> [PUMP TIMEOUT] 5 seconds watering cycle complete. Pump turned OFF.");
        Serial.print(">>> Entering soak cooldown for ");
        Serial.print(COOLDOWN_DURATION_MS / 1000);
        Serial.println(" seconds before re-reading...\n");

        cooldownStartMillis = currentMillis;
        currentState = STATE_COOLDOWN;
      }
      break;

    case STATE_COOLDOWN:
      // Wait for 5-second soak cooldown
      if (currentMillis - cooldownStartMillis >= COOLDOWN_DURATION_MS) {
        Serial.println("\n>>> [COOLDOWN COMPLETE] Reading fresh soil moisture after 5s soak...");

        // Re-read actual soil moisture
        lastRawADC = readAveragedADC(SOIL_SENSOR_PIN, NUM_SAMPLES);
        if (lastRawADC >= MIN_VALID_ADC && lastRawADC <= MAX_VALID_ADC) {
          isSensorValid = true;
          lastMoisturePercent = calculateMoisturePercent(lastRawADC, DRY_VALUE, WET_VALUE);
          lastCondition = getSoilCondition(lastMoisturePercent);
        } else {
          isSensorValid = false;
        }

        Serial.print(">>> Measured Moisture: ");
        Serial.print(lastMoisturePercent, 1);
        Serial.print("% (Raw ADC: ");
        Serial.print(lastRawADC);
        Serial.println(")");

        // Stop watering when moisture reaches 35%
        if (isSensorValid && (lastMoisturePercent >= SHUTOFF_THRESHOLD_PERCENT)) {
          Serial.println(">>> [SUCCESS] Soil moisture restored to 35%+ target. Returning to IDLE.\n");
          wateringAttemptCount = 0;
          currentState = STATE_IDLE;
        } 
        // Repeat watering if moisture is still below 30%
        else if (isSensorValid && (lastMoisturePercent < DRY_THRESHOLD)) {
          Serial.print(">>> [ATTENTION] Soil still below 30% dry threshold (Attempt ");
          Serial.print(wateringAttemptCount);
          Serial.print(" of ");
          Serial.print(MAX_WATERING_ATTEMPTS);
          Serial.println(").");

          // Keep maximum watering attempts at 3
          if (wateringAttemptCount < MAX_WATERING_ATTEMPTS) {
            Serial.println(">>> Initiating next 5-second watering cycle...\n");
            setPump(true);
            pumpStartMillis = currentMillis;
            currentState = STATE_WATERING;
          } else {
            Serial.println("\n>>> [LOCKOUT ALERT] Max 3 watering attempts reached! Pump locked OFF for safety.\n");
            setPump(false);
            currentState = STATE_LOCKOUT;
          }
        } 
        else {
          // Moisture in acceptable range (>=30% and <35%)
          Serial.println(">>> [INFO] Soil moisture in acceptable range (>=30%). Returning to IDLE.\n");
          wateringAttemptCount = 0;
          currentState = STATE_IDLE;
        }
      }
      break;

    case STATE_LOCKOUT:
      setPump(false);
      // Auto-recovery when moisture restored to 35%+
      if (isSensorValid && (lastMoisturePercent >= SHUTOFF_THRESHOLD_PERCENT)) {
        Serial.println(">>> [AUTO-RECOVERY] Soil moisture detected at 35.0%+. Resetting lockout.\n");
        wateringAttemptCount = 0;
        currentState = STATE_IDLE;
      }
      break;
  }
}
