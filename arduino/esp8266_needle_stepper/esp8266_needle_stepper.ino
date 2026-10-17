/*
 * FaceX ESP8266 Needle Stepper Controller
 * 
 * Receives MQTT messages from face_tracking.py and moves a gauge needle
 * to mirror the locked person's horizontal position.
 * 
 * Hardware:
 *   - ESP8266 (NodeMCU or compatible)
 *   - 28BYJ-48 stepper motor (bipolar-hacked: red wire disconnected)
 *   - L298N H-bridge motor driver
 * 
 * Wiring:
 *   D1 (GPIO5)  -> L298N IN1 -> Blue wire
 *   D2 (GPIO4)  -> L298N IN2 -> Pink wire
 *   D5 (GPIO14) -> L298N IN3 -> Yellow wire
 *   D6 (GPIO12) -> L298N IN4 -> Orange wire
 *   ESP8266 GND -> L298N GND (common ground with motor supply)
 *   Motor supply: 5V (for most 28BYJ-48) to L298N power input
 * 
 * MQTT Payload (from src/mqtt_bridge.py):
 *   {"state":"LOCKED","locked":true,"horizontal":"LEFT","error_x":-0.4,"angle":54}
 * 
 * Angle Mapping:
 *   error_x = -1.0 (far LEFT)  -> angle = 0°
 *   error_x =  0.0 (CENTER)    -> angle = 90°
 *   error_x = +1.0 (far RIGHT) -> angle = 180°
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ========== Configuration ==========

// WiFi - edit for your network
const char* WIFI_SSID = "RCA-A";
const char* WIFI_PASSWORD = "RCA@2024";

// MQTT - use your PC's WiFi IPv4 address (from ipconfig), NOT localhost
const char* MQTT_HOST = "10.12.74.224";
const uint16_t MQTT_PORT = 1883;
const char* MQTT_TOPIC = "facex/tracking";
const char* MQTT_CLIENT_ID = "esp8266-needle";

// GPIO pin assignments (NodeMCU silkscreen -> GPIO number)
const uint8_t PIN_IN1 = 5;   // D1
const uint8_t PIN_IN2 = 4;   // D2
const uint8_t PIN_IN3 = 14;  // D5
const uint8_t PIN_IN4 = 12;  // D6

// Motor parameters for 28BYJ-48
const float STEPS_PER_REVOLUTION = 2048.0f;  // Standard for 28BYJ-48 with 4-step sequence
const float STEPS_PER_DEGREE = STEPS_PER_REVOLUTION / 360.0f;
const unsigned long STEP_INTERVAL_US = 4000;  // 4ms between steps; increase if motor stalls
const float ANGLE_TOLERANCE = 0.9f;  // Stop when within this many degrees of target

// ========== Global State ==========

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

// Four-state full-step sequence for bipolar stepper
// To reverse motor direction, swap this to {3,2,1,0} order
const uint8_t STEP_SEQUENCE[4][4] = {
  {1, 0, 1, 0},  // Step 0
  {0, 1, 1, 0},  // Step 1
  {0, 1, 0, 1},  // Step 2
  {1, 0, 0, 1}   // Step 3
};

int stepIndex = 0;               // Current position in STEP_SEQUENCE
long currentStepPosition = 0;    // Signed step count (0 = 90 degrees)
float targetAngleDeg = 90.0f;    // Target needle angle [0, 180]
bool lockedFromMqtt = false;     // Only move when face is LOCKED
unsigned long lastStepUs = 0;
String serialBuffer;

// ========== Position Functions ==========

float currentAngleDeg() {
  // Position 0 = 90 degrees (center)
  // Positive steps = angles > 90 (right)
  // Negative steps = angles < 90 (left)
  return 90.0f + ((float)currentStepPosition / STEPS_PER_DEGREE);
}

// ========== Motor Control ==========

void writeStep(uint8_t index) {
  digitalWrite(PIN_IN1, STEP_SEQUENCE[index][0]);
  digitalWrite(PIN_IN2, STEP_SEQUENCE[index][1]);
  digitalWrite(PIN_IN3, STEP_SEQUENCE[index][2]);
  digitalWrite(PIN_IN4, STEP_SEQUENCE[index][3]);
}

void stepOnce(int direction) {
  // direction: +1 = increase angle (toward 180), -1 = decrease angle (toward 0)
  if (direction != 1 && direction != -1) {
    return;
  }
  
  // Update sequence index (wraps around 0-3)
  stepIndex = (stepIndex + direction + 4) % 4;
  currentStepPosition += direction;
  
  writeStep(stepIndex);
  
  // Debug output (comment out for production)
  Serial.print("[STEP] dir=");
  Serial.print(direction > 0 ? "+1" : "-1");
  Serial.print(" stepIdx=");
  Serial.print(stepIndex);
  Serial.print(" pos=");
  Serial.print(currentStepPosition);
  Serial.print(" angle=");
  Serial.println(currentAngleDeg(), 1);
}

void updateNeedleControl() {
  float current = currentAngleDeg();
  float error = targetAngleDeg - current;
  
  if (fabsf(error) <= ANGLE_TOLERANCE) {
    return;  // Already at target
  }
  
  unsigned long now = micros();
  if ((unsigned long)(now - lastStepUs) < STEP_INTERVAL_US) {
    return;  // Not enough time elapsed
  }
  
  lastStepUs = now;
  
  // Calculate direction based on error
  // error > 0: target is higher, need to increase angle (direction = +1)
  // error < 0: target is lower, need to decrease angle (direction = -1)
  int direction = (error > 0.0f) ? 1 : -1;
  
  stepOnce(direction);
}

// ========== MQTT ==========

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<256> doc;
  
  if (deserializeJson(doc, payload, length) != DeserializationError::Ok) {
    Serial.println("[MQTT] JSON parse error");
    return;
  }
  
  bool isLocked = doc["locked"] | false;
  float receivedAngle = doc["angle"] | 90.0f;
  float error_x = doc["error_x"] | 0.0f;
  const char* horizontal = doc["horizontal"] | "CENTER";
  const char* state = doc["state"] | "?";
  
  // Clamp angle to valid range
  receivedAngle = constrain(receivedAngle, 0.0f, 180.0f);
  
  // Only update target when locked
  if (isLocked) {
    targetAngleDeg = receivedAngle;
    lockedFromMqtt = true;
  } else {
    lockedFromMqtt = false;
    // Hold current position when not locked
  }
  
  Serial.print("[MQTT] state=");
  Serial.print(state);
  Serial.print(" locked=");
  Serial.print(isLocked ? "true" : "false");
  Serial.print(" horizontal=");
  Serial.print(horizontal);
  Serial.print(" error_x=");
  Serial.print(error_x, 3);
  Serial.print(" angle=");
  Serial.print(receivedAngle, 1);
  Serial.print(" current=");
  Serial.print(currentAngleDeg(), 1);
  Serial.print(" target=");
  Serial.println(targetAngleDeg, 1);
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  Serial.print("[WiFi] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.print("[WiFi] Connected, IP: ");
  Serial.println(WiFi.localIP());
}

void configureMqtt() {
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMqttMessage);
}

void ensureMqttConnected() {
  if (mqtt.connected()) {
    return;
  }
  
  Serial.print("[MQTT] Connecting to ");
  Serial.print(MQTT_HOST);
  Serial.print(":");
  Serial.print(MQTT_PORT);
  Serial.print("...");
  
  if (mqtt.connect(MQTT_CLIENT_ID)) {
    Serial.println(" connected");
    mqtt.subscribe(MQTT_TOPIC);
    Serial.print("[MQTT] Subscribed to ");
    Serial.println(MQTT_TOPIC);
  } else {
    Serial.print(" failed, rc=");
    Serial.println(mqtt.state());
  }
}

// ========== Serial Commands ==========

void runMotorTest() {
  Serial.println("\n=== BIDIRECTIONAL MOTOR TEST ===");
  Serial.print("Starting angle: ");
  Serial.println(currentAngleDeg(), 1);
  
  Serial.println("\nMoving 50 steps in direction +1 (should increase angle)...");
  for (int i = 0; i < 50; i++) {
    stepOnce(1);
    delay(10);
  }
  Serial.print("Angle after +1 steps: ");
  Serial.println(currentAngleDeg(), 1);
  
  Serial.println("\nPausing 2 seconds...");
  delay(2000);
  
  Serial.println("Moving 50 steps in direction -1 (should decrease angle)...");
  for (int i = 0; i < 50; i++) {
    stepOnce(-1);
    delay(10);
  }
  Serial.print("Angle after -1 steps: ");
  Serial.println(currentAngleDeg(), 1);
  
  Serial.println("\n=== TEST COMPLETE ===");
  Serial.println("Did the needle return to its starting position?");
  Serial.println("If NO, the step sequence is reversed or wiring is incorrect.\n");
}

void handleSerialInput() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    
    if (c == '\r' || c == '\n') {
      serialBuffer.trim();
      
      if (serialBuffer.length() == 0) {
        continue;
      }
      
      if (serialBuffer.equalsIgnoreCase("test")) {
        runMotorTest();
      } else if (serialBuffer.equalsIgnoreCase("status")) {
        Serial.print("Current angle: ");
        Serial.print(currentAngleDeg(), 1);
        Serial.print("°  Target: ");
        Serial.print(targetAngleDeg, 1);
        Serial.print("°  Locked: ");
        Serial.println(lockedFromMqtt ? "true" : "false");
      } else if (serialBuffer.equalsIgnoreCase("reset")) {
        currentStepPosition = 0;
        Serial.println("Position reset to 90°");
      } else {
        // Try to parse as angle
        float angle = serialBuffer.toFloat();
        if (angle >= 0.0f && angle <= 180.0f) {
          targetAngleDeg = angle;
          lockedFromMqtt = false;
          Serial.print("[SERIAL] Target set to ");
          Serial.print(targetAngleDeg, 1);
          Serial.println("°");
        } else {
          Serial.println("Commands: test, status, reset, or angle (0-180)");
        }
      }
      
      serialBuffer = "";
    } else if (serialBuffer.length() < 32) {
      serialBuffer += c;
    }
  }
}

// ========== Setup & Loop ==========

void setup() {
  Serial.begin(115200);
  delay(100);
  
  Serial.println("\n\nFaceX Needle Controller");
  Serial.println("Commands: test, status, reset, or enter angle (0-180)");
  
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);
  
  writeStep(stepIndex);  // Initialize motor coils
  
  connectWifi();
  configureMqtt();
  
  Serial.println("\nReady. Waiting for MQTT messages...\n");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
  }
  
  ensureMqttConnected();
  mqtt.loop();
  
  handleSerialInput();
  updateNeedleControl();
}
