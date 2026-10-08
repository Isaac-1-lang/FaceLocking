/*
 * esp8266_needle_stepper.ino
 *
 * Drives a gauge needle to an angle (0-180 degrees) using:
 *   - A 28BYJ-48 stepper, bipolar-hacked through an L298N driver's
 *     IN1-IN4 (no ENA/ENB wiring needed - see "No ENA/ENB" below) -
 *     its 5th/red common wire is left disconnected; see STEPS_PER_REVOLUTION
 *     and the PIN_IN1-4 comments below for the coil wiring
 *   - Step counting for position - a stepper's position IS its step
 *     count, so no tachometer/encoder hardware is needed at all
 *   - MQTT (matches the Python side's paho-mqtt publisher in
 *     src/mqtt_bridge.py) so the needle follows the face-tracking
 *     identity lock in src/face_tracking.py
 *   - Serial input so you can type an angle by hand for bench testing
 *     without the camera/MQTT pipeline running
 *
 * Payload received on MQTT topic FACEX_TOPIC (JSON, published once per
 * tracked frame by NeedlePublisher.publish in src/mqtt_bridge.py):
 *   {"state":"LOCKED","locked":true,"target":"Alice",
 *    "horizontal":"RIGHT","error_x":0.42,"angle":127.8}
 *
 * angle is already mapped from error_x: 90 = center, 0 = full LEFT,
 * 180 = full RIGHT, so the needle turns the same direction the locked
 * face actually moved. The needle only moves while "locked" is true;
 * otherwise it holds its last position (SEARCHING/LOST/UNCERTAIN don't
 * yank the needle around on a shaky, unverified detection).
 *
 * Libraries:
 *   - ESP8266 board package ("esp8266" by ESP8266 Community) - install
 *     via Boards Manager, not Library Manager; provides ESP8266WiFi.h
 *   - PubSubClient (Nick O'Leary) - Library Manager
 *   - ArduinoJson (Benoit Blanchon), v6 or v7 - Library Manager
 *
 * No ENA/ENB:
 *   Cheap L298N boards have an ENA/ENB pin next to IN1/IN2 (and IN3/IN4)
 *   with a removable jumper cap. If yours only exposes IN1-IN4 and you
 *   can't find/read an ENA/ENB jumper, it's almost certainly still in
 *   place, tying the enable lines permanently HIGH. That's fine here:
 *   driving a stepper through IN1-IN4 only ever needs plain HIGH/LOW
 *   levels (no PWM), so ENA/ENB being hard-wired enabled is exactly
 *   what this firmware wants - nothing else to wire.
 *
 * Why no tachometer:
 *   A stepper motor has no "slip" the way a DC motor does - one step
 *   command reliably moves the shaft by one fixed angle. That means the
 *   firmware's own step counter IS an accurate position sensor, so
 *   there's nothing extra to wire up. The only hardware fact you must
 *   get right is STEPS_PER_REVOLUTION below, which depends on your
 *   specific motor (see the comment at that constant).
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ---------------------------------------------------------------------
// Wiring / configuration - edit these for your setup
// ---------------------------------------------------------------------
const char *WIFI_SSID = "Main Hall";
const char *WIFI_PASSWORD = "Meeting@2024";

// MQTT_HOST is your PC's LAN IP address (the machine running the
// broker and face_tracking.py), NOT "localhost" - the ESP8266 is a
// separate device on the WiFi network and can't resolve "localhost"
// to your PC. Find it with `ipconfig` on Windows (look for "IPv4
// Address" under the WiFi adapter) and make sure the PC and the
// ESP8266 are on the same WiFi network ("Main Hall" above).
const char *MQTT_HOST = "10.12.74.170"; // your PC's IPv4 address (from ipconfig)
const uint16_t MQTT_PORT = 1883;
const char *MQTT_TOPIC = "facex/tracking"; // must match --mqtt-topic in face_tracking.py
const char *MQTT_CLIENT_ID = "esp8266-needle";

// L298N driver pins - all four, one pair per stepper coil.
// ESP8266 dev boards print a "Dx" silkscreen label next to each pin,
// but the firmware (and the ESP8266 core) only understands the GPIO
// number, usually printed smaller, right under/next to the Dx label.
// Pick pins by GPIO number using this table for the common NodeMCU /
// Wemos D1 Mini layout (ESP-12E module):
//
//   silkscreen | GPIO | notes
//   -----------|------|----------------------------------------------
//   D0         | 16   | no interrupt support - avoid
//   D1         | 5    | safe for output
//   D2         | 4    | safe for output
//   D3         | 0    | must be HIGH at boot - avoid
//   D4         | 2    | must be HIGH at boot; onboard LED - avoid
//   D5         | 14   | safe for output
//   D6         | 12   | safe for output
//   D7         | 13   | safe for output
//   D8         | 15   | must be LOW at boot - avoid
//
// If your board is a "Wemos D1 R1/R2" (Arduino-Uno-shaped), its D-labels
// map to DIFFERENT GPIOs - go by the GPIO number on the board/schematic,
// not the D-label, for that variant.
// 28BYJ-48 coil wiring, driven in "bipolar hack" mode (the usual 5th/red
// wire - the common center tap - is left disconnected; only the 4
// colored coil leads go to the L298N). Standard colors: Blue+Pink are
// one coil, Yellow+Orange are the other - no multimeter needed, the
// pairing is fixed by the motor's own wiring.
const int PIN_IN1 = 5;  // D1 (GPIO5)  -> L298N IN1 -> Blue
const int PIN_IN2 = 4;  // D2 (GPIO4)  -> L298N IN2 -> Pink
const int PIN_IN3 = 14; // D5 (GPIO14) -> L298N IN3 -> Yellow
const int PIN_IN4 = 12; // D6 (GPIO12) -> L298N IN4 -> Orange

// 28BYJ-48: 32 full-steps per internal-rotor revolution x ~64:1 gearbox
// = 2048 steps per output-shaft revolution, for the 4-step full-step
// sequence this firmware uses (STEP_SEQUENCE below). This is the
// standard value used across Arduino 28BYJ-48 code/tutorials.
const float STEPS_PER_REVOLUTION = 2048.0;
const float STEPS_PER_DEGREE = STEPS_PER_REVOLUTION / 360.0;

// 28BYJ-48 is a slow, low-torque motor (even more so bipolar-hacked
// without its matched ULN2003 board) - a longer delay than a NEMA17
// needs is safer against stalling/skipping steps.
const unsigned long STEP_INTERVAL_US = 4000; // delay between steps; lower = faster, raise if it stalls/skips
const float ANGLE_TOLERANCE = 0.9; // degrees; smaller than one step, so "arrived" means "exact step"

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

// Classic 2-phases-on full-step sequence for a bipolar stepper driven
// through two H-bridges (IN1/IN2 for coil A, IN3/IN4 for coil B).
const uint8_t STEP_SEQUENCE[4][4] = {
  {1, 0, 1, 0},
  {0, 1, 1, 0},
  {0, 1, 0, 1},
  {1, 0, 0, 1},
};
int stepIndex = 0;          // current index into STEP_SEQUENCE
long currentStepPosition = 0; // signed step count = current position; this IS the position sensor

float targetAngleDeg = 90.0;  // commanded needle angle, 0-180
bool lockedFromMqtt = false;  // only follow MQTT target while the face is LOCKED
unsigned long lastStepUs = 0;
String serialBuffer;

// ---------------------------------------------------------------------
// Stepper driver
// ---------------------------------------------------------------------
void writeStep(int index) {
  digitalWrite(PIN_IN1, STEP_SEQUENCE[index][0]);
  digitalWrite(PIN_IN2, STEP_SEQUENCE[index][1]);
  digitalWrite(PIN_IN3, STEP_SEQUENCE[index][2]);
  digitalWrite(PIN_IN4, STEP_SEQUENCE[index][3]);
}

// direction: +1 = toward 180 deg (RIGHT), -1 = toward 0 deg (LEFT)
void stepOnce(int direction) {
  stepIndex = (stepIndex + direction + 4) % 4;
  writeStep(stepIndex);
  currentStepPosition += direction;
}

void releaseCoils() {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, LOW);
}

float currentAngleDeg() {
  // 90 degrees is the mechanical center the shaft is assumed to start at
  // on power-up; re-home manually (or add a limit switch) if that's not true.
  return 90.0 + (float)currentStepPosition / STEPS_PER_DEGREE;
}

// Called often from loop(); advances at most one step per
// STEP_INTERVAL_US toward targetAngleDeg. Non-blocking, so MQTT/Serial
// keep being serviced while the needle is travelling.
void updateNeedleControl() {
  float error = targetAngleDeg - currentAngleDeg();
  if (fabs(error) <= ANGLE_TOLERANCE) {
    return; // arrived; leave coils energized at the last step for holding torque
  }
  unsigned long now = micros();
  if (now - lastStepUs < STEP_INTERVAL_US) {
    return;
  }
  lastStepUs = now;
  stepOnce(error > 0 ? 1 : -1);
}

// ---------------------------------------------------------------------
// MQTT
// ---------------------------------------------------------------------
void onMqttMessage(char *topic, byte *payload, unsigned int length) {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, payload, length) != DeserializationError::Ok) {
    return;
  }
  lockedFromMqtt = doc["locked"] | false;
  float angle = doc["angle"] | 90.0;
  angle = constrain(angle, 0.0, 180.0);

  // The needle only follows the lock while the target is actually
  // LOCKED; this is what ties "locking" to needle movement - a
  // SEARCHING/LOST/UNCERTAIN frame holds the last commanded angle
  // instead of swinging the needle on an unverified face.
  if (lockedFromMqtt) {
    targetAngleDeg = angle;
  }

  Serial.print("[MQTT] state=");
  Serial.print((const char *)(doc["state"] | "?"));
  Serial.print(" horizontal=");
  Serial.print((const char *)(doc["horizontal"] | "?"));
  Serial.print(" angle=");
  Serial.print(angle);
  Serial.print(" locked=");
  Serial.println(lockedFromMqtt ? "true" : "false");
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected, IP=");
  Serial.println(WiFi.localIP());
}

void connectMqtt() {
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMqttMessage);
}

void ensureMqttConnected() {
  if (mqtt.connected()) {
    return;
  }
  Serial.print("Connecting to MQTT broker ");
  Serial.print(MQTT_HOST);
  Serial.print(" ...");
  if (mqtt.connect(MQTT_CLIENT_ID)) {
    Serial.println(" connected");
    mqtt.subscribe(MQTT_TOPIC);
  } else {
    Serial.print(" failed, rc=");
    Serial.println(mqtt.state());
  }
}

// ---------------------------------------------------------------------
// Serial: type an angle (0-180) and press Enter to move the needle by
// hand, independent of MQTT. Typing manually releases MQTT control
// until the next "locked":true message arrives.
// ---------------------------------------------------------------------
void handleSerialInput() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        float angle = serialBuffer.toFloat();
        targetAngleDeg = constrain(angle, 0.0, 180.0);
        lockedFromMqtt = false; // manual entry takes precedence until MQTT locks again
        Serial.print("[SERIAL] target angle set to ");
        Serial.println(targetAngleDeg);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
    }
  }
}

// ---------------------------------------------------------------------
// Setup / loop
// ---------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("FaceX needle stepper - type an angle (0-180) + Enter to test manually");

  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);

  writeStep(stepIndex);
  connectWifi();
  connectMqtt();
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
