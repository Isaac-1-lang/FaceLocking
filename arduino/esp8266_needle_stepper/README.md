# ESP8266 needle stepper

A physical gauge needle that mirrors the identity-lock tracker in
[`src/face_tracking.py`](../../src/face_tracking.py): when the locked face
moves right, the needle turns right by the same amount; while the target
isn't `LOCKED` the needle holds still instead of chasing noise.

```
Python (paho-mqtt)                    ESP8266 (PubSubClient)
------------------                    -----------------------
face_tracking.py --mqtt-host ...  -->  MQTT broker  -->  esp8266_needle_stepper.ino
  (src/mqtt_bridge.py publishes)        (on your PC)       L298N (IN1-IN4) -> 28BYJ-48 (bipolar-hacked) -> needle
```

## Your motor: 28BYJ-48, bipolar-hacked, no tachometer

The 28BYJ-48 is normally a *unipolar* stepper driven by its matching
ULN2003 board, with 5 wires: 4 coil leads plus a red common/center-tap
wire. Driving only the 4 coil leads through an H-bridge like your L298N
and **leaving the red wire disconnected** is a well-known hack that turns
it into a bipolar stepper — gives a bit more torque/speed than the stock
ULN2003 setup, and is exactly what this firmware expects.

A stepper's shaft position *is* its step count: command one step, the
shaft reliably moves one fixed angle, no slip. So there's no tachometer or
encoder to wire at all — the firmware's own step counter is the position
sensor.

**Standard 28BYJ-48 coil colors** (same on essentially every one sold):
Blue and Pink are one coil; Yellow and Orange are the other. No multimeter
needed — just wire Blue/Pink to `IN1`/`IN2` and Yellow/Orange to `IN3`/`IN4`,
and leave Red unconnected. If the needle only buzzes/vibrates without
turning, you've swapped a pair across coils by mistake — double check Blue
and Pink are together on `IN1`/`IN2`, not split across `IN1`/`IN3`.

**Power:** most 28BYJ-48 motors are rated **5V** (there's also a less common
12V variant — check the label on yours). Feed the L298N's motor-supply
input the voltage that matches your motor, not 12V into a 5V one.

## Your L298N: IN1-IN4, no visible ENA/ENB

Cheap L298N boards put an `ENA`/`ENB` pin next to `IN1`/`IN2` (and `IN3`/`IN4`),
usually under a small removable jumper cap. If you only see `IN1`-`IN4` and
can't find or read an `ENA`/`ENB` jumper, it's almost certainly still in
place, tying the enable lines permanently HIGH. That's exactly what this
firmware wants: driving a stepper through `IN1`-`IN4` only ever needs plain
HIGH/LOW levels, never PWM, so there's nothing else to wire for ENA/ENB.

## Wiring

ESP8266 dev boards (NodeMCU, Wemos D1 Mini) print a `Dx` label next to each
pin, but the firmware needs the **GPIO number** underneath/next to it:

| Silkscreen | GPIO | Use here                       |
|------------|------|----------------------------------|
| D0         | 16   | avoid — no interrupt support     |
| **D1**     | **5**| **L298N IN1** ← Blue             |
| **D2**     | **4**| **L298N IN2** ← Pink             |
| D3         | 0    | avoid — must be HIGH at boot     |
| D4         | 2    | avoid — must be HIGH at boot, onboard LED |
| **D5**     | **14**| **L298N IN3** ← Yellow          |
| **D6**     | **12**| **L298N IN4** ← Orange          |
| D7         | 13   | free                              |
| D8         | 15   | avoid — must be LOW at boot      |

| Signal              | ESP8266 pin  |
|---------------------|--------------|
| L298N IN1           | D1 (GPIO5)   |
| L298N IN2           | D2 (GPIO4)   |
| L298N IN3           | D5 (GPIO14)  |
| L298N IN4           | D6 (GPIO12)  |
| L298N +12V / motor  | external 5V supply (or 12V if yours is the 12V variant) — **not** the ESP8266 5V/VIN pin |
| L298N GND           | ESP8266 GND — common ground with the external supply |
| 28BYJ-48 red wire   | **not connected** (bipolar hack — see above) |

> If your board is a **"Wemos D1 R1/R2"** (Arduino-Uno-shaped, not the small
> D1 Mini), its `Dx` labels map to **different** GPIOs than the table above —
> go by the GPIO number on the board/schematic for that variant.

`PIN_IN1`-`PIN_IN4` are plain `const int` GPIO numbers at the top of the
`.ino` — change them there if you wire to different pins.
`STEPS_PER_REVOLUTION` is already set to **2048**, the standard value for a
28BYJ-48 driven with this firmware's 4-step full-step sequence (32 rotor
steps x ~64:1 gearbox); you shouldn't need to touch it unless you swap to a
different motor.

## Libraries

- **ESP8266 board package** ("esp8266" by ESP8266 Community) — via Boards
  Manager, not Library Manager. In the Arduino IDE: File → Preferences → add
  `https://arduino.esp8266.com/stable/package_esp8266com_index.json` to
  "Additional Boards Manager URLs", install "esp8266", then select your
  board (e.g. "NodeMCU 1.0 (ESP-12E Module)").
- `PubSubClient` (Nick O'Leary) — Library Manager
- `ArduinoJson` (Benoit Blanchon), v6 or v7 — Library Manager

## Running the MQTT broker on your own PC

You don't need any external/cloud broker — your laptop running `mosquitto`
is the broker. The ESP8266 and the PC just both need to be on the **same
WiFi network** (the `WIFI_SSID` in the sketch).

### Install and run Mosquitto on Windows

1. Download the installer from
   [mosquitto.org/download](https://mosquitto.org/download/) (Windows
   section), or with Chocolatey:

   ```bash
   choco install mosquitto
   ```

2. It installs as a Windows service and starts listening on port `1883`
   automatically. If it isn't running, start it from an elevated prompt:

   ```bash
   net start mosquitto
   ```

3. Allow inbound connections on port 1883 through Windows Defender Firewall
   (the ESP8266 connects in from the WiFi network, so this is required):

   ```bash
   netsh advfirewall firewall add rule name="Mosquitto MQTT" dir=in action=allow protocol=TCP localport=1883
   ```

4. Find your PC's WiFi IPv4 address — this is the address the **ESP8266**
   needs (it can't use `localhost`, since it's a separate device):

   ```bash
   ipconfig
   ```

   Look under your WiFi adapter for "IPv4 Address", e.g. `192.168.1.50`.

5. Put that IP in `esp8266_needle_stepper.ino` as `MQTT_HOST`, reflash.

6. Python (`face_tracking.py`) runs on the same PC as the broker, so it can
   use `--mqtt-host 127.0.0.1` (or `localhost`) — it doesn't need the WiFi
   IP the ESP8266 uses.

### Quick test the broker is reachable (optional)

Mosquitto ships `mosquitto_sub`/`mosquitto_pub` CLI tools. From the install
directory (or add it to PATH):

```bash
mosquitto_sub -h localhost -t facex/tracking
```

Leave that running, then in another terminal:

```bash
mosquitto_pub -h localhost -t facex/tracking -m "{\"state\":\"LOCKED\",\"locked\":true,\"angle\":45}"
```

You should see the message show up in the `mosquitto_sub` window — that
confirms the broker itself works before you involve the ESP8266 or Python.

## Running it end to end

1. Start Mosquitto on your PC (see above) and flash
   `esp8266_needle_stepper.ino` to the ESP8266; open its Serial monitor at
   115200 baud — it logs WiFi/MQTT connection and every message it receives.
2. Run the tracker with MQTT publishing enabled:

   ```bash
   python -m src.face_tracking --target "Your Name" --mqtt-host 127.0.0.1
   ```

3. While `face_tracking.py` reports `LOCKED`, move the recognized face left
   and right — the needle should track in the same direction. While
   `SEARCHING`/`LOST`/`UNCERTAIN`, the needle holds its last position.

## Testing the needle without the camera pipeline

Open the ESP8266's Serial monitor and type a number from 0 to 180, then
press Enter — the needle steps to that angle. Typing a value manually
releases MQTT control until the next `"locked": true` message arrives, so
you can bench-test the stepper wiring independently of face tracking.

## MQTT payload contract

Published by [`src/mqtt_bridge.py`](../../src/mqtt_bridge.py)'s
`NeedlePublisher`, once per tracked frame, on the topic passed to
`--mqtt-topic` (default `facex/tracking`):

```json
{
  "state": "LOCKED",
  "locked": true,
  "target": "Alice",
  "horizontal": "RIGHT",
  "error_x": 0.42,
  "angle": 127.8
}
```

- `state` — one of `SEARCHING`, `LOCKED`, `UNCERTAIN`, `LOST`
  (`src.face_tracking.LockState`).
- `locked` — `true` only when `state == "LOCKED"`; the firmware ignores
  `angle` updates otherwise.
- `error_x` — the tracker's horizontal error in `[-1, 1]`
  (`TrackingSignal.error_x`); negative is LEFT, positive is RIGHT.
- `angle` — `error_x` mapped onto `[0, 180]` with `90` as center, so the
  needle's left/right sense matches the tracker's `horizontal` field
  directly.
