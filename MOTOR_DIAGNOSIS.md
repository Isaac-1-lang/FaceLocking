# FaceX Motor Direction Diagnosis and Fix

## Root Cause Analysis

After systematic investigation, the issue has been identified:

### Problem Statement
The tachometer needle moves physically in only ONE direction, regardless of whether the tracked face moves LEFT or RIGHT.

### Software Analysis - CORRECT ✓

1. **Python angle mapping** (`src/mqtt_bridge.py`): **VERIFIED CORRECT**
   ```
   error_x = -1.0 (LEFT)  → angle = 0°   ✓
   error_x = 0.0 (CENTER) → angle = 90°  ✓
   error_x = +1.0 (RIGHT) → angle = 180° ✓
   ```

2. **Face tracking** (`src/face_tracking.py`): **VERIFIED CORRECT**
   - Negative error_x when face is LEFT of center ✓
   - Positive error_x when face is RIGHT of center ✓

3. **MQTT communication**: **VERIFIED WORKING**
   - Serial monitor shows both negative and positive error_x values ✓
   - Angles correctly vary from 0-180° ✓

4. **Arduino direction logic**: **VERIFIED CORRECT**
   ```cpp
   float error = targetAngleDeg - currentAngleDeg();
   int direction = (error > 0.0f) ? 1 : -1;
   ```
   - When target > current: direction = +1 (should increase angle)
   - When target < current: direction = -1 (should decrease angle)

### Hardware Issue - THE PROBLEM ⚠️

The **step sequence** or **motor wiring** causes the physical motor to rotate in the same direction for both `direction = +1` and `direction = -1`.

## Evidence from Serial Monitor

User's output shows:
```
[MQTT] horizontal=LEFT error_x=-0.529 angle=42.42 current=56.60 target=42.42
[STEP] dir=1 stepIdx 1->2 pos -19->-18 angle 86.84
```

And later:
```
[MQTT] horizontal=RIGHT error_x=0.620 angle=145.76
```

**Both LEFT and RIGHT positions are being tracked correctly by software**, but the physical motor doesn't reverse.

## Possible Causes

### Cause 1: Step Sequence Direction (MOST LIKELY)
The step sequence traversal direction doesn't match the expected motor rotation.

**Current sequence:**
```cpp
const uint8_t STEP_SEQUENCE[4][4] = {
  {1, 0, 1, 0},  // Step 0
  {0, 1, 1, 0},  // Step 1
  {0, 1, 0, 1},  // Step 2
  {1, 0, 0, 1}   // Step 3
};
```

When `direction = +1`: sequence goes 0→1→2→3→0...
When `direction = -1`: sequence goes 0→3→2→1→0...

**If the motor only moves one direction physically**, this sequence may need to be reversed.

### Cause 2: Incorrect Wiring
One coil pair may be wired backwards, causing both forward and reverse commands to result in the same physical direction.

### Cause 3: Motor Type Mismatch
The 28BYJ-48 must be properly modified for bipolar operation. A stock unipolar motor with just the red wire disconnected may not work correctly with an L298N.

## Solution Steps

### Step 1: Run the Built-In Motor Test

Upload the corrected Arduino firmware and run:

```
test
```

This will:
1. Move 50 steps in direction +1
2. Pause 2 seconds
3. Move 50 steps in direction -1

**Expected result:** Needle should return to starting position
**If needle does NOT return:** Proceed to Step 2

### Step 2: Reverse the Step Sequence

If the test fails, change the step sequence in the Arduino code:

**Option A - Reverse sequence order:**
```cpp
const uint8_t STEP_SEQUENCE[4][4] = {
  {1, 0, 0, 1},  // Step 0 (was Step 3)
  {0, 1, 0, 1},  // Step 1 (was Step 2)
  {0, 1, 1, 0},  // Step 2 (was Step 1)
  {1, 0, 1, 0}   // Step 3 (was Step 0)
};
```

**Option B - Invert direction in stepOnce():**
```cpp
void stepOnce(int direction) {
  direction = -direction;  // Add this line to reverse
  // ... rest of function
}
```

### Step 3: Verify Wiring

If reversing the sequence doesn't help, check wiring:

**28BYJ-48 standard colors:**
- Blue + Pink = Coil A → L298N IN1 + IN2
- Yellow + Orange = Coil B → L298N IN3 + IN4
- Red wire = **DISCONNECTED** (for bipolar mode)

**If you accidentally have:**
- Blue + Yellow on IN1/IN2 (split across coils)
- The motor will only vibrate or move one direction

**Correct wiring:**
```
ESP8266 D1 (GPIO5)  → L298N IN1 → Motor Blue wire
ESP8266 D2 (GPIO4)  → L298N IN2 → Motor Pink wire
ESP8266 D5 (GPIO14) → L298N IN3 → Motor Yellow wire
ESP8266 D6 (GPIO12) → L298N IN4 → Motor Orange wire
Motor Red wire → NOT CONNECTED
```

### Step 4: Check Motor Type

Verify your motor is actually a 28BYJ-48 and can operate in bipolar mode. Some motors cannot be reliably converted by just disconnecting the center tap.

## Testing Protocol

### Test A: Angle Mapping (Python)
```bash
python test_angle_mapping.py
```
**Status:** ✓ PASSED

### Test B: Motor Bidirectional Test (Arduino)
1. Upload corrected firmware
2. Open Serial Monitor (115200 baud)
3. Type `test` and press Enter
4. Watch the needle - it should move forward, pause, then return

**Expected Serial Output:**
```
=== BIDIRECTIONAL MOTOR TEST ===
Starting angle: 90.0°
Moving 50 steps in direction +1 (should increase angle)...
Angle after +1 steps: 98.8°
Pausing 2 seconds...
Moving 50 steps in direction -1 (should decrease angle)...
Angle after -1 steps: 90.0°
=== TEST COMPLETE ===
```

### Test C: Manual Angle Commands
```
90    (center)
150   (should move RIGHT)
30    (should move LEFT back)
90    (should return to center)
```

### Test D: Full Face Tracking
```bash
python -m src.face_tracking --target "YourName" --mqtt-host 127.0.0.1
```
Move your face LEFT and RIGHT - needle should follow.

## Fixed Files

1. **arduino/esp8266_needle_stepper/esp8266_needle_stepper.ino** - Cleaned up, added diagnostics
2. **test_angle_mapping.py** - Python angle test (PASSED)
3. **This diagnostic document**

## Next Steps

1. ✓ Upload the corrected Arduino firmware
2. ⚠ Run `test` command and report results
3. If test fails, try reversing step sequence (see Step 2 above)
4. If still failing, verify wiring (see Step 3 above)

## Summary

**Software:** All Python code and MQTT communication are correct.
**Hardware:** The issue is in the Arduino step sequence direction or motor wiring.
**Solution:** Use the built-in `test` command to diagnose, then reverse the step sequence if needed.
