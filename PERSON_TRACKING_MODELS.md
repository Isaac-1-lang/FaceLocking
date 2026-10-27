# Person Tracking Models - src.person_tracking

## Overview

The `src.person_tracking` module implements **whole-person tracking** that survives temporary face loss. Unlike `face_tracking.py` which only tracks faces, this module tracks the entire person's body.

## Models Used

### 1. **YuNet** - Face Detection (Same as face_tracking)
- **File:** `models/detector_yunet.onnx` (227 KB)
- **Purpose:** Detects faces and provides 5 facial landmarks
- **Source:** OpenCV Zoo
- **Usage:** Primary method for detecting and identifying the person
- **Downscaling:** Runs on downscaled frames (default: 640px max width) for CPU efficiency

### 2. **ArcFace ResNet100** - Face Recognition (Same as face_tracking)
- **File:** `models/embedder_arcface.onnx` (249 MB)
- **Purpose:** Identity verification when face is visible
- **Usage:** Sparingly called to verify identity (controlled by `--max-verifications`)
- **CPU Budget:** Limited number of calls per frame (default: 1)

### 3. **Person/Body Detector** - Three Options:

#### Option A: HOG (Default) ✓ NO MODEL DOWNLOAD REQUIRED
- **Model:** OpenCV built-in HOG + Linear SVM
- **File:** None - built into OpenCV
- **Backend:** `--person-backend hog`
- **Class:** `HogPersonDetector` in `detectors.py`
- **Features:**
  - Zero download required
  - Works out of the box
  - Best for standing people at moderate distance
  - May not detect close-up webcam framing
- **Fallback:** If HOG doesn't detect, uses face-derived body boxes

#### Option B: YOLO (Optional - Requires Model)
- **Model:** YOLOv5/YOLOv8 ONNX export
- **File:** User-provided (via `--person-model`)
- **Backend:** `--person-backend yolo`
- **Class:** `OnnxPersonDetector` in `detectors.py`
- **Features:**
  - More accurate than HOG
  - Requires separate YOLO model download
  - Supports YOLOv5, YOLOv8, YOLOX formats
  - Input size: 640×640 (or 320×320 for speed)
  - Only uses 'person' class (COCO class 0)
- **Requirements:**
  ```bash
  pip install onnxruntime
  # Then provide your YOLO model:
  python -m src.person_tracking --target "Name" --person-backend yolo --person-model path/to/yolo.onnx
  ```

#### Option C: None (Face-only)
- **Backend:** `--person-backend none`
- **Behavior:** No body detection; relies entirely on face-derived body boxes
- **Use case:** Maximum CPU savings when tracking only faces

## Complete Person Tracking Pipeline

```
Camera Frame / Video
    ↓
[Person Detection - Every N frames]
    ├─ HOG Detector (default, built-in)
    ├─ YOLO Detector (optional, user model)
    └─ OR None (face-derived boxes only)
    ↓
[Face Detection - Every frame, downscaled]
    └─ YuNet (640px max width for CPU efficiency)
    ↓
[Kalman Filter Tracking] ⭐
    └─ Constant-velocity motion model
    └─ State: [cx, cy, w, h, vx, vy]
    └─ Predicts position even when person not detected
    └─ Velocity damping for coasting tracks
    ↓
[Data Association]
    └─ Hungarian algorithm (linear assignment)
    └─ Match detections to existing tracks
    └─ Create new tracks for unmatched detections
    ↓
[Identity Verification - Sparingly]
    └─ ArcFace (max N calls per frame)
    └─ Budget-controlled, prioritized by track
    ↓
[Confidence Management]
    └─ Exponential decay without face
    └─ Half-life: default 3 seconds
    └─ Chi-squared gating for outlier rejection
    ↓
[State Machine]
    ├─ UNKNOWN (confidence < 0.20)
    ├─ TEMPORARY (0.20 ≤ confidence < 0.70)
    └─ CONFIRMED (confidence ≥ 0.70)
    ↓
[Centering Controller]
    └─ Horizontal/vertical aim calculation
    ↓
[Output]
    ├─ Visual overlay
    ├─ CSV log
    └─ MQTT (optional, for needle)
```

## Key Differences from face_tracking.py

| Feature | face_tracking.py | person_tracking |
|---------|-----------------|----------------|
| **Tracks** | Face box only | Whole body box |
| **Body Detection** | None | HOG/YOLO/None |
| **Face Loss** | Loses lock after ~24 frames | Survives with confidence decay |
| **Identity Checks** | Every frame (default) | Budget-controlled (max N/frame) |
| **Confidence** | Binary (locked/not) | Continuous decay over time |
| **States** | SEARCHING/LOCKED/UNCERTAIN/LOST | UNKNOWN/TEMPORARY/CONFIRMED |
| **CPU Usage** | Moderate | Lower (scheduled detections) |
| **Use Case** | Desktop/close-up | Surveillance/distance |

## Command-Line Model Selection

```bash
# Default: HOG person detector (no download needed)
python -m src.person_tracking --target "Isaac"

# YOLO person detector (requires model)
python -m src.person_tracking \
    --target "Isaac" \
    --person-backend yolo \
    --person-model models/yolov8n.onnx

# No person detector (face-only tracking)
python -m src.person_tracking \
    --target "Isaac" \
    --person-backend none

# Custom face models (same as face_tracking)
python -m src.person_tracking \
    --target "Isaac" \
    --detector models/detector_yunet.onnx \
    --model models/embedder_arcface.onnx \
    --preprocessing raw
```

## Performance Optimization Parameters

### CPU Budget Control:

```bash
# Detect bodies every 3 frames (default)
--detect-every 3

# Find faces at max 640px width (default, big CPU saving)
--detect-width 640

# Max 1 ArcFace call per frame (default)
--max-verifications 1
```

### Example High-Speed Configuration:
```bash
python -m src.person_tracking \
    --target "Isaac" \
    --person-backend hog \
    --detect-every 5 \
    --detect-width 480 \
    --max-verifications 1
```

## Confidence Management

Unlike binary LOCKED/LOST states, person tracking uses **continuous confidence**:

```
Confidence = e^(-t / half_life)

Where:
  t = seconds since last face verification
  half_life = time for confidence to drop by 50% (default: 3.0s)
```

**Parameters:**
- `--half-life 3.0`: Seconds for confidence to halve (default: 3.0)
- `--confirm 0.70`: Confidence needed for CONFIRMED state (default: 0.70)
- `--unknown 0.20`: Below this, state becomes UNKNOWN (default: 0.20)

**Example:**
- Face verified at t=0: confidence = 1.0 (CONFIRMED)
- Face lost at t=1: confidence decays
  - t=4s: confidence ≈ 0.50 (TEMPORARY)
  - t=7s: confidence ≈ 0.25 (TEMPORARY)
  - t=10s: confidence ≈ 0.12 (UNKNOWN)

## Model Files Summary

| Component | Model | File | Size | Required? |
|-----------|-------|------|------|-----------|
| **Face Detection** | YuNet | `models/detector_yunet.onnx` | 227 KB | ✓ Yes |
| **Face Recognition** | ArcFace ResNet100 | `models/embedder_arcface.onnx` | 249 MB | ✓ Yes |
| **Body Detection (HOG)** | Built-in OpenCV | None | 0 | ✓ Default |
| **Body Detection (YOLO)** | YOLOv5/v8 | User-provided | ~6-50 MB | ✗ Optional |

## YOLO Model Options

If you want to use YOLO instead of HOG:

### YOLOv8n (Nano - Fastest)
```bash
# Download from Ultralytics
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640

# Use with person_tracking
python -m src.person_tracking --target "Name" \
    --person-backend yolo \
    --person-model yolov8n.onnx
```

### YOLOv5s (Small)
```bash
# Download from official YOLOv5 repo
git clone https://github.com/ultralytics/yolov5
cd yolov5
python export.py --weights yolov5s.pt --include onnx

# Use with person_tracking
python -m src.person_tracking --target "Name" \
    --person-backend yolo \
    --person-model yolov5s.onnx
```

### YOLOv8m (Medium - Best Accuracy)
```bash
yolo export model=yolov8m.pt format=onnx imgsz=640

python -m src.person_tracking --target "Name" \
    --person-backend yolo \
    --person-model yolov8m.onnx
```

## Video Processing

Person tracking supports video files (not just live camera):

```bash
# Process video file
python -m src.person_tracking \
    --target "Isaac" \
    --video path/to/video.mp4 \
    --no-display

# With frame limit
python -m src.person_tracking \
    --target "Isaac" \
    --video clip.mp4 \
    --max-frames 1000
```

## Implementation Details

### Face Detection (YuNet)
**File:** `pipeline.py`, method `_faces()`
```python
def _faces(self, frame):
    # Downscale for CPU efficiency
    scale = min(1.0, self.detect_width / frame.shape[1])
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    
    # Detect on small frame
    faces = self.face_detector.detect(small)
    
    # Map boxes/landmarks back to full resolution
    for face in faces:
        face.box = face.box / scale
        face.points = face.points / scale
```

### Person Detection (HOG)
**File:** `detectors.py`, class `HogPersonDetector`
```python
class HogPersonDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )
    
    def detect(self, frame):
        # Built-in OpenCV HOG detector
        rects, weights = self.hog.detectMultiScale(frame)
        # NMS and filtering
        return detections
```

### Person Detection (YOLO)
**File:** `detectors.py`, class `OnnxPersonDetector`
```python
class OnnxPersonDetector:
    def __init__(self, model):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model)
    
    def detect(self, frame):
        # Letterbox, inference, decode
        # Only returns 'person' class (COCO index 0)
        return decode_yolo(output, ...)
```

## Scheduling and CPU Efficiency

Both face and body detection can be scheduled:

```python
# Body detection runs every N frames
person = Scheduled(HogPersonDetector(), every=3)

# Returns None on skipped frames
detections = person.detect(frame)
if detections is None:
    # Skipped this frame
```

This dramatically reduces CPU usage while maintaining tracking through Kalman filtering.

---

## Kalman Filter - The Secret Sauce 🎯

The **Kalman filter** is what makes person tracking robust. It's **not a machine learning model** - it's a mathematical algorithm that tracks object motion.

### What is the Kalman Filter?

**File:** `src/person_tracking/kalman.py`  
**Class:** `BoxKalman`  
**Implementation:** Pure NumPy (CPU, ~microseconds per update)

**Purpose:** Track a bounding box's position and velocity over time, predicting where it will be even when temporarily not detected.

### State Vector (6 dimensions):

```
[cx, cy, w, h, vx, vy]

Where:
  cx, cy = center position (x, y)
  w, h   = box width, height
  vx, vy = velocity (x, y direction)
```

### How It Works:

#### 1. **Initialization** (When track is created)
```python
# Person detected at position (x1, y1, x2, y2)
kf = BoxKalman(box)

# Initial state:
# - Position: center of detected box
# - Velocity: [0, 0] (unknown)
# - Uncertainty: moderate (10% of box size)
```

#### 2. **Prediction** (Every frame)
```python
kf.predict(dt, coasting=True)

# Updates state based on constant-velocity model:
# new_position = old_position + velocity * dt
# new_velocity = old_velocity * damping_factor

# Increases uncertainty (covariance grows)
```

**During occlusion:**
- Position predicted from last known velocity
- Velocity damped exponentially (doesn't fly off screen)
- Uncertainty grows over time

#### 3. **Update** (When measurement available)
```python
kf.update(detected_box)

# Fuses prediction with new measurement
# Uses Chi-squared gating:
#   - If measurement is wildly inconsistent → REJECT
#   - If borderline → down-weight
#   - If consistent → full update
# Reduces uncertainty
```

**Robust gating thresholds:**
- `GATE_CHI2 = 9.49` (95% confidence) - start down-weighting
- `REJECT_CHI2 = 18.5` (99.9% confidence) - reject outliers

### Adaptive Noise Scaling

**Key feature:** All noise scales with box size!

```python
# Position noise ∝ box size
# A person 10 meters away has proportionally larger position noise
# than someone 2 meters away

q[position] = (accel_std * box_size * dt^2)^2
```

This makes the filter work equally well for:
- Close-up webcam framing (small box)
- Distant surveillance (large box)

### Velocity Damping During Coasting

When no measurement for a while:

```python
F[vx, vx] = exp(-damping * dt)
F[vy, vy] = exp(-damping * dt)

# Default damping = 0.35
# After 1 second: velocity = 70% of original
# After 2 seconds: velocity = 50% of original
# After 3 seconds: velocity = 35% of original
```

**Result:** Track slows down instead of flying off screen during occlusion.

### Chi-Squared Gating (Outlier Rejection)

Prevents false detections from corrupting the track:

```python
# Compute Mahalanobis distance (innovation)
d² = (measurement - prediction)ᵀ S⁻¹ (measurement - prediction)

if d² > 18.5:   # 99.9% threshold
    # Reject - this is probably a different person
    return False
elif d² > 9.49:  # 95% threshold
    # Down-weight by increasing measurement noise
    R = R * (d² / 9.49)
```

### Performance Characteristics:

- **Speed:** ~5-50 microseconds per update (pure NumPy)
- **Memory:** ~500 bytes per track (6x6 covariance matrix)
- **Accuracy:** Can track through 1-2 seconds of occlusion
- **Drift:** Minimal with good velocity estimates

### Example Tracking Scenario:

```
t=0.0s: Person detected → Kalman initialized
  State: [400, 300, 80, 200, 0, 0]
  Uncertainty: low

t=0.1s: Person detected again → Update
  State: [405, 300, 80, 200, 50, 0]  (moving right at 50 px/s)
  Uncertainty: very low

t=0.2s: Person detected → Update
  State: [410, 300, 80, 200, 50, 0]
  Uncertainty: very low

t=0.3s: Person OCCLUDED (no detection) → Predict only
  State: [415, 300, 80, 200, 50, 0]  (predicted from velocity)
  Uncertainty: increasing

t=0.4s: Still occluded → Predict
  State: [420, 300, 80, 200, 48, 0]  (velocity damped slightly)
  Uncertainty: increasing

t=0.5s: Person RE-DETECTED → Update
  Measured: [422, 302, 78, 198]
  State fused with prediction
  Uncertainty: drops back down

Result: Track maintained through 0.2s occlusion!
```

### Code Implementation:

**File:** `src/person_tracking/kalman.py`

```python
class BoxKalman:
    """Constant-velocity Kalman filter for body box tracking."""
    
    def __init__(self, box, accel_std=1.5, size_std=0.25, 
                 meas_std=0.04, damping=0.35):
        # Initialize state [cx, cy, w, h, vx, vy]
        # Initialize covariance matrix P (6x6)
    
    def predict(self, dt, coasting=False):
        # State transition: x = F @ x
        # Covariance: P = F @ P @ F.T + Q
        # Apply velocity damping if coasting
    
    def update(self, box, noise_scale=1.0):
        # Measurement: z = [cx, cy, w, h]
        # Innovation: y = z - H @ x
        # Chi-squared gating
        # Kalman gain: K = P @ H.T @ inv(S)
        # Update: x = x + K @ y
        #         P = (I - K @ H) @ P
```

### Integration with Tracker:

**File:** `src/person_tracking/tracker.py`

```python
class Track:
    def __init__(self, track_id, box, t, ...):
        self.kf = BoxKalman(box)  # Each track has its own Kalman filter
        self.gap = 0.0  # Time since last measurement
    
    def predict(self, dt):
        coasting = self.gap > config.coast_grace
        self.kf.predict(dt, coasting=coasting)
    
    def update(self, box):
        self.kf.update(box)
        self.gap = 0.0
```

### Why Kalman Filter?

**Advantages:**
1. **Smooth tracking** - Filters noisy detections
2. **Occlusion handling** - Predicts position when person hidden
3. **Velocity estimation** - Knows how fast person is moving
4. **Outlier rejection** - Ignores false detections
5. **Fast** - Pure math, no neural network inference
6. **Deterministic** - Same input → same output

**Compared to just using raw detections:**
- Raw detections: jittery, lost during occlusion
- Kalman filtering: smooth, maintains track through gaps

### Tunable Parameters:

```python
BoxKalman(
    box,
    accel_std=1.5,      # Process noise (motion unpredictability)
    size_std=0.25,      # Box size change rate
    meas_std=0.04,      # Measurement noise (4% of box size)
    damping=0.35        # Velocity decay rate
)
```

**Higher `accel_std`:** More responsive to sudden movements, less smooth  
**Lower `accel_std`:** Smoother, but slower to react

**Higher `damping`:** Faster velocity decay during occlusion  
**Lower `damping`:** Maintains velocity longer

---

## Scheduling and CPU Efficiency (continued)

## Summary

**person_tracking uses 2-3 models:**

1. **YuNet** (face detection) - Required, 227 KB
2. **ArcFace** (face recognition) - Required, 249 MB  
3. **Body detector** - Choose one:
   - **HOG** (default) - Built-in, no download
   - **YOLO** (optional) - User-provided ONNX model
   - **None** (face-only) - No body detection

**Kalman Filter (Built-in):**
- **Not a machine learning model** - Pure mathematical algorithm
- **Implementation:** Plain NumPy (~microseconds per update)
- **File:** `src/person_tracking/kalman.py`
- **Purpose:** Track body box position and velocity over time
- **State vector:** `[center_x, center_y, width, height, velocity_x, velocity_y]`
- **Key features:**
  - Constant-velocity motion model
  - Predicts position when person not detected
  - Adaptive noise scaling (scales with box size)
  - Chi-squared gating for outlier rejection
  - Velocity damping during occlusion

**Key advantage:** Survives temporary face loss through confidence decay, Kalman filtering, and body tracking, making it better for surveillance scenarios where the person may turn away or be partially occluded.
