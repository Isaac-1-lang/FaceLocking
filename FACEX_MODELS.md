# FaceX Models Used in Face Tracking

## Summary

The `face_tracking.py` module (used for person tracking) uses **two models**:

### 1. Face Detection Model

**Model:** YuNet (OpenCV Face Detection)
- **File:** `models/detector_yunet.onnx`
- **Size:** 232,589 bytes (~227 KB)
- **Source:** [OpenCV Zoo YuNet 2023mar](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Purpose:** Detects faces in camera frames and provides 5 facial landmarks
- **Class:** `FaceDetector` in `src/landmarks.py`
- **Implementation:** Uses OpenCV's `cv2.FaceDetectorYN`

**Detection Output:**
- Bounding box: (x, y, width, height)
- 5 landmarks: left eye, right eye, nose tip, left mouth corner, right mouth corner
- Confidence score

**Parameters:**
- Minimum confidence: 0.85 (default)
- NMS threshold: 0.3
- Minimum face size: 70 pixels

### 2. Face Recognition Model

**Model:** ArcFace ResNet100
- **File:** `models/embedder_arcface.onnx`
- **Size:** 261,036,388 bytes (~249 MB)
- **Source:** [ONNX Model Zoo ArcFace ResNet100](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface)
- **Purpose:** Extracts 512-dimensional face embeddings for identity verification
- **Class:** `ArcFace` in `src/embed.py`
- **Implementation:** ONNX Runtime with CPU execution

**Embedding Process:**
1. Face detected by YuNet
2. Face aligned to 112×112 using 5-point similarity transform
3. Preprocessing: RGB float32, NCHW format, 0-255 range (default "raw")
4. Model inference produces 512-dimensional vector
5. Vector normalized to unit length
6. Cosine similarity matching against enrolled identities

**Preprocessing Options:**
- `--preprocessing raw` (default): RGB 0-255
- `--preprocessing normalized`: (RGB - 127.5) / 127.5

## Complete Face Tracking Pipeline

```
Camera Frame (640×480)
    ↓
YuNet Face Detector
    ↓ (detects faces + 5 landmarks)
Face Bounding Boxes
    ↓
5-Point Alignment Transform
    ↓ (112×112 aligned face)
ArcFace ResNet100 Embedder
    ↓ (512-dimensional vector)
Cosine Similarity Matching
    ↓ (compare to enrolled database)
Identity Lock + Position Tracking
    ↓
Horizontal Position (error_x)
    ↓
Angle Mapping (0-180°)
    ↓
MQTT → Arduino → Needle Movement
```

## Model Initialization in Code

**File:** `src/workflow.py`
```python
def models(args):
    return FaceDetector(args.detector), ArcFace(args.model, args.preprocessing)
```

**File:** `src/face_tracking.py`
```python
detector, embedder = models(args)
db = FaceDatabase(args.db, embedder.signature)
tracker = LockedFaceTracker(args.target, detector, embedder, db, ...)
```

## Command-Line Model Selection

You can override the default models when running face tracking:

```bash
python -m src.face_tracking \
    --target "YourName" \
    --detector models/detector_yunet.onnx \
    --model models/embedder_arcface.onnx \
    --preprocessing raw \
    --mqtt-host 127.0.0.1
```

**Options:**
- `--detector`: Path to YuNet ONNX model (default: `models/detector_yunet.onnx`)
- `--model`: Path to ArcFace ONNX model (default: `models/embedder_arcface.onnx`)
- `--preprocessing`: Either `raw` or `normalized` (default: `raw`)
- `--threshold`: Cosine similarity threshold (default: 0.45)
- `--margin`: Minimum margin over runner-up (default: 0.05)

## Performance Characteristics

### YuNet (Detector)
- **Speed:** Fast (~30-60 FPS on CPU)
- **Accuracy:** Good for frontal faces
- **Limitations:** Performance degrades with extreme angles, occlusions

### ArcFace ResNet100 (Embedder)
- **Speed:** Moderate (~10-20 FPS on CPU)
- **Accuracy:** High for face recognition
- **Limitations:** CPU-intensive, slower with multiple faces

**Overall System:** The embedder is the bottleneck. Recognition speed decreases with each additional detected face since each requires alignment and embedding extraction.

## Model Download

Models are downloaded automatically via:

```bash
python init_project.py --download-models
```

**Verification:**
- SHA-256 checksums verified
- Minimum file size check (1024 bytes)
- Existing verified models are reused

## Alternative Models

You can use different ArcFace models (e.g., ResNet50 for faster performance):

1. Replace `models/embedder_arcface.onnx` with your model
2. Ensure model accepts `[1, 3, 112, 112]` float32 input
3. Ensure model outputs 512-dimensional embeddings
4. Use appropriate `--preprocessing` flag
5. Re-enroll all identities (embeddings are model-specific)

## Expression Analysis (Optional)

**Model:** MediaPipe FaceMesh
- **Library:** `mediapipe==0.10.21`
- **Purpose:** Eye state (EAR) and smile detection
- **Class:** `FaceSignalExtractor` in `src/face_signals.py`
- **Usage:** Runs on the locked face for blink counting and smile detection

This is separate from the core YuNet+ArcFace pipeline and only activates when face tracking is running.

## Summary Table

| Component | Model | Size | Speed | Purpose |
|-----------|-------|------|-------|---------|
| **Detection** | YuNet | 227 KB | Fast | Find faces + landmarks |
| **Recognition** | ArcFace ResNet100 | 249 MB | Moderate | Identity verification |
| **Expression** | MediaPipe FaceMesh | N/A | Fast | Blink/smile (optional) |

## Key Points

1. **Two models work together:**
   - YuNet detects WHERE faces are
   - ArcFace determines WHO the person is

2. **Identity lock requires both:**
   - Detection must succeed (YuNet)
   - Recognition must match target (ArcFace)

3. **Tracking uses detection only:**
   - Once locked, tracks bounding box
   - Periodically verifies identity with ArcFace
   - Default: verify every frame (`--verify-every 1`)

4. **Position tracking is purely geometric:**
   - Uses face bounding box center
   - EMA smoothing for stability
   - No ML model involved in position calculation
