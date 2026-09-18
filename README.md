# FaceX

FaceX is a local webcam face recognition project. It detects faces and five facial landmarks with YuNet, aligns each face to 112 x 112 pixels, extracts a 512-dimensional ArcFace embedding, and compares it with enrolled people. It runs on the CPU and displays names or `Unknown` in an OpenCV window.

## Quick start (Windows PowerShell)

Install 64-bit Python 3.11 or 3.12. Run these commands from the FaceX folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe init_project.py --download-models
.\.venv\Scripts\python.exe -m src.camera
.\.venv\Scripts\python.exe -m src.enroll --name "Your Name"
.\.venv\Scripts\python.exe -m src.recognize
```

Close each camera window with **Q** before starting the next command. Explicit interpreter paths avoid PowerShell activation-policy issues. The original `venv` in this checkout points to a missing Python installation; create a fresh `.venv` instead of reusing it.

A project-local Python runtime may also be present from development. If `.tools/python/python.exe` exists, use it directly in place of `.venv/Scripts/python.exe`; it does not require activation. `.tools` is ignored by Git and is not part of the portable project.

On Linux/macOS, create the environment with `python3 -m venv .venv`, activate with `source .venv/bin/activate`, then use `python` for the commands below. A graphical desktop and camera permissions are required. Install `opencv-python`, not its headless variant.

## Model setup

```sh
python init_project.py --download-models
```

This downloads YuNet (232,589 bytes) and ArcFace ResNet100 (261,036,388 bytes), verifies their SHA-256 checksums, and installs them under `models/`. Internet is needed only for installing dependencies and downloading models. Existing verified models are reused; different nonempty models are preserved with an error. Empty placeholders are replaced. Without `--download-models`, setup only creates directories.

| File | Source |
| --- | --- |
| `models/detector_yunet.onnx` | [OpenCV Zoo YuNet 2023mar](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `models/embedder_arcface.onnx` | [ONNX Model Zoo ArcFace ResNet100](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface) |

Default preprocessing is **RGB float32, NCHW, 0-255** (`--preprocessing raw`) for this Model Zoo export. For a compatible external ArcFace model that expects `(RGB - 127.5) / 127.5`, explicitly pass `--preprocessing normalized` to enrollment, evaluation, and recognition. This includes the book's `w600k_r50.onnx` setup. Do not infer preprocessing from the `.onnx` filename. The model must accept `[1,3,112,112]` float32 input and emit 512 features. See the upstream model documentation and licenses before redistributing weights.

## Enroll a person

```sh
python -m src.enroll --name "Alice" --samples 8
python -m src.enroll --name "Bob" --camera 1
```

1. Keep exactly one person in view, with their whole face visible and good front lighting.
2. Wait for **Ready**, then press **Space** to capture each sample. Change expression and head angle slightly between captures.
3. After all samples are captured, enrollment saves automatically and closes the camera.

The default is eight captures, at least half a second apart. Faces must be at least 90 pixels across and pass a blur check. At least three samples are required. **Q**, **Escape**, or closing the main window cancels without saving partial enrollment. If needed, adjust `--min-face-size` or `--min-sharpness`; lowering them can admit poorer samples.

To deliberately redo an existing identity:

```sh
python -m src.enroll --name "Alice" --replace
```

Names are case-sensitive; `Unknown` and `Too small` are reserved. Embeddings are stored in `data/db/face_db.json`; raw photos and camera recordings are not saved. Each identity retains its normalized samples; matching uses their normalized mean. Writes replace the JSON file atomically. Run one enrollment process at a time. Restart recognition after adding or replacing identities.

The database records the model hash, preprocessing, and alignment version. A mismatch requires re-enrollment into a new database, preventing comparisons between incompatible embeddings. Use `--db data/db/another.json` consistently to keep separate galleries. Back up the database before manually editing or removing identities.

## Live recognition

```sh
python -m src.recognize --camera 0 --threshold 0.45 --margin 0.05
```

The window displays face boxes, five landmark dots, names, cosine similarity, and processing FPS. Press **Q** or **Escape** to quit. Multiple faces can be processed; CPU speed decreases with each additional face. Faces below 70 pixels across are marked `Too small`.

The best match must meet `--threshold` and exceed the runner-up by `--margin`; otherwise the result is `Unknown`. Higher thresholds accept fewer matches. Scores are similarities, not confidence percentages. The defaults are starting values, not calibrated accuracy guarantees. This implementation processes current frames without temporal identity tracking, so labels may flicker under poor lighting or motion.

## Lock and track one enrolled person

```sh
python -m src.face_tracking --target "Isaac"
```

Use the exact enrolled name. This reuses the same YuNet detection, five-point
alignment, ArcFace model, JSON database, and similarity threshold/margin as
`src.recognize`. Unknown faces and other enrolled names cannot acquire the lock.
Only the target gets a box; the window shows `SEARCHING`, `LOCKED`, or `LOST`,
plus the smoothed face position and normalized horizontal/vertical errors.
There is no current box or position output while the target is missing.
After more than 24 missed frames, it clears the old geometry and searches for
the same identity again. Q, Escape, or closing the window exits.

Identity is checked every frame by default. `--verify-every 10` uses the PDF's
lighter periodic verification, but can briefly follow an unverified face between
checks. Reappearance after a gap and scenes with multiple eligible faces always
trigger verification. `--lost-timeout`, `--ema-alpha`, and `--dead-zone` control
the grace period, position smoothing, and centered region. Existing `--camera`,
`--db`, `--model`, `--detector`, `--preprocessing`, `--threshold`, and `--margin`
options are supported. This command implements identity locking and position;
smile/blink detection is not enabled.

## Inspect individual stages

```sh
python -m src.camera       # Original camera and FPS demo
python -m src.detect       # Original Haar box detector demo
python -m src.landmarks    # YuNet boxes and five points, plus aligned preview
python -m src.alignment    # Same preview for inspecting 112x112 alignment
python -m src.embed        # Live embedding dimension and unit norm
```

The original camera/Haar demos retain their `CAMERA_INDEX` constants. The new webcam commands accept `--camera`, `--detector`, `--model`, `--db`, and `--preprocessing` where applicable; use `--help` for details. `python -m src.harr_5pt` is also a landmark preview, preserving the original filename. Haar cascades alone do not provide the five measured landmarks used by the recognition pipeline.

## Evaluate with held-out photos

Collect fresh images in the expected camera conditions, including enrolled people and people absent from the gallery. Use different captures from enrollment. Create a CSV such as `data/evaluation.csv`:

```csv
path,label
validation/alice_01.jpg,Alice
validation/bob_01.jpg,Bob
validation/visitor_01.jpg,Unknown
```

Paths are relative to the CSV. Each image must contain exactly one face. `Unknown` is the label for any unenrolled person.

```sh
python -m src.evaluate --manifest data/evaluation.csv --threshold 0.45
python -m src.evaluate --manifest data/evaluation.csv --threshold 0.55
```

Compare correct predictions, false accepts of unknown people, false rejects of known people, and wrong identities. Read/detection failures are reported separately and count against total accuracy. Tune on a validation set, then report performance on a separate test set. No evaluation photos or actual enrollments are bundled.

## Structure

```text
FaceX/
  init_project.py        Directory setup and verified model downloads
  requirements.txt      Runtime dependencies
  models/               YuNet and ArcFace ONNX weights
  data/db/              Local enrollment database
  src/
    camera.py           Existing webcam demo
    detect.py           Existing Haar detector demo
    landmarks.py        YuNet detector, Face records, drawing
    alignment.py        Five-point similarity transform
    embed.py            ONNX inference and normalization
    database.py         Storage, model compatibility, cosine matching
    enroll.py           Quality-gated webcam enrollment
    recognize.py        Webcam recognition
    evaluate.py         Held-out image evaluation
    harr_5pt.py          Landmark/alignment preview
    config.py           Project-relative paths
    workflow.py         CLI helpers and camera cleanup
  tests/                Automated mathematical/storage tests
  book/                 Reference material directory
```

Run modules with `python -m src.<module>` from the project root, rather than `python src/<module>.py`. Model and database defaults resolve relative to the project, not the shell's working directory.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests cover known-transform recovery, invalid landmarks/embeddings, database round-trips, replacement, model mismatches, unknown rejection, ambiguity, and camera cleanup. No camera is required. Two additional smoke tests run real model inference when weights are present and skip otherwise.

Development validation: all 10 tests passed with Python 3.12.10, NumPy 2.5.3, OpenCV 4.14.0.94, and ONNX Runtime 1.29.0; `pip check` passed. Webcam 0 successfully returned a 640x480 frame. No face was visible during that check, so a real person's enrollment and recognition accuracy remain to be verified interactively.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `No Python at ...` | Recreate `.venv` using an installed Python; the original environment is not portable. |
| Missing/empty model | Run `python init_project.py --download-models`. A zero-byte file or Git LFS pointer is not a model. |
| Camera cannot open | Close Teams/Zoom/other camera programs, enable desktop camera permissions, or try `--camera 1`. |
| No window / GUI error | Run in a desktop session and install `opencv-python`; remove conflicting headless OpenCV packages. |
| Everything is Unknown | Inspect landmarks/alignment, verify preprocessing, improve lighting, and re-enroll before lowering the threshold. |
| Wrong people accepted | Increase the threshold/margin and evaluate with held-out unknown people. |
| Model mismatch | Use the original model/settings or enroll again into a separate `--db` file. |
| Slow recognition | ResNet100 is a large CPU model; reduce people in view or supply a compatible smaller ArcFace model and re-enroll. |
| Corrupt JSON | Restore a backup or choose a new database path; invalid data is not silently discarded. |

## Reference and limitations

This project follows the modular enrollment/alignment/embedding/recognition architecture in Gabriel Baziramwabo's *Face Recognition with ArcFace ONNX and 5-Point Alignment*, supplied with this project. The implementation uses YuNet for measured landmarks, one JSON database, and held-out image evaluation; its commands and data format differ from the book's Haar/MediaPipe and NPZ examples. The existing `camera.py` and `detect.py` demos are preserved.

This is an educational recognition application with no liveness or anti-spoofing check. Enroll people with their consent and protect the local biometric database; it is not encrypted. Do not use a webcam match alone as an access-control decision. `.gitignore` covers new databases, images, and model downloads, but files already tracked by Git remain tracked: the original database and ArcFace placeholder are tracked in this repository, so inspect staged changes before committing real biometric data or large weights.
