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

On Linux/macOS, create the environment with `python3 -m venv .venv`, activate with `source .venv/bin/activate`, then use `python` for the commands below. A graphical desktop and camera permissions are required. Install the `opencv-contrib-python` version from requirements.txt, not a headless variant.

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
python -m src.enroll --name "Belise" --samples 8
python -m src.enroll --name "Isaac" --camera 1
```

1. Keep exactly one person in view, with their whole face visible and good front lighting.
2. Wait for **Ready**, then press **Space** to capture each sample. Change expression and head angle slightly between captures.
3. After all samples are captured, enrollment saves automatically and closes the camera.

The default is eight captures, at least half a second apart. Faces must be at least 90 pixels across and pass a blur check. At least three samples are required. **Q**, **Escape**, or closing the main window cancels without saving partial enrollment. If needed, adjust `--min-face-size` or `--min-sharpness`; lowering them can admit poorer samples.

To deliberately redo an existing identity:

```sh
python -m src.enroll --name "Belise" --replace
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
Only the target gets a box; the window shows `SEARCHING`, `LOCKED`, `UNCERTAIN`, or `LOST`,
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
options are supported. A separate status panel displays position, SMILE/NEUTRAL,
eye state, session blink count, EAR, smile score, and all tracking/expression
calibration settings. Missing observations show N/A rather than stale readings.
Expression analysis runs only on the locked face, before any overlays are drawn.

Brief identity failures on a continuously detected face now retain an orange
`UNCERTAIN` box for up to five processed frames (`--uncertain-grace 5`). Each
new box must overlap the previous box by at least 50% IoU. Identity is checked
every frame during uncertainty; a successful target match restores `LOCKED`.
A different recognized identity, multiple eligible faces, invalid face alignment,
or an exhausted grace period drops the lock immediately. Missing detections
still hide the box and position; reacquisition after a gap requires a target match.
Use `--uncertain-grace 0` for the previous strict behavior. Grace can briefly
follow an unknown person occupying the same location; it is continuity evidence,
not identity confirmation. It cannot recover details absent in a dark camera image.

The panel shows the tracking reason, current detection confidence, and best
database match similarity (not necessarily the target's score). `Unknown` means
the match failed the threshold or margin. `N/A` means no valid score was obtained
on that frame, including frames between scheduled identity checks. Detection
confidence is shown only for eligible detections; no eligible face can also mean
the face is smaller than the 70-pixel minimum. These readings help distinguish
detection failures from identity matching failures when lighting changes.

### Tracking logs

Every tracking run automatically creates a unique CSV file under `data/logs/`
and prints its path. Use `--log-dir path/to/logs` to choose another directory.
Files have exactly three columns: `Timestamp`, `Action Type`, and `Description`.
Timestamps use UTC with an explicit timezone and millisecond precision.

Actions are `SESSION_START`, `TRACKING_SEARCHING`, `TRACKING_LOCKED`,
`TRACKING_UNCERTAIN`, `TRACKING_LOST`, `ERROR`, and `SESSION_END`. One tracking
row is saved per processed frame. Its JSON description contains the frame
number, target, tracking reason, detection confidence, match name and score,
position/direction, smile status and score, eye state, EAR, blink event, and
session blink count. Unavailable readings are `null`. The start record contains
the command settings; the end record distinguishes normal exit, interruption,
and error. Startup/runtime exceptions are recorded before being propagated.

Rows are flushed as they are written, and previous sessions are never overwritten.
Logs accumulate until manually archived or removed; recording every frame can
produce large files on long runs. Forced process termination may omit the final
session record. A log write failure stops the run with an error. The default log
directory is ignored by Git. Logs contain readings and names, not images or
face embeddings.

Install the updated dependencies before running this version. When upgrading an
older environment, replace its OpenCV package first to avoid two packages owning
the same `cv2` files:

```sh
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install --upgrade -r requirements.txt
python -m pip install --force-reinstall --no-deps opencv-contrib-python==4.11.0.86
```

The existing FaceMesh extractor uses MediaPipe 0.10.21 and compatible NumPy/OpenCV
versions pinned in requirements.txt. Tune `--ear-threshold` (0.21),
`--blink-min-frames` (2), `--blink-max-frames` (7), `--closed-frames` (8),
`--smile-on` (0.38), and `--smile-off` (0.35) using the displayed readings.
Blink duration is measured in processed frames, so tune for your actual frame rate.
Short low-EAR periods display EYES CLOSING; sustained closure displays EYES CLOSED.
Blink counts increment once upon reopening after 2-7 low-EAR frames. Missing
landmarks or a lost target reset pending expression state, retaining the session count.

## Whole-person tracking that survives face loss (Part 3)

`src/person_tracking/` keeps following the enrolled target when the face is not
visible (head turn, occlusion, someone walking in front) and says honestly how
much it still knows. It is CPU-only, reuses the Part 1/2 YuNet, ArcFace, aligner
and JSON gallery, and does not modify them.

```sh
python -m src.person_tracking --target "Isaac"                    # webcam
python -m src.person_tracking --target "Isaac" --video clip.mp4 --no-display
python -m src.person_tracking.sim --list                           # no camera, no models needed
python -m src.person_tracking.sim --scenario path_crossing --show
python -m src.person_tracking.sim.evaluate --seeds 5               # scores vs ground truth
python -m src.person_tracking.benchmark                            # per-stage CPU timings
```

### What is new compared with the Part 2 lock

Part 2 (`src.face_tracking`) and Part 3 (`src.person_tracking`) both have states with similar
names, but they answer different questions. Part 2 asks: *is the target's face in view and recognised
right now?* Part 3 asks: *where is the target's body, and how sure are we it is still the same person?*

| | Part 2 `face_tracking` | Part 3 `person_tracking` |
| --- | --- | --- |
| What is tracked | The face box | The whole body of every person, with the face attached to its body |
| Needs a detected face to output a position | Yes, every frame | No, only to *gain* or *refresh* confidence |
| Face disappears (head turn, hidden, looking down) | `LOST`: no box, no position, gives up after 24 frames | `TEMPORARY`: body still followed, position still output, confidence decays |
| Face visible but not recognised | `UNCERTAIN` (up to 5 frames) | Weak negative evidence on that track; not a separate state |
| Timing | Counted in frames (24 lost, 5 uncertain) | Seconds, so it behaves the same at any frame rate |
| What a state carries | A label | A label plus a confidence number that decays |
| Identity decision | A fresh ArcFace decision on each check | Evidence accumulates; one weak match cannot confirm |
| Other people in view | Mostly ignored | Tracked too, so crossings can be detected and swaps refused |
| ArcFace cost | Roughly every frame | About once a second for a held target |

Old `UNCERTAIN` and new `TEMPORARY` are nearly opposites: `UNCERTAIN` means the face is visible but
recognition is unsure, while `TEMPORARY` means the face is not visible at all but the body is still followed.
Part 2 is unchanged and still works; Part 3 is a separate module that reuses its detector, recognizer and
database.

### The three states

| State | Meaning | How it is reached | Aim output |
| --- | --- | --- | --- |
| `CONFIRMED` | A face matched the target recently enough to still be trusted | Positive face evidence only | Full gain |
| `TEMPORARY` | Was confirmed, face no longer seen; body followed by motion while confidence decays | Only from `CONFIRMED`, by decay | Reduced gain (0.35 to 1, in proportion to confidence) |
| `UNKNOWN` | No trustworthy claim: never confirmed, ruled out, or decayed away | Decay below `--unknown`, a different person's face, or a suspected swap | **None** (no centring command) |

```text
UNKNOWN --(face match, enough evidence)--> CONFIRMED --(no face, decay below 0.50)--> TEMPORARY
   ^                                           ^                                           |
   |                                           +--------(new face match, back to 0.70)-----+
   +-----------(decay below 0.20, other person's face, suspected swap)---------------------+
```

Confidence follows `c(t) = c0 * 0.5^(t / half_life)` between face observations
(`--half-life`, default 3 s), measured in seconds so it is frame-rate independent. With the defaults a
fully confident lock stays `CONFIRMED` for about 3 s without a face, then `TEMPORARY`, then `UNKNOWN` after
about 7 s. Decay is faster while the person is crossing another one (x2.5), coasting undetected (x1.5), or
predicted to have left the image (x4). Hysteresis (`--confirm` 0.70, drop 0.50, `--unknown` 0.20)
prevents flicker.

Rules that keep the states honest:

- **Only a face can raise confidence.** Following a body, however well, never does.
- **`TEMPORARY` can only be entered from `CONFIRMED`.** A track building up evidence from nothing stays
  `UNKNOWN` until it crosses the confirm threshold.
- **One weak match cannot confirm.** A single match confirms only from about 0.57 similarity (the
  threshold is 0.45); otherwise two agreeing matches are needed.
- **A different enrolled person's face** removes the lock after two strikes.
- **Only one track can hold the target** at a time; a new face match moves the lock rather than duplicating it.

### How the body is related to the face

No body is ever stored or recognised. The database holds face embeddings only, exactly as in Part 1.
The tracker works like this:

1. ArcFace recognises a face as the target (the same match as Part 1).
2. The face is mapped to a body by position: the face must sit where a head should be inside one body box.
3. That body's *track*, which lives only in memory while the program runs, is now labelled "target".
4. From then on the track is followed by motion (the Kalman filter below). The identity is never re-read
   from the body; it is a label carried from the last face match and decays until the next one.

The face-to-body score (`face_body.py`) is a Gaussian prior around the expected head position, which is
centred horizontally and about 8.5% of the body height below the top of the box. It is loose
horizontally (sigma 20% of body width, so turned shoulders still fit) and tight vertically (sigma 8% of body
height). It is combined with a size check (face height 5% to 30% of body height). A face whose centre is
outside the body horizontally or below the upper half of the body is rejected outright. All faces and bodies
are then matched together with the Hungarian algorithm, so one face goes to exactly one body. When a face fits
two overlapping bodies almost equally well (runner-up within 60% of the winner) it is flagged ambiguous and
**not used as identity evidence**, which stops a crossing from handing the target label to the wrong person.

The reverse also works: when the body detector finds nothing (common for close-up webcam framing), a body
box is built from the face, with a body height of the face height divided by 0.135. The tracker treats it as a
noisy measurement.

A small colour summary of each torso (72-bin HSV histogram) is learned live and kept in memory only. It is used
only as a soft association cue and to detect swaps after two people cross. It cannot say who someone is
and never confirms anyone. Consequently, if a track is lost completely (the person leaves, a long occlusion, or a
restart), that body cannot be recognised again: a new face match is required.

### The Kalman filter

Each body has its own constant-velocity Kalman filter (`kalman.py`, plain NumPy, about microseconds per update).
It exists so the box keeps moving sensibly between detector runs and when the person is hidden.

- **State:** `[cx, cy, w, h, vx, vy]`, the box centre, its size and the centre velocity.
- **Predict (every frame):** the box moves by its velocity times the elapsed time. Uncertainty grows with
  time, and the noise is scaled by the body size so a near and a far person behave the same.
- **Update (when a detection is matched):** the prediction and the detection are blended by their uncertainties,
  which also teaches the filter the person's speed.
- **Coasting (nothing measured):** the velocity is damped (rate 0.35 per second) so the box slows down instead
  of flying off during a long occlusion, while its uncertainty keeps growing. The tracker uses that
  uncertainty and the time without a measurement to report `predicted` evidence and to decay confidence faster.
- **Skipped detector frames:** the body detector runs every few frames to save CPU, and the Kalman prediction
  carries the box through the frames in between.
- **Outlier handling:** a measurement far from the prediction (a false positive or a wrong box) is down-weighted
  (chi-square 95%) or ignored (99.9%) instead of dragging the track.
- **Truncated boxes:** when an occluder hides part of a person, the visible sliver is not used directly, since
  its centre would bias position and speed. The tracker takes the full-size box nearest to the prediction that
  still contains the sliver.
- **Association:** each frame, tracks and detections are matched with the Hungarian algorithm using overlap,
  centre distance and clothing colour, with a gate that widens the longer a track has been unseen.

Practical effect: a person walking behind a wall keeps a moving `TEMPORARY` box at roughly the right place,
though the longer they stay hidden the less accurate the prediction (and the confidence) becomes.

### Pipeline

1. **Detect**: YuNet faces (on a frame at most `--detect-width` 640 px wide) and a body detector
   every `--detect-every` frames (`--person-backend hog` = OpenCV HOG, no download; `yolo` = any
   YOLOv5/v8 person ONNX via `--person-model`, e.g. a 320-pixel export for speed; `none` = bodies are
   derived from faces). On skipped frames the Kalman filter carries the boxes.
2. **Track bodies**: constant-velocity Kalman filter per person (size-scaled noise, damped coasting,
   robust to outliers), Hungarian assignment (no SciPy), colour signature as a soft association cue.
   Truncated boxes from occluders are completed rather than trusted, which avoids biasing the velocity.
3. **Map faces to bodies**: a Gaussian head-region prior plus face/body size plausibility,
   solved globally; faces that fit two overlapping bodies are flagged ambiguous and ignored as evidence.
4. **Verify sparingly**: ArcFace dominates CPU cost, so it runs on at most `--max-verifications` (1)
   frontal, large enough, unambiguous face per frame, and held targets are re-checked about once a second
   (about 0.1 calls/frame in the simulator versus 1-3 for a per-frame face lock).
5. **Crossings**: when two people overlap, both lose confidence faster; after they separate, torso colour
   recorded *before* the overlap is compared. If a track now looks like its former partner, its identity
   claim is withdrawn (never transferred); if colours cannot tell them apart, confidence is halved.
6. **Centre**: face centre when visible, estimated head position on the body otherwise (with a learned
   offset so the aim does not jump when the face disappears), time-constant smoothing, dead zone,
   and a gain below 1 while `TEMPORARY`. `UNKNOWN` returns no aim. To drive the ESP needle use
   `--mqtt-host`; `NeedleAdapter` maps CONFIRMED/TEMPORARY/UNKNOWN to the existing LOCKED/UNCERTAIN/SEARCHING
   payload without changing `src.mqtt_bridge`.

### Simulation and evaluation (no hardware)

`src/person_tracking/sim/` renders scripted people, walls and head turns, and derives realistic
detector, YuNet and ArcFace behaviour from ground truth (partial boxes when occluded, faces vanishing
as the head turns, similarity falling with yaw, a lookalike). The tracker never sees ground truth; both
systems under test receive identical observations. Scenarios: `head_turn`, `long_turn`, `occlusion`,
`long_occlusion`, `path_crossing`, `crossing_same_colour`, `leave_return`, `strangers_only`, `crowd`.
`--save x.mp4` or `--sheet x.png` records an annotated run. Reports are written to `data/reports/`.

Results (5 seeds, person detector every frame; the Part 2 style face lock is the baseline):

| Scenario | Coverage 3-state | Coverage face lock | During face loss | Wrong-hold | Ghost lock (s) |
| --- | --- | --- | --- | --- | --- |
| head_turn (5 s away) | 100% | 63% | 100% vs 0% | 0% | 0 |
| long_turn (12 s away) | 72% | 41% | 54% vs 0% | 0% | 0 |
| occlusion (wall) | 93% | 76% | 71% vs 0% | 7% | 0 |
| long_occlusion (9 s) | 46% | 35% | 16% vs 0% | 4% | 0 |
| path_crossing | 100% | 86% | 97% vs 0% | 0% | 0 |
| crossing_same_colour | 100% | 86% | 97% vs 0% | 0% | 0 |
| leave_return | 98% | 92% | 70% vs 0% | 0% | 1.6 |
| crowd (4 people) | 80% | 60% | 52% vs 1% | 0% | 0 |
| strangers_only (4% lookalike false matches) | n/a | n/a | n/a | n/a | 0.3 (face lock 0.5) |

With the target in view, no `CONFIRMED` frame pointed at the wrong person and there were no identity switches
(the only false locks are the lookalike "ghost" seconds above, when the target is absent). Where the tracker
is wrong it is almost entirely while `TEMPORARY` and `evidence: predicted` (the target hidden behind a wall
and extrapolated), with lower confidence than correct frames, which is the intended behaviour. Wall-occlusion
coverage depends on whether partial sliver detections occur, so treat single-scenario differences between
runs (for example `--detect-every 4`) as noise, not tuning signal.

Measured on a 20-thread Intel laptop CPU at 640x480 (`python -m src.person_tracking.benchmark`):
YuNet about 18 ms, HOG about 5 ms (every third frame), tracker about 1 ms, **ArcFace R100 about 108 ms
per face**. That gives roughly 33 FPS in the typical case versus about 8 FPS when ArcFace runs on
every frame. A 1280x720 video without faces ran at about 51 FPS through the whole CLI.

Limitations, stated plainly: the simulator's detectors are models of real ones, so the numbers show the tracking
logic, not real-camera accuracy; tune `--half-life`, `--min-face` and the thresholds on your own footage.
HOG rarely detects close-up webcam framing (the face-derived body is then used) and a stranger
who also matches the target above the threshold is accepted, exactly as in Part 1. Colour cues cannot separate
people in identical clothing, so the tracker withdraws confidence after such a crossing rather than guessing.
There is no liveness or anti-spoofing check. No real-person video was evaluated here; no ONNX person model is bundled.

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
validation/belise_01.jpg,belise
validation/isaac_01.jpg,isaac
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
    person_tracking/    Part 3: whole-person tracker, three-state identity, centring, simulator
  tests/                Automated mathematical/storage tests
  book/                 Reference material directory
```

Run modules with `python -m src.<module>` from the project root, rather than `python src/<module>.py`. Model and database defaults resolve relative to the project, not the shell's working directory.

## Tests

```sh
python -m unittest discover -s tests -v
python -m unittest tests.test_person_tracking -v   # Part 3 only; needs no camera or models
```

Tests cover known-transform recovery, invalid landmarks/embeddings, database round-trips, replacement, model mismatches, unknown rejection, ambiguity, and camera cleanup. No camera is required. Two additional smoke tests run real model inference when weights are present and skip otherwise.

Development validation: all 10 tests passed with Python 3.12.10, NumPy 2.5.3, OpenCV 4.14.0.94, and ONNX Runtime 1.29.0; `pip check` passed. Webcam 0 successfully returned a 640x480 frame. No face was visible during that check, so a real person's enrollment and recognition accuracy remain to be verified interactively.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `No Python at ...` | Recreate `.venv` using an installed Python; the original environment is not portable. |
| Missing/empty model | Run `python init_project.py --download-models`. A zero-byte file or Git LFS pointer is not a model. |
| Camera cannot open | Close Teams/Zoom/other camera programs, enable desktop camera permissions, or try `--camera 1`. |
| No window / GUI error | Run in a desktop session and install `opencv-contrib-python` from requirements.txt; remove conflicting OpenCV packages. |
| Everything is Unknown | Inspect landmarks/alignment, verify preprocessing, improve lighting, and re-enroll before lowering the threshold. |
| Wrong people accepted | Increase the threshold/margin and evaluate with held-out unknown people. |
| Model mismatch | Use the original model/settings or enroll again into a separate `--db` file. |
| Slow recognition | ResNet100 is a large CPU model; reduce people in view or supply a compatible smaller ArcFace model and re-enroll. |
| Corrupt JSON | Restore a backup or choose a new database path; invalid data is not silently discarded. |

## Reference and limitations

This project follows the modular enrollment/alignment/embedding/recognition architecture in Gabriel Baziramwabo's *Face Recognition with ArcFace ONNX and 5-Point Alignment*, supplied with this project. The implementation uses YuNet for measured landmarks, one JSON database, and held-out image evaluation; its commands and data format differ from the book's Haar/MediaPipe and NPZ examples. The existing `camera.py` and `detect.py` demos are preserved.

This is an educational recognition application with no liveness or anti-spoofing check. Enroll people with their consent and protect the local biometric database; it is not encrypted. Do not use a webcam match alone as an access-control decision. `.gitignore` covers new databases, images, and model downloads, but files already tracked by Git remain tracked: the original database and ArcFace placeholder are tracked in this repository, so inspect staged changes before committing real biometric data or large weights.
