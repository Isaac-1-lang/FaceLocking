"""CPU person detectors: zero-download HOG, optional ONNX YOLO, and a frame scheduler.

``None`` is returned (not an empty list) when a detector is skipped to save CPU,
so the tracker can tell "nobody found" from "did not look".
"""
import os
from pathlib import Path

import cv2
import numpy as np

from .records import Detection


class HogPersonDetector:
    """OpenCV's built-in HOG + linear SVM pedestrian detector. No model file needed.

    Best for standing people seen head-to-toe at a moderate distance. It rarely fires
    on close-up webcam framing; the tracker then falls back to boxes derived from
    faces, so tracking still works.
    """

    def __init__(self, scale=0.5, hit_threshold=0.0, min_score=0.5):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.scale, self.hit_threshold, self.min_score = scale, hit_threshold, min_score

    def detect(self, frame):
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        rects, weights = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.08, hitThreshold=self.hit_threshold)
        if len(rects) == 0:
            return []
        weights = np.asarray(weights).ravel()
        keep = cv2.dnn.NMSBoxes([list(map(int, r)) for r in rects], weights.tolist(),
                                self.min_score, 0.4)
        out = []
        for i in np.asarray(keep).ravel():
            x, y, w, h = rects[i] / self.scale
            # HOG windows are padded; shrink to the body itself.
            out.append(Detection([x + 0.15 * w, y + 0.06 * h, x + 0.85 * w, y + 0.94 * h],
                                 float(weights[i])))
        return out


def decode_yolo(output, ratio, pad, confidence=0.4, nms=0.5, person_class=0):
    """Decode a YOLOv8-style ``[1, 4+C, N]`` or YOLOv5/X-style ``[1, N, 5+C]`` tensor to person boxes.

    ``ratio``/``pad`` undo the letterbox: original = (letterboxed - pad) / ratio.
    """
    out = np.squeeze(np.asarray(output))
    if out.ndim != 2:
        raise ValueError(f'Unexpected detector output shape {np.shape(output)}')
    if out.shape[0] < out.shape[1] and out.shape[0] <= 128:   # channels-first -> rows
        out = out.T
    cols = out.shape[1]
    if cols > 84 and (cols - 5) in (1, 80, 91):   # objectness column present (YOLOv5 / YOLOX)
        scores = out[:, 4] * out[:, 5 + person_class]
    else:                                         # YOLOv8: class scores directly after the box
        scores = out[:, 4 + person_class]
    keep = scores >= confidence
    if not keep.any():
        return []
    boxes, scores = out[keep, :4], scores[keep]
    xyxy = np.column_stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                            boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2])
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio
    wh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy]
    idx = np.asarray(cv2.dnn.NMSBoxes(wh, scores.tolist(), confidence, nms)).ravel()
    return [Detection(xyxy[i], float(scores[i])) for i in idx]


class OnnxPersonDetector:
    """YOLO-family person detector through ONNX Runtime on the CPU.

    Pass any YOLOv5/YOLOv8-style export with 640 (or smaller, e.g. 320 for speed)
    square input; only the 'person' class (index 0, COCO) is used. No model is
    bundled; see the README for how to supply one.
    """

    def __init__(self, model, size=None, confidence=0.4, threads=None):
        path = Path(model)
        if not path.is_file() or path.stat().st_size < 1024:
            raise ValueError(f'Missing or empty person model: {path}')
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or max(1, min(4, (os.cpu_count() or 2) // 2))
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        shape = self.session.get_inputs()[0].shape
        fixed = shape[2] if isinstance(shape[2], int) else None
        self.size = int(size or fixed or 640)
        self.name = self.session.get_inputs()[0].name
        self.confidence = confidence

    def detect(self, frame):
        h, w = frame.shape[:2]
        ratio = self.size / max(h, w)
        nw, nh = int(round(w * ratio)), int(round(h * ratio))
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.name: np.ascontiguousarray(blob)})[0]
        return decode_yolo(output, ratio, (px, py), self.confidence)


class Scheduled:
    """Runs a detector every ``every`` frames; other frames return ``None``."""

    def __init__(self, detector, every=3):
        if every < 1:
            raise ValueError('every must be >= 1')
        self.detector, self.every, self.count = detector, every, 0

    def detect(self, frame):
        run = self.count % self.every == 0
        self.count += 1
        return self.detector.detect(frame) if run else None


def make_person_detector(kind, model=None, every=3):
    """``kind``: 'hog', 'yolo', or 'none' (faces alone drive the body boxes)."""
    if kind == 'none':
        return None
    if kind == 'hog':
        return Scheduled(HogPersonDetector(), every)
    if kind == 'yolo':
        if not model:
            raise ValueError('--person-model is required for the yolo backend')
        return Scheduled(OnnxPersonDetector(model), every)
    raise ValueError(f'Unknown person backend {kind!r}')
