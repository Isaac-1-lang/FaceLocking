"""YuNet face boxes and measured five-point landmarks (image-left first)."""
from dataclasses import dataclass
import cv2
import numpy as np
from .config import DETECTOR, require_model


@dataclass
class Face:
    box: np.ndarray
    points: np.ndarray
    confidence: float


class FaceDetector:
    def __init__(self, model=DETECTOR, confidence=0.85):
        self.net = cv2.FaceDetectorYN.create(str(require_model(model)), "", (320, 320),
                                             confidence, 0.3, 5000)

    def detect(self, frame):
        self.net.setInputSize((frame.shape[1], frame.shape[0]))
        _, rows = self.net.detect(frame)
        if rows is None:
            return []
        return [Face(row[:4].copy(), row[4:14].reshape(5, 2).copy(), float(row[14]))
                for row in rows if np.isfinite(row).all()]


def draw_face(frame, face, label="", color=(0, 220, 0)):
    x, y, w, h = np.rint(face.box).astype(int)
    # A restrained outline keeps the face visible; corners emphasize the bounds.
    edge = max(1, min(20, int(min(w, h) * 0.15)))
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1, cv2.LINE_AA)
    for px, py, dx, dy in ((x, y, 1, 1), (x+w, y, -1, 1),
                          (x, y+h, 1, -1), (x+w, y+h, -1, -1)):
        corner = np.array(((px, py+dy*edge), (px, py), (px+dx*edge, py)),
                          dtype=np.int32)
        cv2.polylines(frame, [corner], False, color, 2, cv2.LINE_AA)
    for point in face.points:
        center = tuple(np.rint(point).astype(int))
        cv2.circle(frame, center, 4, (28, 32, 38), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 2, (0, 200, 255), -1, cv2.LINE_AA)
    if label:
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.48
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, 1)
        height, width = frame.shape[:2]
        padding = 6
        tag_h = text_h + baseline + 2*padding
        left = max(0, min(x, width-text_w-2*padding))
        top = max(0, min(y-tag_h-4 if y >= tag_h+4 else y+4, height-tag_h))
        cv2.rectangle(frame, (left, top), (left+text_w+2*padding, top+tag_h),
                      (28, 32, 38), -1)
        cv2.line(frame, (left, top), (left, top+tag_h), color, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (left+padding, top+padding+text_h),
                    font, scale, (240, 243, 246), 1, cv2.LINE_AA)


if __name__ == "__main__":
    from .harr_5pt import main
    from .workflow import run
    run(main)
