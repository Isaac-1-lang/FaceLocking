"""Plain data records and box geometry shared by the whole-person tracker.

All boxes are float ``(x1, y1, x2, y2)``. (YuNet's own ``Face.box`` is
``x, y, w, h``; use ``FaceObs.from_face`` to convert.)
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def as_box(box):
    return np.asarray(box, dtype=np.float64).reshape(4)


def box_center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def box_size(box):
    return float(max(box[2] - box[0], 0.0)), float(max(box[3] - box[1], 0.0))


def box_area(box):
    w, h = box_size(box)
    return w * h


def intersection(a, b):
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def iou(a, b):
    inter = intersection(a, b)
    return inter / max(box_area(a) + box_area(b) - inter, 1e-9)


def overlap_min(a, b):
    """Intersection over the smaller box: 1.0 when one box hides inside the other."""
    return intersection(a, b) / max(min(box_area(a), box_area(b)), 1e-9)


def iou_matrix(boxes_a, boxes_b):
    out = np.zeros((len(boxes_a), len(boxes_b)))
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            out[i, j] = iou(a, b)
    return out


def clip_box(box, shape):
    height, width = shape[:2]
    return np.array([np.clip(box[0], 0, width - 1), np.clip(box[1], 0, height - 1),
                     np.clip(box[2], 0, width - 1), np.clip(box[3], 0, height - 1)])


@dataclass
class Detection:
    """A person (whole-body) box. ``source`` is 'person' or 'face' (derived from a face)."""
    box: np.ndarray
    score: float = 1.0
    source: str = 'person'

    def __post_init__(self):
        self.box = as_box(self.box)


@dataclass
class FaceObs:
    """A detected face. ``meta`` is opaque (the simulator hides ground truth there)
    and is never read by the tracker."""
    box: np.ndarray
    points: Optional[np.ndarray] = None
    confidence: float = 1.0
    meta: Any = field(default=None, repr=False)

    def __post_init__(self):
        self.box = as_box(self.box)
        if self.points is not None:
            self.points = np.asarray(self.points, dtype=np.float64).reshape(5, 2)

    @classmethod
    def from_face(cls, face):
        x, y, w, h = (float(v) for v in face.box)
        return cls(np.array([x, y, x + w, y + h]), face.points, float(face.confidence))

    @property
    def center(self):
        return box_center(self.box)

    @property
    def size(self):
        return min(box_size(self.box))

    @property
    def yaw_ratio(self):
        """|nose offset| / eye distance: ~0 frontal, >0.5 clearly turned.

        Without landmarks the face is assumed frontal.
        """
        if self.points is None:
            return 0.0
        eye_a, eye_b, nose = self.points[0], self.points[1], self.points[2]
        eye_dist = float(np.linalg.norm(eye_b - eye_a))
        if eye_dist < 1e-6:
            return 9.0
        return float(abs(nose[0] - (eye_a[0] + eye_b[0]) / 2.0) / eye_dist)


@dataclass
class TargetEstimate:
    """What the module reports for the locked target on one frame.

    ``state`` is an ``IdentityState`` value; ``confidence`` the decayed belief
    that ``track_id`` is the target. ``evidence`` says how the position was
    obtained: 'face' (face visible now), 'body' (body detected, face not
    visible) or 'predicted' (coasting on the motion model, nothing seen).
    """
    state: Any
    confidence: float
    track_id: Optional[int] = None
    body_box: Optional[np.ndarray] = None
    face_box: Optional[np.ndarray] = None
    evidence: str = 'none'
    reason: str = ''
    seconds_since_face: Optional[float] = None
