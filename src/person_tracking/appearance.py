"""Tiny clothing-colour signature used only as *soft* evidence across crossings.

It can reveal that two tracks swapped places, but it never confirms identity.
A 72-bin HSV histogram of a 16x24 torso crop costs well under a millisecond.
"""
import cv2
import numpy as np

_BINS = (8, 3, 3)


def torso_histogram(frame, box):
    """L1-normalised HSV histogram of the torso, or ``None`` if the crop is unusable."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 16:
        return None
    left, right = int(max(0, x1 + 0.22 * w)), int(min(width, x2 - 0.22 * w))
    top, bottom = int(max(0, y1 + 0.20 * h)), int(min(height, y1 + 0.55 * h))
    if right - left < 4 or bottom - top < 4:
        return None
    crop = cv2.resize(frame[top:bottom, left:right], (16, 24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(_BINS), [0, 180, 0, 256, 0, 256]).ravel()
    total = hist.sum()
    return None if total <= 0 else (hist / total).astype(np.float32)


def similarity(a, b):
    """Bhattacharyya coefficient in [0, 1]; 1 means identical colour distributions."""
    if a is None or b is None:
        return None
    return float(np.sqrt(a * b).sum())


def blend(old, new, alpha=0.3):
    if old is None:
        return new
    if new is None:
        return old
    mixed = (1 - alpha) * old + alpha * new
    return mixed / mixed.sum()
