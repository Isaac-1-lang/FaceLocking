"""A small 2D stage with scripted people, occluders, rendering and ground truth.

People are upright boxes (head on top) with distinct clothing, moving along
waypoints. Depth ordering follows the feet, so a person lower in the image
hides one higher up. Visibility is computed analytically from a low-resolution
label mask, which gives realistic *partial* boxes when someone is occluded.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720
BODY_ASPECT = 0.38          # width / height
HEAD_H, HEAD_W = 0.135, 0.100   # face box size as fractions of body height
_MASK_SCALE = 0.2


def _smooth(u):
    return u * u * (3 - 2 * u)


def _interp(keys, t):
    """Smoothstep interpolation of ``[(t, v0, v1, ...), ...]``; clamps outside the range."""
    if t <= keys[0][0]:
        return np.array(keys[0][1:], dtype=float)
    for a, b in zip(keys, keys[1:]):
        if t <= b[0]:
            u = (t - a[0]) / max(b[0] - a[0], 1e-9)
            return np.array(a[1:], dtype=float) * (1 - _smooth(u)) + np.array(b[1:], dtype=float) * _smooth(u)
    return np.array(keys[-1][1:], dtype=float)


@dataclass
class SimPerson:
    pid: int
    name: str                       # enrolled name, or a label for strangers
    enrolled: bool
    shirt: Tuple[int, int, int]     # BGR
    pants: Tuple[int, int, int]
    waypoints: List[tuple]          # (t, cx, cy, height)
    yaw_keys: List[tuple] = field(default_factory=lambda: [(0.0, 0.0)])  # (t, degrees); 0 = facing camera
    present: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 1e9)])
    lookalike_of: Optional[str] = None   # sometimes recognised as this enrolled person

    def pose(self, t):
        cx, cy, h = _interp(self.waypoints, t)
        return float(cx), float(cy), float(h)

    def yaw(self, t):
        return float(np.deg2rad(_interp(self.yaw_keys, t)[0]))

    def is_present(self, t):
        return any(a <= t <= b for a, b in self.present)

    def body_box(self, t):
        cx, cy, h = self.pose(t)
        w = BODY_ASPECT * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def head_box(self, t):
        cx, cy, h = self.pose(t)
        top = cy - h / 2
        return np.array([cx - HEAD_W * h / 2, top, cx + HEAD_W * h / 2, top + HEAD_H * h])

    def depth(self, t):
        return self.body_box(t)[3]


@dataclass
class Occluder:
    box: Tuple[float, float, float, float]
    active: Tuple[float, float] = (0.0, 1e9)
    color: Tuple[int, int, int] = (96, 96, 100)


@dataclass
class Scenario:
    name: str
    description: str
    duration: float
    persons: List[SimPerson]
    occluders: List[Occluder] = field(default_factory=list)
    fps: float = 15.0
    target: str = 'Isaac'
    lookalike_rate: float = 0.0


@dataclass
class PersonTruth:
    pid: int
    name: str
    box: np.ndarray            # full (un-occluded) body box
    head_box: np.ndarray
    center: np.ndarray
    head_center: np.ndarray
    yaw: float
    body_visibility: float     # visible pixels / full body area
    head_visibility: float
    visible_box: Optional[np.ndarray]   # bbox of visible pixels, full-res coords
    in_scene: bool             # present and at least partly inside the frame


class World:
    def __init__(self, scenario, seed=0):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        ramp = (70 + 40 * np.linspace(0, 1, HEIGHT)[:, None] + 12 * np.linspace(0, 1, WIDTH)[None, :])
        noise = np.random.default_rng(1234).normal(0, 3, (HEIGHT, WIDTH))
        gray = np.clip(ramp + noise, 0, 255).astype(np.uint8)
        self._background = np.ascontiguousarray(np.repeat(gray[..., None], 3, axis=2))
        self._mh, self._mw = int(HEIGHT * _MASK_SCALE), int(WIDTH * _MASK_SCALE)

    @property
    def n_frames(self):
        return int(self.scenario.duration * self.scenario.fps)

    # -- ground truth ---------------------------------------------------------
    def _active_occluders(self, t):
        return [o for o in self.scenario.occluders if o.active[0] <= t <= o.active[1]]

    def truth(self, t) -> List[PersonTruth]:
        persons = [p for p in self.scenario.persons if p.is_present(t)]
        labels = np.full((self._mh, self._mw), -1, dtype=np.int32)
        s = _MASK_SCALE
        for p in sorted(persons, key=lambda p: p.depth(t)):
            b = p.body_box(t) * s
            x1, y1 = max(int(np.floor(b[0])), 0), max(int(np.floor(b[1])), 0)
            x2, y2 = min(int(np.ceil(b[2])), self._mw), min(int(np.ceil(b[3])), self._mh)
            if x2 > x1 and y2 > y1:
                labels[y1:y2, x1:x2] = p.pid
        for o in self._active_occluders(t):
            x1, y1, x2, y2 = (np.array(o.box) * s).astype(int)
            labels[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = -2
        out = []
        for p in persons:
            box, head = p.body_box(t), p.head_box(t)
            cx, cy, h = p.pose(t)
            full_area = max((box[2] - box[0]) * (box[3] - box[1]) * s * s, 1e-9)
            mask = labels == p.pid
            visible = float(mask.sum()) / full_area
            vis_box = None
            if mask.any():
                ys, xs = np.nonzero(mask)
                vis_box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) / s
                vis_box = np.clip(vis_box, [box[0], box[1], 0, 0], [WIDTH, HEIGHT, box[2], box[3]])
                vis_box = np.minimum(vis_box, [WIDTH, HEIGHT, WIDTH, HEIGHT])
            hb = head * s
            hx1, hy1 = max(int(np.floor(hb[0])), 0), max(int(np.floor(hb[1])), 0)
            hx2, hy2 = min(int(np.ceil(hb[2])), self._mw), min(int(np.ceil(hb[3])), self._mh)
            head_area = max((hb[2] - hb[0]) * (hb[3] - hb[1]), 1e-9)
            head_vis = 0.0
            if hx2 > hx1 and hy2 > hy1:
                head_vis = float((labels[hy1:hy2, hx1:hx2] == p.pid).sum()) / head_area
            in_scene = (min(box[2], WIDTH) - max(box[0], 0) > 0.2 * (box[2] - box[0])
                        and min(box[3], HEIGHT) - max(box[1], 0) > 0.2 * (box[3] - box[1]))
            out.append(PersonTruth(
                p.pid, p.name, box, head,
                np.array([cx, cy]), np.array([(head[0] + head[2]) / 2, (head[1] + head[3]) / 2]),
                p.yaw(t), min(visible, 1.0), min(head_vis, 1.0), vis_box, in_scene))
        return out

    # -- rendering --------------------------------------------------------------
    def render(self, t):
        # Slow global brightness drift (exposure change), done in uint8 for speed.
        frame = cv2.convertScaleAbs(self._background, alpha=1.0 + 0.02 * np.sin(t * 1.7))
        persons = sorted((p for p in self.scenario.persons if p.is_present(t)),
                         key=lambda p: p.depth(t))
        for p in persons:
            cx, cy, h = p.pose(t)
            b = p.body_box(t)
            top, bottom = b[1], b[3]
            x1, x2 = int(b[0]), int(b[2])
            cv2.rectangle(frame, (x1, int(top + 0.48 * h)), (x2, int(bottom)), p.pants, -1)
            cv2.rectangle(frame, (x1, int(top + 0.14 * h)), (x2, int(top + 0.50 * h)), p.shirt, -1)
            head_c = (int(cx), int(top + 0.0675 * h))
            axes = (int(HEAD_W * h * 0.5 * 1.05), int(HEAD_H * h * 0.5))
            yaw = abs(np.rad2deg(p.yaw(t)))
            facing_away = yaw > 100
            cv2.ellipse(frame, head_c, axes, 0, 0, 360, (45, 40, 35) if facing_away else (150, 175, 215), -1)
            if not facing_away:
                shift = int(np.sin(p.yaw(t)) * axes[0] * 0.5)
                for dx in (-0.35, 0.35):
                    cv2.circle(frame, (head_c[0] + shift + int(dx * axes[0]), head_c[1] - 4), 2, (30, 30, 30), -1)
        for o in self._active_occluders(t):
            x1, y1, x2, y2 = (int(v) for v in o.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), o.color, -1)
        return frame
