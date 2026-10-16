"""Target-centring logic: turns a TargetEstimate into a smoothed aim error.

The aim point is the face when it is visible and the *estimated head position
on the body* otherwise. A per-track offset between the two is learned while
both are available, so when the face disappears (head turn) the aim point does
not jump. Output is gated by the identity state: UNKNOWN produces no command.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .face_body import head_anchor
from .identity import IdentityState
from .records import box_center


@dataclass
class Aim:
    error_x: float            # [-1, 1], negative = target left of centre (same convention as Part 2)
    error_y: float
    horizontal: str           # LEFT / CENTER / RIGHT after the dead zone
    vertical: str
    source: str               # 'face' | 'body-head' | 'predicted'
    gain: float               # 1.0 CONFIRMED, scaled down by confidence when TEMPORARY
    angle: float              # needle angle 0..180, 90 = centre (matches src.mqtt_bridge.angle_for)
    point: np.ndarray         # smoothed aim point in pixels


class CenteringController:
    def __init__(self, dead_zone=0.07, smooth_tau=0.20, offset_alpha=0.15,
                 temporary_min_gain=0.35, confirm_threshold=0.70):
        if not 0 <= dead_zone < 1 or smooth_tau < 0 or not 0 < offset_alpha <= 1:
            raise ValueError('Invalid centring parameters')
        self.dead_zone, self.smooth_tau = dead_zone, smooth_tau
        self.offset_alpha, self.temporary_min_gain = offset_alpha, temporary_min_gain
        self.confirm_threshold = confirm_threshold
        self.point = None
        self.track_id = None
        self.offset = np.zeros(2)
        self.t = None

    def reset(self):
        self.point, self.track_id, self.t = None, None, None
        self.offset = np.zeros(2)

    def update(self, estimate, shape, t) -> Optional[Aim]:
        if estimate.state == IdentityState.UNKNOWN or estimate.body_box is None:
            self.reset()
            return None
        if estimate.track_id != self.track_id:
            self.reset()
            self.track_id = estimate.track_id
        anchor = head_anchor(estimate.body_box)
        if estimate.face_box is not None:
            face_center = box_center(estimate.face_box)
            self.offset = (face_center - anchor if not self.offset.any()
                           else (1 - self.offset_alpha) * self.offset
                           + self.offset_alpha * (face_center - anchor))
            raw, source = face_center, 'face'
        else:
            raw = anchor + self.offset
            source = 'predicted' if estimate.evidence == 'predicted' else 'body-head'
        if self.point is None or self.t is None or self.smooth_tau <= 0:
            self.point = raw.astype(float)
        else:
            alpha = 1.0 - np.exp(-max(t - self.t, 0.0) / self.smooth_tau)
            self.point = self.point + alpha * (raw - self.point)
        self.t = t
        height, width = shape[:2]
        ex = float(np.clip((self.point[0] - width / 2) / (width / 2), -1, 1))
        ey = float(np.clip((self.point[1] - height / 2) / (height / 2), -1, 1))
        dz = self.dead_zone
        if estimate.state == IdentityState.CONFIRMED:
            gain = 1.0
        else:
            gain = float(np.clip(estimate.confidence / self.confirm_threshold,
                                 self.temporary_min_gain, 1.0))
        return Aim(ex, ey,
                   'LEFT' if ex < -dz else 'RIGHT' if ex > dz else 'CENTER',
                   'UP' if ey < -dz else 'DOWN' if ey > dz else 'CENTER',
                   source, gain, 90.0 + ex * 90.0, self.point.copy())
