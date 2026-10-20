"""Constant-velocity Kalman filter for a body box, in plain NumPy (CPU, ~microseconds)."""
import numpy as np

from .records import as_box


class BoxKalman:
    """State ``[cx, cy, w, h, vx, vy]``; measurement ``[cx, cy, w, h]``.

    Noise scales with the box size so the filter behaves the same for a person
    near or far from the camera. While nothing is measured the velocity is
    damped, so a coasting box slows down instead of flying off during a long
    occlusion, and its covariance (``uncertainty``) keeps growing.
    """

    GATE_CHI2 = 9.49        # 95% chi-square quantile, 4 degrees of freedom: start down-weighting
    REJECT_CHI2 = 18.5      # 99.9% quantile: a measurement this inconsistent is ignored

    def __init__(self, box, accel_std=1.5, size_std=0.25, meas_std=0.04, damping=0.35):
        box = as_box(box)
        w, h = box[2] - box[0], box[3] - box[1]
        self.x = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, w, h, 0.0, 0.0])
        self.P = np.diag([(0.1 * w) ** 2, (0.1 * h) ** 2, (0.1 * w) ** 2, (0.1 * h) ** 2,
                          (0.8 * w) ** 2, (0.8 * h) ** 2])
        self.accel_std, self.size_std = accel_std, size_std
        self.meas_std, self.damping = meas_std, damping

    @property
    def box(self):
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @property
    def center(self):
        return self.x[:2].copy()

    @property
    def velocity(self):
        return self.x[4:6].copy()

    @property
    def uncertainty(self):
        """Position standard deviation relative to body height (dimensionless)."""
        return float(np.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0)) / max(self.x[3], 1.0))

    def predict(self, dt, coasting=False):
        if dt <= 0:
            return
        w, h = self.x[2], self.x[3]
        F = np.eye(6)
        F[0, 4] = F[1, 5] = dt
        if coasting:
            F[4, 4] = F[5, 5] = float(np.exp(-self.damping * dt))
        q = np.zeros((6, 6))
        for axis, scale in ((0, w), (1, h)):
            sa2 = (self.accel_std * scale) ** 2
            q[axis, axis] = sa2 * dt ** 4 / 4
            q[axis, axis + 4] = q[axis + 4, axis] = sa2 * dt ** 3 / 2
            q[axis + 4, axis + 4] = sa2 * dt ** 2
        q[2, 2] = (self.size_std * w * dt) ** 2
        q[3, 3] = (self.size_std * h * dt) ** 2
        self.x = F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 4.0)
        self.P = F @ self.P @ F.T + q

    def update(self, box, noise_scale=1.0):
        box = as_box(box)
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                      box[2] - box[0], box[3] - box[1]])
        w, h = max(self.x[2], 1.0), max(self.x[3], 1.0)
        r = self.meas_std * noise_scale
        R = np.diag([(r * w) ** 2, (r * h) ** 2, (r * w) ** 2, (r * h) ** 2])
        H = np.zeros((4, 6))
        H[:4, :4] = np.eye(4)
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        # Robust update: an innovation far outside what the model predicts (false positive,
        # wrong box) is ignored; a borderline one is down-weighted. Returns whether it was used.
        d2 = float(innovation @ np.linalg.solve(S, innovation))
        if d2 > self.REJECT_CHI2:
            return False
        if d2 > self.GATE_CHI2:
            S = H @ self.P @ H.T + R * (d2 / self.GATE_CHI2)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P
        self.x[2:4] = np.maximum(self.x[2:4], 4.0)
        return True
