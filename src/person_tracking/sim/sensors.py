"""Synthetic stand-ins for the person detector, YuNet and ArcFace.

They are driven by ground truth but behave like the real components: boxes are
noisy and *partial* when someone is occluded, faces vanish when the head turns
or is hidden, recognition degrades with head angle, and strangers can score
close to the target. The tracker never sees ground truth; only ``FaceObs.meta``
carries the recognition result computed here so that every system under test
sees identical observations.
"""
from dataclasses import dataclass

import numpy as np

from ..records import Detection, FaceObs
from .world import BODY_ASPECT, HEAD_H, HEAD_W, HEIGHT, WIDTH


@dataclass
class SensorConfig:
    person_jitter: float = 0.02       # box noise, fraction of box size
    min_body_visibility: float = 0.35
    false_positive_rate: float = 0.01
    face_jitter: float = 0.03
    min_head_visibility: float = 0.85
    match_mean: float = 0.72          # genuine similarity at frontal pose
    match_std: float = 0.06
    yaw_penalty: float = 0.35         # similarity lost per unit of yaw_ratio
    stranger_mean: float = 0.15
    stranger_std: float = 0.07
    threshold: float = 0.45
    detect_every: int = 1             # person detector runs every N frames (CPU saving)


def _nms(dets, scores, thr=0.7):
    from ..records import iou
    order = np.argsort(scores)[::-1]
    keep = []
    for i in order:
        if all(iou(dets[i].box, dets[j].box) < thr for j in keep):
            keep.append(i)
    return [dets[i] for i in sorted(keep)]


class Sensors:
    def __init__(self, scenario, config=None, seed=0):
        self.scenario = scenario
        self.cfg = config or SensorConfig()
        self.rng = np.random.default_rng([seed, 7919])

    def _noisy(self, box, jitter):
        w, h = box[2] - box[0], box[3] - box[1]
        return box + self.rng.normal(0, jitter, 4) * np.array([w, h, w, h])

    def persons(self, truths, frame_index):
        """List of Detection, or ``None`` when the detector is skipped this frame."""
        cfg = self.cfg
        if frame_index % cfg.detect_every != 0:
            return None
        dets, scores = [], []
        for tr in truths:
            if tr.visible_box is None or tr.body_visibility < cfg.min_body_visibility:
                continue
            vis = tr.body_visibility
            p_detect = 0.97 if vis >= 0.7 else 0.97 * (0.25 + 0.75 * (vis - cfg.min_body_visibility) / 0.35)
            if self.rng.random() > p_detect:
                continue
            box = self._noisy(tr.visible_box, cfg.person_jitter)
            dets.append(Detection(box, float(tr.body_visibility)))
            scores.append(tr.body_visibility)
        if self.rng.random() < cfg.false_positive_rate:
            h = self.rng.uniform(380, 580)
            x = self.rng.uniform(0, WIDTH - BODY_ASPECT * h)
            y = self.rng.uniform(0, max(HEIGHT - h, 1))
            dets.append(Detection([x, y, x + BODY_ASPECT * h, y + h], 0.4))
            scores.append(0.4)
        return _nms(dets, scores) if dets else []

    def faces(self, truths):
        """Face observations; ``meta`` holds the hidden truth and the recognition result."""
        cfg = self.cfg
        faces = []
        for tr in truths:
            yaw_deg = abs(np.rad2deg(tr.yaw))
            if tr.head_visibility < cfg.min_head_visibility or yaw_deg >= 85 or not tr.in_scene:
                continue
            p_detect = 0.97 if yaw_deg <= 45 else 0.97 - (yaw_deg - 45) / 40 * 0.67
            if self.rng.random() > p_detect:
                continue
            h = tr.box[3] - tr.box[1]
            fw = HEAD_W * h * (0.55 + 0.45 * np.cos(np.deg2rad(yaw_deg)))
            fh = HEAD_H * h
            cx = tr.head_center[0] + np.sin(tr.yaw) * 0.08 * HEAD_W * h
            cy = tr.head_center[1]
            box = self._noisy(np.array([cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2]), cfg.face_jitter)
            eye_dist = 0.44 * fw * np.cos(np.deg2rad(yaw_deg)) + 1e-3
            mid = cx
            nose_dx = 0.40 * fw * np.sin(tr.yaw)
            points = np.array([[mid - eye_dist / 2, cy - 0.12 * fh], [mid + eye_dist / 2, cy - 0.12 * fh],
                               [mid + nose_dx, cy + 0.08 * fh], [mid - eye_dist * 0.4, cy + 0.28 * fh],
                               [mid + eye_dist * 0.4, cy + 0.28 * fh]])
            obs = FaceObs(box, points, 0.95)
            obs.meta = {'pid': tr.pid, 'recog': self._recognize(tr, obs)}
            faces.append(obs)
        return faces

    def _recognize(self, truth, obs):
        cfg, sc = self.cfg, self.scenario
        person = next(p for p in sc.persons if p.pid == truth.pid)
        if person.lookalike_of and self.rng.random() < sc.lookalike_rate:
            return person.lookalike_of, float(np.clip(self.rng.normal(0.52, 0.03), 0, 1))
        if person.enrolled:
            score = self.rng.normal(cfg.match_mean, cfg.match_std) - cfg.yaw_penalty * obs.yaw_ratio
            if obs.size < 60:
                score -= 0.08
            score = float(np.clip(score, -1, 1))
            return (person.name if score >= cfg.threshold else 'Unknown'), score
        return 'Unknown', float(np.clip(self.rng.normal(cfg.stranger_mean, cfg.stranger_std), -1, 1))
