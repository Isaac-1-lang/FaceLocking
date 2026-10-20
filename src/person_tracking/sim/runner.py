"""Run a tracking system over a scenario and score it against ground truth.

Two systems see byte-identical observations (same boxes, faces and recognition
results), so differences come only from the tracking logic:

* ``three_state`` - this module's WholePersonTracker + CenteringController.
* ``face_lock``   - the Part 2 behaviour: lock exists only on frames where the
                    target's face is detected and recognised; otherwise LOST.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from ..centering import CenteringController
from ..face_body import head_anchor
from ..identity import IdentityState
from ..tracker import TrackerConfig, WholePersonTracker
from .sensors import SensorConfig, Sensors
from .world import World

CONFIRMED, TEMPORARY, UNKNOWN = IdentityState.CONFIRMED, IdentityState.TEMPORARY, IdentityState.UNKNOWN


@dataclass
class FrameRecord:
    t: float
    state: IdentityState
    confidence: float
    points_to: object            # pid the estimate lies on, or None
    target_present: bool         # target exists in the scene (even if hidden)
    target_face_usable: bool     # a recognisable frontal target face was observed
    aim_error: object            # |aim - true head| / frame width, or None
    ms: float
    verifications: int
    evidence: str = ''


@dataclass
class Run:
    scenario: str
    system: str
    seed: int
    fps: float
    target_pid: object = None
    frames: list = field(default_factory=list)


def _points_to(estimate_xy, truths, target_pid, tolerance=0.25):
    """Which person does the estimate lie on? The target wins ties (heads can coincide)."""
    best, best_d = None, 1e9
    for tr in truths:
        d = float(np.linalg.norm(estimate_xy - tr.head_center) / (tr.box[3] - tr.box[1]))
        if d <= tolerance and (tr.pid == target_pid or (best != target_pid and d < best_d)):
            best, best_d = tr.pid, d
            if tr.pid == target_pid:
                break
    return best


def sim_tracker_config(target, **overrides):
    return TrackerConfig(target=target, **overrides)


def run(scenario, seed=0, system='three_state', sensor_config=None, tracker_config=None,
        world=None, cache=None):
    sensor_config = sensor_config or SensorConfig()
    world = world or World(scenario, seed)
    fps, target = scenario.fps, scenario.target
    target_pid = next((p.pid for p in scenario.persons if p.name == target), None)
    tracker = WholePersonTracker(tracker_config or sim_tracker_config(target))
    centering = CenteringController()
    result = Run(scenario.name, system, seed, fps, target_pid)
    sensors = Sensors(scenario, sensor_config, seed)
    for k in range(world.n_frames):
        t = k / fps
        if cache is not None and k in cache:
            frame, truths, persons, faces = cache[k]
        else:
            truths = world.truth(t)
            frame = world.render(t)
            persons, faces = sensors.persons(truths, k), sensors.faces(truths)
            if cache is not None:
                cache[k] = (frame, truths, persons, faces)
        truth_by_pid = {tr.pid: tr for tr in truths}
        target_truth = truth_by_pid.get(target_pid)
        present = target_truth is not None and target_truth.in_scene
        usable = any(f.meta['pid'] == target_pid and f.yaw_ratio <= tracker.config.yaw_limit
                     and f.size >= tracker.config.min_face_size for f in faces)

        start = time.perf_counter()
        if system == 'three_state':
            calls = []
            est = tracker.update(t, persons, faces, frame,
                                 recognize=lambda f: (calls.append(1), f.meta['recog'])[1])
            aim = centering.update(est, frame.shape, t)
            ms = (time.perf_counter() - start) * 1000
            if est.state == UNKNOWN or est.body_box is None:
                xy, aim_err = None, None
            else:
                xy = (0.5 * (est.face_box[:2] + est.face_box[2:]) if est.face_box is not None
                      else head_anchor(est.body_box))
                aim_err = (None if aim is None or target_truth is None
                           else float(np.linalg.norm(aim.point - target_truth.head_center) / frame.shape[1]))
            pt = None if xy is None else _points_to(xy, truths, target_pid)
            result.frames.append(FrameRecord(t, est.state, est.confidence, pt, present, usable,
                                             aim_err, ms, len(calls), est.evidence))
        elif system == 'face_lock':
            hit = next((f for f in faces if f.meta['recog'][0] == target), None)
            ms = (time.perf_counter() - start) * 1000
            if hit is None:
                result.frames.append(FrameRecord(t, UNKNOWN, 0.0, None, present, usable, None, ms, len(faces)))
            else:
                xy = 0.5 * (hit.box[:2] + hit.box[2:])
                err = (None if target_truth is None
                       else float(np.linalg.norm(xy - target_truth.head_center) / frame.shape[1]))
                result.frames.append(FrameRecord(t, CONFIRMED, 1.0, _points_to(xy, truths, target_pid), present,
                                                 usable, err, ms, len(faces), 'face'))
        else:
            raise ValueError(f'Unknown system {system!r}')
    return result
