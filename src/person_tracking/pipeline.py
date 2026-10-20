"""Glue between the real Part 1/2 components (YuNet, ArcFace, JSON gallery) and the tracker."""
import time
from dataclasses import dataclass
from typing import Optional

import cv2

from ..align import align_face
from .centering import Aim, CenteringController
from .identity import IdentityState
from .records import FaceObs, TargetEstimate
from .tracker import TrackerConfig, WholePersonTracker


class ArcFaceVerifier:
    """Callable ``(frame, FaceObs) -> (name, similarity)`` using the Part 1 matcher."""

    def __init__(self, embedder, matcher, threshold=0.45, margin=0.05):
        self.embedder, self.matcher = embedder, matcher
        self.threshold, self.margin = threshold, margin
        self.calls = 0

    def __call__(self, frame, face):
        self.calls += 1
        try:
            vector = self.embedder.embed(align_face(frame, face.points))
            return self.matcher.match(vector, self.threshold, self.margin)
        except ValueError:      # degenerate landmarks / embedding
            return 'Unknown', 0.0


@dataclass
class FrameResult:
    estimate: TargetEstimate
    aim: Optional[Aim]
    faces: list
    ms: float
    verifications: int


class PersonLockPipeline:
    """One call per frame: detect, track, verify (sparingly), centre."""

    def __init__(self, tracker_config, face_detector, verifier, person_detector=None,
                 centering=None, detect_width=640):
        self.detect_width = detect_width    # faces are found on a frame no wider than this (CPU)
        self.tracker = WholePersonTracker(tracker_config)
        self.face_detector, self.verifier = face_detector, verifier
        self.person_detector = person_detector
        self.centering = centering or CenteringController(
            confirm_threshold=tracker_config.identity.confirm_threshold)

    def _faces(self, frame):
        """YuNet on a downscaled copy; boxes and landmarks are mapped back to full resolution,
        where ArcFace alignment still samples the original pixels."""
        scale = min(1.0, self.detect_width / frame.shape[1]) if self.detect_width else 1.0
        small = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale,
                                                      interpolation=cv2.INTER_AREA)
        faces = []
        for face in self.face_detector.detect(small):
            obs = FaceObs.from_face(face)
            if scale != 1.0:
                obs.box = obs.box / scale
                obs.points = obs.points / scale
            faces.append(obs)
        return faces

    def process(self, frame, t) -> FrameResult:
        start = time.perf_counter()
        faces = self._faces(frame)
        persons = None if self.person_detector is None else self.person_detector.detect(frame)
        estimate = self.tracker.update(
            t, persons, faces, frame,
            recognize=lambda face: self.verifier(frame, face))
        aim = self.centering.update(estimate, frame.shape, t)
        return FrameResult(estimate, aim, faces, (time.perf_counter() - start) * 1000,
                           self.tracker.verifications)


_LEGACY_STATE = {IdentityState.CONFIRMED: 'LOCKED', IdentityState.TEMPORARY: 'UNCERTAIN',
                 IdentityState.UNKNOWN: 'SEARCHING'}


class NeedleAdapter:
    """Lets the existing ``src.mqtt_bridge.NeedlePublisher`` publish this module's output.

    The publisher reads ``tracker.state.name`` / ``tracker.target_name`` and a
    position with ``horizontal`` and ``error_x``. UNKNOWN publishes no position, and
    only CONFIRMED is reported as ``locked``.
    """

    class _State:
        def __init__(self, name):
            self.name = name

    def __init__(self, publisher, target):
        self.publisher, self.target_name = publisher, target
        self.state = self._State('SEARCHING')

    def publish(self, result: FrameResult):
        self.state = self._State(_LEGACY_STATE[result.estimate.state])
        self.publisher.publish(self, result.aim)

    def close(self):
        self.publisher.close()


def tracker_config_from_args(args, detect_interval_s):
    from .identity import IdentityConfig
    identity = IdentityConfig(half_life=args.half_life, confirm_threshold=args.confirm,
                              drop_threshold=min(args.confirm - 0.05, 0.5),
                              unknown_threshold=args.unknown)
    return TrackerConfig(
        target=args.target, match_threshold=args.threshold, min_face_size=args.min_face,
        max_verifications=args.max_verifications,
        coast_grace=max(0.4, 2.5 * detect_interval_s), identity=identity)
