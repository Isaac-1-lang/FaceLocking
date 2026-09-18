"""Lock one enrolled identity using the existing Part 1 recognition pipeline."""
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

from src.align import align_face
from src.database import FaceDatabase
from src.landmarks import draw_face
from src.workflow import camera, exiting, models, parser, read_frame, run


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()


def center(box):
    """Center of an (x1, y1, x2, y2) box."""
    return (np.asarray(box[:2]) + np.asarray(box[2:])) / 2.0


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1.0)


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


class LockedFaceTracker:
    def __init__(self, target_name, detector, embedder, matcher,
                 verify_every=1, lost_timeout=24, ema_alpha=0.30,
                 dead_zone=0.07, threshold=0.45, margin=0.05):
        if target_name not in matcher.people:
            raise ValueError(f'Target {target_name!r} is not enrolled; use the exact enrolled name')
        if not -1 <= threshold <= 1 or not 0 <= margin <= 2:
            raise ValueError('Threshold must be [-1,1] and margin [0,2]')
        if (not isinstance(verify_every, int) or verify_every < 1
                or not isinstance(lost_timeout, int) or lost_timeout < 0):
            raise ValueError('verify_every must be a positive integer; lost_timeout must be nonnegative')
        if not 0 < ema_alpha <= 1 or not 0 <= dead_zone < 1:
            raise ValueError('ema_alpha must be (0,1] and dead_zone [0,1)')
        self.target_name = target_name
        self.detector, self.embedder, self.matcher = detector, embedder, matcher
        self.verify_every, self.lost_timeout = verify_every, lost_timeout
        self.ema_alpha, self.dead_zone = ema_alpha, dead_zone
        self.threshold, self.margin = threshold, margin
        self.state = LockState.SEARCHING
        self.last_box = None
        self.smooth_center = None
        self.lost_frames = 0
        self.frame_index = 0

    @staticmethod
    def box(face):
        # YuNet stores (x, y, width, height), unlike the PDF's detector.
        x, y, width, height = face.box
        return np.array([x, y, x + width, y + height], dtype=np.float32)

    def identity(self, frame, face):
        try:
            vector = self.embedder.embed(align_face(frame, face.points))
            return self.matcher.match(vector, self.threshold, self.margin)
        except ValueError:
            return 'Unknown', 0.0

    def acquire(self, frame, faces):
        best, best_score = None, -float('inf')
        for face in faces:
            name, score = self.identity(frame, face)
            if name == self.target_name and score > best_score:
                best, best_score = face, score
        return best

    def associate(self, faces):
        if self.last_box is None or not faces:
            return None
        last_center = center(self.last_box)
        diagonal = max(float(np.linalg.norm(self.last_box[2:] - self.last_box[:2])), 1.0)
        ranked = []
        for face in faces:
            box = self.box(face)
            displacement = np.linalg.norm(center(box) - last_center) / diagonal
            ranked.append((iou(self.last_box, box) - 0.35 * displacement, face))
        score, candidate = max(ranked, key=lambda item: item[0])
        return candidate if score > -0.30 else None

    def update(self, frame):
        self.frame_index += 1
        faces = [face for face in self.detector.detect(frame) if min(face.box[2:]) >= 70]
        if self.state == LockState.SEARCHING:
            candidate = self.acquire(frame, faces)
        else:
            candidate = self.associate(faces)
            # Always verify after a gap or when multiple people could cross.
            verify = (self.state == LockState.LOST or len(faces) > 1
                      or self.frame_index % self.verify_every == 0)
            if candidate is not None and verify:
                if self.identity(frame, candidate)[0] != self.target_name:
                    candidate = None

        if candidate is None:
            if self.last_box is not None:
                self.lost_frames += 1
                self.state = LockState.LOST
                if self.lost_frames > self.lost_timeout:
                    self.state = LockState.SEARCHING
                    self.last_box = None
                    self.smooth_center = None
            return None, None

        self.state = LockState.LOCKED
        self.lost_frames = 0
        self.last_box = self.box(candidate)
        raw_center = center(self.last_box)
        if self.smooth_center is None:
            self.smooth_center = raw_center
        else:
            self.smooth_center = (self.ema_alpha * raw_center
                                  + (1 - self.ema_alpha) * self.smooth_center)
        return candidate, self.position_signal(frame.shape)

    def position_signal(self, shape):
        height, width = shape[:2]
        ex = float((self.smooth_center[0] - width / 2) / (width / 2))
        ey = float((self.smooth_center[1] - height / 2) / (height / 2))
        horizontal = 'LEFT' if ex < -self.dead_zone else 'RIGHT' if ex > self.dead_zone else 'CENTER'
        vertical = 'UP' if ey < -self.dead_zone else 'DOWN' if ey > self.dead_zone else 'CENTER'
        return TrackingSignal(ex, ey, horizontal, vertical)


def main():
    cli = parser(__doc__)
    cli.add_argument('--target', required=True, help='Exact enrolled name to lock')
    cli.add_argument('--threshold', type=float, default=0.45, help='Part 1 cosine similarity threshold')
    cli.add_argument('--margin', type=float, default=0.05)
    cli.add_argument('--verify-every', type=int, default=1,
                     help='Identity check interval; larger values can briefly follow an unverified face')
    cli.add_argument('--lost-timeout', type=int, default=24, help='Missed frames allowed before searching again')
    cli.add_argument('--ema-alpha', type=float, default=0.30)
    cli.add_argument('--dead-zone', type=float, default=0.07)
    args = cli.parse_args()
    detector, embedder = models(args)
    db = FaceDatabase(args.db, embedder.signature)
    if not db.people:
        cli.error('No faces enrolled. Run python -m src.enroll --name "Your Name" first.')
    tracker = LockedFaceTracker(args.target, detector, embedder, db,
                                args.verify_every, args.lost_timeout, args.ema_alpha,
                                args.dead_zone, args.threshold, args.margin)
    window = 'FaceX identity lock'
    print(f'Tracking {args.target!r}. Q or Escape quits. Scores use Part 1 cosine matching.')
    with camera(args.camera) as cap:
        while True:
            frame = read_frame(cap)
            face, position = tracker.update(frame)
            # All inference uses the original pixels before overlays are drawn.
            color = (0, 200, 0) if face is not None else (0, 140, 255)
            if face is not None:
                draw_face(frame, face, f'{args.target} | LOCKED', color)
                cx, cy = np.rint(tracker.smooth_center).astype(int)
                cv2.circle(frame, (cx, cy), 5, (255, 170, 0), -1)
                text = (f'{position.horizontal} / {position.vertical} '
                        f'error=({position.error_x:+.2f}, {position.error_y:+.2f})')
                cv2.putText(frame, text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 170, 0), 2)
            cv2.putText(frame, f'{tracker.state.name}: {args.target} | Q: quit',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            height, width = frame.shape[:2]
            dz = tracker.dead_zone
            cv2.rectangle(frame, (int(width * (0.5 - dz / 2)), int(height * (0.5 - dz / 2))),
                          (int(width * (0.5 + dz / 2)), int(height * (0.5 + dz / 2))),
                          (120, 120, 120), 1)
            cv2.imshow(window, frame)
            if exiting(window, cv2.waitKey(1) & 0xff):
                break


if __name__ == '__main__':
    run(main)
