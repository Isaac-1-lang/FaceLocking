from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# MediaPipe FaceMesh landmark indices
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)

MOUTH_LEFT = 61
MOUTH_RIGHT = 291

FACE_LEFT = 234
FACE_RIGHT = 454


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance between two 2D points.
    """
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(
    points: np.ndarray,
    idx: Tuple[int, ...],
) -> float:
    """
    Calculate Eye Aspect Ratio (EAR).

    EAR becomes smaller when the eye closes.
    """
    p1, p2, p3, p4, p5, p6 = (
        points[i] for i in idx
    )

    width = max(
        distance(p1, p4),
        1e-6,
    )

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    return (
        vertical_1 + vertical_2
    ) / (2.0 * width)


@dataclass
class FaceSignals:
    ear: float
    blink: bool
    eyes_closed: bool

    smile_score: float
    smiling: bool


class FaceSignalExtractor:
    def __init__(
        self,
        ear_threshold: float = 0.21,
        blink_min_frames: int = 2,
        blink_max_frames: int = 7,
        closed_frames: int = 8,
        smile_on: float = 0.38,
        smile_off: float = 0.35,
    ):
        if not np.isfinite(ear_threshold) or ear_threshold <= 0:
            raise ValueError('ear_threshold must be positive and finite')
        if not (1 <= blink_min_frames <= blink_max_frames < closed_frames):
            raise ValueError('Require 1 <= blink_min_frames <= blink_max_frames < closed_frames')
        if not (0 <= smile_off < smile_on <= 1):
            raise ValueError('Require 0 <= smile_off < smile_on <= 1')
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError('Face signals require MediaPipe. Run python -m pip install -r requirements.txt') from error
        if not hasattr(mp, 'solutions'):
            raise RuntimeError('Face signals require mediapipe==0.10.21; install requirements.txt')
        self.ear_threshold = ear_threshold

        self.blink_min_frames = blink_min_frames
        self.blink_max_frames = blink_max_frames

        self.closed_frames = closed_frames

        self.smile_on = smile_on
        self.smile_off = smile_off

        self.low_ear_frames = 0
        self.smiling = False

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def reset(self) -> None:
        """
        Reset temporal facial state.

        Used when no locked target is currently available.
        """
        self.low_ear_frames = 0
        self.smiling = False

    def close(self) -> None:
        """
        Release MediaPipe resources.
        """
        self.mesh.close()

    def analyze(
        self,
        frame: np.ndarray,
        bbox,
    ) -> Optional[FaceSignals]:

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = np.rint(bbox).astype(int)

        box_width = x2 - x1
        box_height = y2 - y1

        # Add padding because the detection box
        # may cut off useful facial landmarks.
        pad_x = int(0.12 * box_width)
        pad_y = int(0.18 * box_height)

        roi_x1 = max(
            0,
            x1 - pad_x,
        )

        roi_y1 = max(
            0,
            y1 - pad_y,
        )

        roi_x2 = min(
            width,
            x2 + pad_x,
        )

        roi_y2 = min(
            height,
            y2 + pad_y,
        )

        roi = frame[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ]

        if roi.size == 0:
            self.reset()
            return None

        rgb_roi = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB,
        )

        result = self.mesh.process(
            rgb_roi
        )

        if not result.multi_face_landmarks:
            self.reset()
            return None

        roi_height, roi_width = roi.shape[:2]

        landmarks = (
            result
            .multi_face_landmarks[0]
            .landmark
        )

        points = np.array(
            [
                [
                    landmark.x * roi_width + roi_x1,
                    landmark.y * roi_height + roi_y1,
                ]
                for landmark in landmarks
            ],
            dtype=np.float32,
        )

        # -----------------------
        # Eye Aspect Ratio
        # -----------------------

        left_ear = eye_aspect_ratio(
            points,
            LEFT_EYE,
        )

        right_ear = eye_aspect_ratio(
            points,
            RIGHT_EYE,
        )

        ear = 0.5 * (
            left_ear + right_ear
        )

        # -----------------------
        # Blink detection
        # -----------------------

        blink = False

        if ear < self.ear_threshold:
            self.low_ear_frames += 1

        else:

            if (
                self.blink_min_frames
                <= self.low_ear_frames
                <= self.blink_max_frames
            ):
                blink = True

            self.low_ear_frames = 0

        # -----------------------
        # Sustained closed eyes
        # -----------------------

        eyes_closed = (
            self.low_ear_frames
            >= self.closed_frames
        )

        # -----------------------
        # Smile detection
        # -----------------------

        face_width = max(
            distance(
                points[FACE_LEFT],
                points[FACE_RIGHT],
            ),
            1e-6,
        )

        mouth_width = distance(
            points[MOUTH_LEFT],
            points[MOUTH_RIGHT],
        )

        smile_score = (
            mouth_width / face_width
        )

        # Hysteresis
        if self.smiling:

            self.smiling = (
                smile_score
                >= self.smile_off
            )

        else:

            self.smiling = (
                smile_score
                >= self.smile_on
            )

        return FaceSignals(
            ear=ear,
            blink=blink,
            eyes_closed=eyes_closed,
            smile_score=smile_score,
            smiling=self.smiling,
        )
