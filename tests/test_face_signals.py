import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.face_signals import FaceSignalExtractor, LEFT_EYE, RIGHT_EYE


class SignalTests(unittest.TestCase):
    def setUp(self):
        # Replace only FaceMesh inference; exercise real crop and temporal logic.
        self.extractor = FaceSignalExtractor.__new__(FaceSignalExtractor)
        self.extractor.ear_threshold = 0.21
        self.extractor.blink_min_frames = 2
        self.extractor.blink_max_frames = 7
        self.extractor.closed_frames = 8
        self.extractor.smile_on = 0.38
        self.extractor.smile_off = 0.35
        self.extractor.mesh = Mock()
        self.extractor.reset()
        self.frame = np.zeros((240, 240, 3), np.uint8)

    def analyze(self, ear=0.3, smile=0.4):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        for indices in (LEFT_EYE, RIGHT_EYE):
            coords = [(0.2, 0.5), (0.3, 0.5 - ear / 2),
                      (0.6, 0.5 - ear / 2), (0.7, 0.5),
                      (0.6, 0.5 + ear / 2), (0.3, 0.5 + ear / 2)]
            # Horizontal width=.5; vertical height=ear*.5 gives requested EAR.
            for index, (x, y) in zip(indices, coords):
                points[index] = SimpleNamespace(x=x, y=0.5 + (y - 0.5) / 2 * 124 / 136)
        for index, x in ((234, 0), (454, 1), (61, 0.5 - smile / 2),
                         (291, 0.5 + smile / 2)):
            points[index] = SimpleNamespace(x=x, y=0.5)
        self.extractor.mesh.process.return_value = SimpleNamespace(
            multi_face_landmarks=[SimpleNamespace(landmark=points)])
        return self.extractor.analyze(self.frame, np.array([100., 100., 200., 200.]))

    def test_float_box_and_blink_count_event(self):
        self.assertAlmostEqual(self.analyze().ear, 0.3, places=5)
        self.assertFalse(self.analyze(0.1).blink)
        self.assertFalse(self.analyze(0.1).blink)
        self.assertTrue(self.analyze().blink)
        self.assertFalse(self.analyze().blink)

    def test_sustained_closure_is_not_a_blink(self):
        for _ in range(8):
            state = self.analyze(0.1)
        self.assertTrue(state.eyes_closed)
        self.assertFalse(self.analyze().blink)

    def test_missing_landmarks_reset_pending_blink(self):
        self.analyze(0.1)
        self.analyze(0.1)
        self.extractor.mesh.process.return_value = SimpleNamespace(multi_face_landmarks=None)
        self.assertIsNone(self.extractor.analyze(self.frame, (20, 20, 200, 200)))
        self.assertFalse(self.analyze().blink)

    def test_smile_hysteresis(self):
        self.assertFalse(self.analyze(smile=0.36).smiling)
        self.assertTrue(self.analyze(smile=0.40).smiling)
        self.assertTrue(self.analyze(smile=0.36).smiling)
        self.assertFalse(self.analyze(smile=0.30).smiling)


if __name__ == '__main__':
    unittest.main()
