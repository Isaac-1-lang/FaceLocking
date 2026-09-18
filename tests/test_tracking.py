import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from contextlib import nullcontext

import numpy as np

from src.align import TEMPLATE
from src.database import FaceDatabase
from src.face_tracking import LockedFaceTracker, LockState
from src.landmarks import Face
from src.face_signals import FaceSignals
from src import face_tracking


class TrackingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = FaceDatabase(Path(directory.name) / 'faces.json', {})
        self.target, self.other, self.unknown = np.eye(3, 512, dtype=np.float32)
        self.db.enroll('Isaac', [self.target])
        self.db.enroll('Alice', [self.other])
        # Only model inference is replaced; alignment and database matching are real.
        self.detector = Mock()
        self.embedder = Mock()
        self.face = Face(np.array([100, 100, 100, 100]), TEMPLATE.copy(), 0.99)
        self.detector.detect.return_value = [self.face]
        self.embedder.embed.return_value = self.target
        self.frame = np.zeros((400, 400, 3), dtype=np.uint8)
        self.tracker = LockedFaceTracker('Isaac', self.detector, self.embedder,
                                         self.db, lost_timeout=2)

    def test_only_requested_identity_acquires_lock(self):
        for vector in (self.other, self.unknown):
            self.embedder.embed.return_value = vector
            self.assertEqual(self.tracker.update(self.frame), (None, None))
            self.assertEqual(self.tracker.state, LockState.SEARCHING)
        self.embedder.embed.return_value = self.target
        face, position = self.tracker.update(self.frame)
        self.assertIs(face, self.face)
        self.assertEqual(self.tracker.state, LockState.LOCKED)
        self.assertAlmostEqual(position.error_x, -0.25)
        self.assertEqual((position.horizontal, position.vertical), ('LEFT', 'UP'))

    def test_crossing_distractor_cannot_inherit_lock(self):
        self.tracker.update(self.frame)
        self.embedder.embed.return_value = self.other
        self.assertEqual(self.tracker.update(self.frame), (None, None))
        self.assertEqual(self.tracker.state, LockState.LOST)
        self.embedder.embed.return_value = self.target
        self.assertIs(self.tracker.update(self.frame)[0], self.face)

    def test_timeout_clears_geometry_and_reacquires_same_identity(self):
        self.tracker.update(self.frame)
        self.detector.detect.return_value = []
        self.assertEqual(self.tracker.update(self.frame), (None, None))
        self.assertEqual(self.tracker.state, LockState.LOST)
        self.tracker.update(self.frame)
        self.tracker.update(self.frame)
        self.assertEqual(self.tracker.state, LockState.SEARCHING)
        self.assertIsNone(self.tracker.last_box)
        self.assertIsNone(self.tracker.smooth_center)
        self.face.box = np.array([250, 250, 100, 100])
        self.detector.detect.return_value = [self.face]
        _, position = self.tracker.update(self.frame)
        self.assertAlmostEqual(position.error_x, 0.5)
        self.assertEqual((position.horizontal, position.vertical), ('RIGHT', 'DOWN'))

    def test_lost_return_is_verified_even_between_scheduled_checks(self):
        self.tracker.verify_every = 10
        self.tracker.update(self.frame)
        self.detector.detect.return_value = []
        self.tracker.update(self.frame)
        self.detector.detect.return_value = [self.face]
        self.embedder.embed.return_value = self.other
        self.assertEqual(self.tracker.update(self.frame), (None, None))
        self.assertEqual(self.tracker.state, LockState.LOST)

    def test_small_and_invalid_landmarks_do_not_lock(self):
        self.face.box = np.array([100, 100, 50, 50])
        self.assertEqual(self.tracker.update(self.frame), (None, None))
        self.face.box = np.array([100, 100, 100, 100])
        self.face.points[:] = 0
        self.assertEqual(self.tracker.update(self.frame), (None, None))

    def test_best_target_candidate_is_selected(self):
        second = Face(np.array([250, 100, 100, 100]), TEMPLATE.copy(), 0.99)
        self.detector.detect.return_value = [self.face, second]
        self.embedder.embed.side_effect = [0.8 * self.target + 0.6 * self.unknown,
                                           self.target]
        self.assertIs(self.tracker.update(self.frame)[0], second)

    def test_center_smoothing_and_distant_face_rejection(self):
        self.face.box = np.array([150, 150, 100, 100])
        _, position = self.tracker.update(self.frame)
        self.assertEqual((position.horizontal, position.vertical), ('CENTER', 'CENTER'))
        self.face.box = np.array([170, 150, 100, 100])
        _, position = self.tracker.update(self.frame)
        self.assertAlmostEqual(position.error_x, 0.03)
        self.assertEqual(position.horizontal, 'CENTER')
        self.face.box = np.array([500, 500, 100, 100])
        self.assertEqual(self.tracker.update(self.frame), (None, None))
        self.assertEqual(self.tracker.state, LockState.LOST)

    def test_periodic_verification_rejects_changed_identity(self):
        self.tracker.verify_every = 3
        self.tracker.update(self.frame)
        self.embedder.embed.return_value = self.other
        self.assertIsNotNone(self.tracker.update(self.frame)[0])
        self.assertEqual(self.tracker.update(self.frame), (None, None))

    def test_unknown_target_and_invalid_options_rejected(self):
        for options in ({'target_name': 'Nobody'}, {'verify_every': 0},
                        {'lost_timeout': -1}, {'ema_alpha': 0},
                        {'dead_zone': 2}, {'threshold': float('nan')}):
            kwargs = dict(target_name='Isaac', detector=self.detector,
                          embedder=self.embedder, matcher=self.db)
            kwargs.update(options)
            with self.assertRaises(ValueError):
                LockedFaceTracker(**kwargs)

    def test_live_loop_counts_blink_and_clears_readings_after_loss(self):
        self.detector.detect.side_effect = [[self.face], []]
        extractor = Mock()
        extractor.analyze.return_value = FaceSignals(.286, True, False, .412, True)
        cap = Mock()
        cap.read.side_effect = [(True, self.frame.copy()), (True, self.frame.copy())]
        with patch('sys.argv', ['tracking', '--target', 'Isaac']), \
                patch.object(face_tracking, 'models', return_value=(self.detector, self.embedder)), \
                patch.object(face_tracking, 'FaceDatabase', return_value=self.db), \
                patch.object(face_tracking, 'FaceSignalExtractor', return_value=extractor), \
                patch.object(face_tracking, 'camera', return_value=nullcontext(cap)), \
                patch.object(face_tracking.cv2, 'imshow') as show, \
                patch.object(face_tracking.cv2, 'waitKey', return_value=0), \
                patch.object(face_tracking, 'exiting', side_effect=[False, True]), \
                patch.object(face_tracking, 'status_panel', wraps=face_tracking.status_panel) as panel:
            face_tracking.main()
        first, second = panel.call_args_list
        self.assertEqual(first.args[4], 1)
        self.assertEqual(second.args[4], 1)
        self.assertIsNone(second.args[2])
        self.assertIsNone(second.args[3])
        self.assertGreater(show.call_args.args[1].shape[1], self.frame.shape[1])
        extractor.reset.assert_called_once()
        extractor.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
