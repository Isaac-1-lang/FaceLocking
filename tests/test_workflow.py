import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from src.workflow import camera, read_frame, run


class CameraTests(unittest.TestCase):
    def test_incomplete_opencv_reports_reinstall_instead_of_second_traceback(self):
        with patch('src.workflow.cv2', SimpleNamespace()):
            with self.assertRaisesRegex(SystemExit, 'OpenCV installation is incomplete'):
                run(lambda: self.fail('Must validate OpenCV before starting'))

    @patch("src.workflow.cv2.destroyAllWindows")
    @patch("src.workflow.cv2.VideoCapture")
    def test_camera_released_when_processing_raises(self, factory, cleanup):
        cap = factory.return_value
        cap.isOpened.return_value = True
        with self.assertRaises(ValueError):
            with camera(1):
                raise ValueError("processing failed")
        cap.release.assert_called_once()
        cleanup.assert_called_once()

    @patch("src.workflow.cv2.destroyAllWindows")
    @patch("src.workflow.cv2.VideoCapture")
    def test_unavailable_camera_released(self, factory, cleanup):
        factory.return_value.isOpened.return_value = False
        with self.assertRaises(RuntimeError):
            with camera(0):
                self.fail("Unavailable camera must not yield")
        factory.return_value.release.assert_called_once()

    def test_frame_failure(self):
        cap = MagicMock()
        cap.read.return_value = (False, None)
        with self.assertRaises(RuntimeError):
            read_frame(cap)
        frame = np.zeros((2, 2, 3), np.uint8)
        cap.read.return_value = (True, frame)
        self.assertIs(read_frame(cap), frame)
