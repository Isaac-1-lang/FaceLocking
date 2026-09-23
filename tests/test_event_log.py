import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from src.event_log import TrackingLog


class TrackingLogTests(unittest.TestCase):
    def test_session_persists_timestamp_settings_readings_and_end(self):
        with tempfile.TemporaryDirectory() as directory:
            with TrackingLog(directory, {'target': 'Name, with commas'}) as log:
                tracker = SimpleNamespace(
                    target_name='Name, with commas', state=SimpleNamespace(name='UNCERTAIN'),
                    reason='Identity uncertain (1/5)', frame_index=2,
                    detection_confidence=.91, match_name='Unknown', match_score=.42)
                position = SimpleNamespace(error_x=.1, error_y=-.2,
                                           horizontal='RIGHT', vertical='UP')
                signals = SimpleNamespace(smiling=True, eyes_closed=False,
                                          ear=.3, smile_score=.5, blink=True)
                log.frame(tracker, position, signals, 1, .21)
                tracker.state.name = 'LOST'
                tracker.detection_confidence = tracker.match_score = tracker.match_name = None
                log.frame(tracker, None, None, 1, .21)
                # Records must be readable before the session closes.
                with log.path.open(newline='', encoding='utf-8') as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 3)
            with log.path.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, ['Timestamp', 'Action Type', 'Description'])
            self.assertEqual([row['Action Type'] for row in rows],
                             ['SESSION_START', 'TRACKING_UNCERTAIN', 'TRACKING_LOST', 'SESSION_END'])
            self.assertIsNotNone(datetime.fromisoformat(rows[1]['Timestamp']).utcoffset())
            reading = json.loads(rows[1]['Description'])
            self.assertEqual(reading['target'], 'Name, with commas')
            self.assertEqual(reading['position']['error_x'], .1)
            self.assertEqual(reading['signals']['eye_state'], 'OPEN')
            self.assertTrue(reading['signals']['blink'])
            self.assertEqual(reading['match_score'], .42)
            lost = json.loads(rows[2]['Description'])
            self.assertIsNone(lost['position'])
            self.assertIsNone(lost['signals'])
            self.assertIsNone(lost['match_score'])
            self.assertEqual(lost['blink_total'], 1)

    def test_failure_is_recorded_and_propagated_without_overwriting_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Camera stopped'):
                with TrackingLog(directory, {}) as first:
                    raise RuntimeError('Camera stopped')
            with TrackingLog(directory, {}) as second:
                pass
            self.assertNotEqual(first.path, second.path)
            self.assertEqual(len(list(Path(directory).glob('*.csv'))), 2)
            with first.path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-2]['Action Type'], 'ERROR')
            self.assertIn('Camera stopped', rows[-2]['Description'])
            self.assertEqual(rows[-1]['Action Type'], 'SESSION_END')
