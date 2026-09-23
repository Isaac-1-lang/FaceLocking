"""Session CSV logs with timezone-aware timestamps and structured descriptions."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class TrackingLog:
    def __init__(self, directory, settings):
        self.directory = Path(directory)
        self.settings = settings
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.path = self.directory / f'tracking_{stamp}_{uuid4().hex}.csv'

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('x', encoding='utf-8', newline='')
        try:
            self.writer = csv.writer(self.stream)
            self.writer.writerow(['Timestamp', 'Action Type', 'Description'])
            self.write('SESSION_START', {'settings': self.settings})
        except BaseException:
            self.stream.close()
            raise
        return self

    def write(self, action, description):
        self.writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            action, json.dumps(description, ensure_ascii=False, allow_nan=False)])
        self.stream.flush()

    def frame(self, tracker, position, signals, blink_total, ear_threshold):
        position_data = None if position is None else {
            'error_x': position.error_x, 'error_y': position.error_y,
            'horizontal': position.horizontal, 'vertical': position.vertical}
        signal_data = None if signals is None else {
            'smiling': bool(signals.smiling), 'smile_score': float(signals.smile_score),
            'ear': float(signals.ear), 'blink': bool(signals.blink),
            'eye_state': ('CLOSED' if signals.eyes_closed else 'CLOSING')
            if signals.ear < ear_threshold else 'OPEN'}
        self.write(f'TRACKING_{tracker.state.name}', {
            'frame': tracker.frame_index, 'target': tracker.target_name,
            'reason': tracker.reason, 'detection_confidence': tracker.detection_confidence,
            'match_name': tracker.match_name, 'match_score': tracker.match_score,
            'position': position_data, 'signals': signal_data, 'blink_total': blink_total})

    def __exit__(self, error_type, error, traceback):
        try:
            if error is not None and not isinstance(error, KeyboardInterrupt):
                self.write('ERROR', {'type': error_type.__name__, 'message': str(error)})
            self.write('SESSION_END', {'reason': 'interrupted' if isinstance(error, KeyboardInterrupt)
                                      else 'error' if error is not None else 'normal exit'})
        finally:
            self.stream.close()
        return False
