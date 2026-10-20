"""Drawing helpers shared by the live viewer and the simulator."""
import cv2
import numpy as np

from .identity import IdentityState

COLORS = {IdentityState.CONFIRMED: (0, 200, 0), IdentityState.TEMPORARY: (0, 150, 255),
          IdentityState.UNKNOWN: (160, 160, 160)}


def _label(frame, text, x, y, color):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = int(max(0, min(x, frame.shape[1] - tw - 8)))
    y = int(max(th + 6, min(y, frame.shape[0] - 4)))
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 8, y + base - 2), (28, 32, 38), -1)
    cv2.putText(frame, text, (x + 4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw(frame, tracker, estimate, aim, dead_zone=0.07, lines=()):
    """Annotate ``frame`` in place and return a copy with a status panel on the right."""
    for tr in tracker.tracks:
        if tr.hits < tracker.config.min_hits:
            continue
        is_target = estimate.track_id == tr.id and estimate.state != IdentityState.UNKNOWN
        color = COLORS[tr.state] if is_target else (110, 110, 110)
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        thickness = 3 if is_target else 1
        if tr.gap > tracker.config.coast_grace:           # dashed look: not currently detected
            for i in range(x1, x2, 16):
                cv2.line(frame, (i, y1), (min(i + 8, x2), y1), color, thickness)
                cv2.line(frame, (i, y2), (min(i + 8, x2), y2), color, thickness)
            for j in range(y1, y2, 16):
                cv2.line(frame, (x1, j), (x1, min(j + 8, y2)), color, thickness)
                cv2.line(frame, (x2, j), (x2, min(j + 8, y2)), color, thickness)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if is_target:
            _label(frame, f'{tracker.config.target} {tr.state.name} {tr.belief.confidence:.2f}',
                   x1, y1 - 2, color)
            bar = int((x2 - x1) * min(tr.belief.confidence, 1.0))
            cv2.rectangle(frame, (x1, y2 + 4), (x2, y2 + 10), (60, 60, 60), -1)
            cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar, y2 + 10), color, -1)
        elif tr.other_name:
            _label(frame, tr.other_name, x1, y1 - 2, (200, 200, 200))
    if estimate.face_box is not None:
        x1, y1, x2, y2 = (int(v) for v in estimate.face_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1, cv2.LINE_AA)
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (int(width * (0.5 - dead_zone / 2)), int(height * (0.5 - dead_zone / 2))),
                  (int(width * (0.5 + dead_zone / 2)), int(height * (0.5 + dead_zone / 2))),
                  (160, 160, 160), 1, cv2.LINE_AA)
    if aim is not None:
        cx, cy = (int(round(v)) for v in aim.point)
        cv2.drawMarker(frame, (cx, cy), (255, 170, 0), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

    text = [f'{estimate.state.name}  conf={estimate.confidence:.2f}', estimate.reason,
            f'evidence: {estimate.evidence}',
            (f'aim: {aim.horizontal}/{aim.vertical} ex={aim.error_x:+.2f} ey={aim.error_y:+.2f} '
             f'gain={aim.gain:.2f}') if aim else 'aim: none (no command)', ''] + list(lines)
    panel = np.full((height, 420, 3), 24, dtype=np.uint8)
    for i, line in enumerate(text):
        color = COLORS[estimate.state] if i == 0 else (230, 230, 230)
        scale = min(0.5, 400 / max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0], 1))
        cv2.putText(panel, line, (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return np.hstack([frame, panel])
