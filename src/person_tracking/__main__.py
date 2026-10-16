"""Track one enrolled person as a whole body, surviving temporary face loss.

    python -m src.person_tracking --target "Isaac"
    python -m src.person_tracking --target "Isaac" --video clip.mp4 --no-display

Needs the Part 1 enrollment database and models; runs on the CPU only.
"""
import time
from contextlib import closing, nullcontext

import cv2

from ..config import LOGS
from ..database import FaceDatabase
from ..event_log import TrackingLog
from ..workflow import camera, exiting, models, parser, read_frame, run
from . import overlay
from .detectors import make_person_detector
from .pipeline import ArcFaceVerifier, NeedleAdapter, PersonLockPipeline, tracker_config_from_args


def build_parser():
    cli = parser(__doc__)
    cli.add_argument('--target', required=True, help='Exact enrolled name to lock')
    cli.add_argument('--video', help='Process this video file instead of a camera (timestamps follow its FPS)')
    cli.add_argument('--no-display', action='store_true', help='Headless run (benchmarks, servers)')
    cli.add_argument('--max-frames', type=int, default=0, help='Stop after N frames (0 = no limit)')
    cli.add_argument('--person-backend', choices=('hog', 'yolo', 'none'), default='hog',
                     help='Body detector; none = rely on face-derived body boxes')
    cli.add_argument('--person-model', help='YOLO ONNX file for --person-backend yolo')
    cli.add_argument('--detect-every', type=int, default=3, help='Run the body detector every N frames')
    cli.add_argument('--detect-width', type=int, default=640,
                     help='Find faces on a frame at most this wide (0 = full size); big CPU saving at 720p+')
    cli.add_argument('--threshold', type=float, default=0.45, help='Face cosine similarity threshold')
    cli.add_argument('--margin', type=float, default=0.05)
    cli.add_argument('--min-face', type=float, default=48.0, help='Smallest face (px) sent to ArcFace')
    cli.add_argument('--max-verifications', type=int, default=1, help='ArcFace calls per frame (CPU budget)')
    cli.add_argument('--half-life', type=float, default=3.0, help='Seconds for confidence to halve without a face')
    cli.add_argument('--confirm', type=float, default=0.70, help='Confidence needed for CONFIRMED')
    cli.add_argument('--unknown', type=float, default=0.20, help='Confidence below which the target is UNKNOWN')
    cli.add_argument('--dead-zone', type=float, default=0.07)
    cli.add_argument('--log-dir', default=str(LOGS))
    cli.add_argument('--mqtt-host', default=None, help='Publish the aim to the ESP needle display')
    cli.add_argument('--mqtt-port', type=int, default=1883)
    cli.add_argument('--mqtt-topic', default='facex/tracking')
    return cli


def main():
    cli = build_parser()
    args = cli.parse_args()
    if args.detect_every < 1 or args.max_verifications < 1 or not 0 < args.unknown < args.confirm <= 1:
        cli.error('Need --detect-every >= 1, --max-verifications >= 1 and 0 < --unknown < --confirm <= 1')
    with TrackingLog(args.log_dir, vars(args)) as log:
        print(f'Session log: {log.path}')
        session(args, cli, log)


class _Source:
    """Camera (wall-clock time) or video file (frame-index time)."""

    def __init__(self, args):
        self.args = args
        self.cap = cv2.VideoCapture(args.video) if args.video else None
        if self.cap is not None and not self.cap.isOpened():
            raise RuntimeError(f'Could not open video {args.video}')
        self.fps = (self.cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.cap is not None else None
        self.index = 0

    def frames(self):
        if self.cap is None:
            with camera(self.args.camera) as cap:
                start = time.perf_counter()
                while True:
                    yield read_frame(cap), time.perf_counter() - start
        else:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    return
                yield frame, self.index / self.fps
                self.index += 1

    def close(self):
        if self.cap is not None:
            self.cap.release()


def session(args, cli, log):
    detector, embedder = models(args)
    db = FaceDatabase(args.db, embedder.signature)
    if not db.people:
        cli.error('No faces enrolled. Run python -m src.enroll --name "Your Name" first.')
    if args.target not in db.people:
        cli.error(f'Target {args.target!r} is not enrolled; use the exact enrolled name')
    person = make_person_detector(args.person_backend, args.person_model, args.detect_every)
    source = _Source(args)
    interval = args.detect_every / (source.fps or 15.0)
    config = tracker_config_from_args(args, interval)
    pipeline = PersonLockPipeline(config, detector, ArcFaceVerifier(embedder, db, args.threshold, args.margin),
                                  person, detect_width=args.detect_width)
    needle = None
    if args.mqtt_host:
        from ..mqtt_bridge import NeedlePublisher
        needle = NeedleAdapter(NeedlePublisher(args.mqtt_host, args.mqtt_port, args.mqtt_topic), args.target)
    window = 'FaceX whole-person lock'
    print(f'Tracking {args.target!r} (CPU). Q or Escape quits.')
    frames = 0
    started = time.perf_counter()
    try:
        with closing(needle) if needle is not None else nullcontext():
            for frame, t in source.frames():
                frames += 1
                result = pipeline.process(frame, t)
                if needle is not None:
                    needle.publish(result)
                est, aim = result.estimate, result.aim
                log.write(f'PERSON_{est.state.name}', {
                    'frame': frames, 't': round(t, 3), 'target': args.target, 'track': est.track_id,
                    'confidence': round(est.confidence, 4), 'evidence': est.evidence, 'reason': est.reason,
                    'seconds_since_face': None if est.seconds_since_face is None else round(est.seconds_since_face, 3),
                    'aim': None if aim is None else {
                        'error_x': round(aim.error_x, 4), 'error_y': round(aim.error_y, 4),
                        'horizontal': aim.horizontal, 'vertical': aim.vertical, 'source': aim.source,
                        'gain': round(aim.gain, 3), 'angle': round(aim.angle, 2)},
                    'ms': round(result.ms, 2), 'verifications': result.verifications})
                if not args.no_display:
                    fps = frames / max(time.perf_counter() - started, 1e-6)
                    view = overlay.draw(frame, pipeline.tracker, est, aim, args.dead_zone,
                                        [f'{fps:.1f} FPS | {result.ms:.0f} ms/frame | '
                                         f'ArcFace calls: {pipeline.verifier.calls}',
                                         f'Half-life {args.half_life:g}s | confirm {args.confirm:g} | '
                                         f'unknown {args.unknown:g}', 'Q / Escape: quit'])
                    cv2.imshow(window, view)
                    if exiting(window, cv2.waitKey(1) & 0xff):
                        break
                if args.max_frames and frames >= args.max_frames:
                    break
    finally:
        source.close()
        cv2.destroyAllWindows()
    elapsed = max(time.perf_counter() - started, 1e-6)
    print(f'{frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} FPS), '
          f'{pipeline.verifier.calls} ArcFace calls')


if __name__ == '__main__':
    run(main)
