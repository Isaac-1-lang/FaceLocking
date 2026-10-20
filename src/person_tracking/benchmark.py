"""Time each stage of the CPU pipeline so budgets (FPS, ArcFace calls) can be chosen from facts.

    python -m src.person_tracking.benchmark
    python -m src.person_tracking.benchmark --width 1280 --height 720 --repeat 30
"""
import argparse
import os
import platform
import time

import cv2
import numpy as np

from ..config import DETECTOR, EMBEDDER
from .detectors import HogPersonDetector
from .sim import runner, scenarios


def timed(fn, repeat):
    fn()                                         # warm-up (lazy allocations, thread pools)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return float(np.median(samples)), float(np.percentile(samples, 95))


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--width', type=int, default=640)
    cli.add_argument('--height', type=int, default=480)
    cli.add_argument('--repeat', type=int, default=20)
    args = cli.parse_args(argv)
    print(f'CPU: {platform.processor()} | logical cores: {os.cpu_count()} | OpenCV {cv2.__version__} '
          f'| frame {args.width}x{args.height}')
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    rows = []

    hog = HogPersonDetector()
    rows.append(('HOG person detector (scale 0.5)',) + timed(lambda: hog.detect(frame), args.repeat))

    if DETECTOR.is_file() and DETECTOR.stat().st_size > 1024:
        from ..landmarks import FaceDetector
        yunet = FaceDetector(DETECTOR)
        rows.append(('YuNet face detector',) + timed(lambda: yunet.detect(frame), args.repeat))
    else:
        print('YuNet model missing: skipped (python init_project.py --download-models)')

    if EMBEDDER.is_file() and EMBEDDER.stat().st_size > 1024:
        from ..embed import ArcFace
        arc = ArcFace(EMBEDDER)
        face = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        rows.append(('ArcFace embedding (one face)',) + timed(lambda: arc.embed(face), max(args.repeat // 2, 5)))
    else:
        print('ArcFace model missing: skipped')

    scenario = scenarios.get('crowd')
    run = runner.run(scenario, 0, 'three_state')
    ms = np.array([f.ms for f in run.frames[5:]])
    rows.append(('Tracker update (4 people, Kalman + association + appearance)',
                 float(np.median(ms)), float(np.percentile(ms, 95))))

    print(f'\n{"stage":62s} {"median ms":>10s} {"p95 ms":>8s}')
    for name, median, p95 in rows:
        print(f'{name:62s} {median:10.2f} {p95:8.2f}')
    by = {name.split(' (')[0]: median for name, median, _ in rows}
    yunet_ms, hog_ms = by.get('YuNet face detector'), by.get('HOG person detector')
    arc_ms, trk = by.get('ArcFace embedding'), by['Tracker update']
    if yunet_ms is not None and arc_ms is not None:
        every = 3
        typical = yunet_ms + hog_ms / every + trk + arc_ms * 0.1
        print(f'\nTypical frame (YuNet + HOG every {every} frames + tracker + ~0.1 ArcFace calls): '
              f'{typical:.1f} ms -> {1000 / typical:.0f} FPS')
        print(f'Face-lock baseline (ArcFace on every face every frame, one face): '
              f'{yunet_ms + arc_ms:.1f} ms -> {1000 / (yunet_ms + arc_ms):.1f} FPS')


if __name__ == '__main__':
    main()
