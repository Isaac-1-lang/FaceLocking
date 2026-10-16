"""Watch or record a simulated scenario with the tracker overlay (no camera, no models).

    python -m src.person_tracking.sim --scenario path_crossing --show
    python -m src.person_tracking.sim --scenario head_turn --save data/reports/head_turn.mp4
    python -m src.person_tracking.sim --list
"""
import argparse
import time
from pathlib import Path

import cv2

from .. import overlay
from ..centering import CenteringController
from ..tracker import TrackerConfig, WholePersonTracker
from . import scenarios
from .sensors import SensorConfig, Sensors
from .world import World


def play(scenario, seed=0, show=False, save=None, sheet=None, detect_every=1, speed=1.0):
    world = World(scenario, seed)
    sensors = Sensors(scenario, SensorConfig(detect_every=detect_every), seed)
    tracker = WholePersonTracker(TrackerConfig(scenario.target))
    centering = CenteringController()
    writer, snapshots, calls = None, [], 0
    fps = scenario.fps
    for k in range(world.n_frames):
        t = k / fps
        truths = world.truth(t)
        frame = world.render(t)
        persons, faces = sensors.persons(truths, k), sensors.faces(truths)
        raw = frame.copy()                                  # inference sees the unannotated frame
        counter = []
        est = tracker.update(t, persons, faces, raw,
                             recognize=lambda f: (counter.append(1), f.meta['recog'])[1])
        calls += len(counter)
        aim = centering.update(est, frame.shape, t)
        view = overlay.draw(frame, tracker, est, aim, centering.dead_zone, [
            f'{scenario.name}  t={t:5.2f}s  seed {seed}',
            'ground truth: ' + ', '.join(f'{p.name}(vis {p.body_visibility:.0%}, yaw {abs(p.yaw) * 57.3:.0f})'
                                         for p in truths),
            f'ArcFace calls so far: {calls}'])
        if save:
            if writer is None:
                Path(save).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(save), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         (view.shape[1], view.shape[0]))
            writer.write(view)
        if sheet and k % max(int(world.n_frames / 8), 1) == 0:
            snapshots.append(cv2.resize(view, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
        if show:
            cv2.imshow('FaceX person-tracking simulation', cv2.resize(view, None, fx=0.7, fy=0.7))
            if cv2.waitKey(max(int(1000 / fps / speed), 1)) & 0xff in (ord('q'), 27):
                break
    if writer is not None:
        writer.release()
    if sheet and snapshots:
        import numpy as np
        rows = [np.hstack(snapshots[i:i + 2]) for i in range(0, len(snapshots) - 1, 2)]
        Path(sheet).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet), np.vstack(rows))
    cv2.destroyAllWindows()


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--scenario', default='path_crossing')
    cli.add_argument('--list', action='store_true')
    cli.add_argument('--seed', type=int, default=0)
    cli.add_argument('--show', action='store_true')
    cli.add_argument('--save', help='Write an .mp4 of the annotated run')
    cli.add_argument('--sheet', help='Write a PNG contact sheet of key frames')
    cli.add_argument('--detect-every', type=int, default=1)
    cli.add_argument('--speed', type=float, default=1.0)
    args = cli.parse_args(argv)
    if args.list:
        for name, make in scenarios.SCENARIOS.items():
            s = make()
            print(f'{name:22s} {s.duration:4.0f}s  {s.description}')
        return
    play(scenarios.get(args.scenario), args.seed, args.show, args.save, args.sheet,
         args.detect_every, args.speed)


if __name__ == '__main__':
    main()
