"""Evaluate the three-state tracker against the Part 2 face-lock baseline, CPU only.

    python -m src.person_tracking.sim.evaluate --seeds 5
    python -m src.person_tracking.sim.evaluate --scenarios head_turn path_crossing --detect-every 4
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np

from ...config import ROOT
from . import metrics, runner, scenarios
from .sensors import SensorConfig
from .world import World

SYSTEMS = ('three_state', 'face_lock')
HEADLINE = ('coverage', 'coverage_during_face_loss', 'wrong_hold_rate', 'ghost_seconds',
            'id_switches', 'confirmed_precision', 'temporary_precision')


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def evaluate(names, seeds, sensor_config, tracker_config=None):
    """Return ``{scenario: {system: [metrics per seed]}}`` with shared observations per seed."""
    results = {}
    for name in names:
        results[name] = {system: [] for system in SYSTEMS}
        for seed in range(seeds):
            scenario = scenarios.get(name)
            world = World(scenario, seed)
            cache = {}
            for system in SYSTEMS:
                run = runner.run(scenario, seed, system, sensor_config, tracker_config, world, cache)
                results[name][system].append(metrics.score(run))
    return results


def aggregate(per_seed):
    keys = per_seed[0].keys()
    out = {}
    for key in keys:
        if key == 'reacquire_s':
            flat = [v for m in per_seed for v in m[key]]
            out[key] = {'n': len(flat), 'mean': _mean(flat), 'max': max(flat) if flat else None}
        elif key == 'ms_max':
            out[key] = max(m[key] for m in per_seed)
        elif key.endswith('_frames') or key == 'id_switches':
            out[key] = float(np.sum([m[key] for m in per_seed]))
        else:
            out[key] = _mean([m[key] for m in per_seed])
    return out


def _fmt(value, pct=False, digits=2):
    if value is None:
        return '-'
    return f'{100 * value:.0f}%' if pct else f'{value:.{digits}f}'


def markdown(results, seeds, sensor_config):
    agg = {n: {s: aggregate(r) for s, r in by.items()} for n, by in results.items()}
    lines = [f'# Whole-person tracking evaluation ({seeds} seeds per scenario, detector every '
             f'{sensor_config.detect_every} frame(s))', '',
             '`three_state` = Confirmed/Temporary/Unknown tracker with exponential decay. '
             '`face_lock` = Part 2 style lock that exists only while the target face is recognised.', '',
             '| scenario | system | coverage | coverage w/o face | wrong-hold | ghost s | ID sw. | '
             'CONF precision | TEMP precision | reacquire s |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for name, by in agg.items():
        for system in SYSTEMS:
            a = by[system]
            lines.append(
                f"| {name} | {system} | {_fmt(a['coverage'], True)} | "
                f"{_fmt(a['coverage_during_face_loss'], True)} | {_fmt(a['wrong_hold_rate'], True)} | "
                f"{_fmt(a['ghost_seconds'])} | {a['id_switches']:.0f} | "
                f"{_fmt(a['confirmed_precision'], True)} | {_fmt(a['temporary_precision'], True)} | "
                f"{_fmt(a['reacquire_s']['mean'])} |")
    lines += ['', '## CPU cost of the tracking logic (per frame, includes appearance + Kalman + association)', '',
              '| scenario | mean ms | p95 ms | max ms | embedder calls/frame (three_state) | (face_lock) |',
              '|---|---|---|---|---|---|']
    for name, by in agg.items():
        a, b = by['three_state'], by['face_lock']
        lines.append(f"| {name} | {a['ms_mean']:.2f} | {a['ms_p95']:.2f} | {a['ms_max']:.1f} | "
                     f"{a['verifications_per_frame']:.2f} | {b['verifications_per_frame']:.2f} |")
    lines += ['', 'Definitions: *coverage* = fraction of frames (after first confirmation) where the target '
              'is present and the tracker points at it while CONFIRMED or TEMPORARY. *wrong-hold* = fraction of held '
              'frames pointing at someone else. *ghost s* = seconds holding a lock while the target is not in the scene. '
              '*precision* = fraction of frames in that state that point at the target.']
    return '\n'.join(lines), agg


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--scenarios', nargs='+', default=list(scenarios.SCENARIOS))
    cli.add_argument('--seeds', type=int, default=5)
    cli.add_argument('--detect-every', type=int, default=1,
                     help='Person detector runs every N frames (CPU saving)')
    cli.add_argument('--out', default=str(ROOT / 'data' / 'reports'))
    args = cli.parse_args(argv)
    sensor_config = SensorConfig(detect_every=args.detect_every)
    results = evaluate(args.scenarios, args.seeds, sensor_config)
    text, agg = markdown(results, args.seeds, sensor_config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'every{args.detect_every}'
    (out / f'person_tracking_eval_{tag}.md').write_text(text + '\n', encoding='utf-8')
    (out / f'person_tracking_eval_{tag}.json').write_text(
        json.dumps({'python': platform.python_version(), 'cpu': platform.processor(),
                    'seeds': args.seeds, 'detect_every': args.detect_every, 'results': agg},
                   indent=2, default=float), encoding='utf-8')
    print(text)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
