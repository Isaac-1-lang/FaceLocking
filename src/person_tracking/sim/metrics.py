"""Ground-truth scoring of a simulated run."""
import numpy as np

from ..identity import IdentityState

C, T, U = IdentityState.CONFIRMED, IdentityState.TEMPORARY, IdentityState.UNKNOWN


def _longest_run(flags, fps):
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best / fps


def score(run, reacquire_gap=1.5):
    """Return a flat dict of metrics. Rates are fractions in [0, 1]; times in seconds."""
    f, fps, pid = run.frames, run.fps, run.target_pid
    n = len(f)
    held = np.array([x.state in (C, T) for x in f])
    conf = np.array([x.state == C for x in f])
    temp = np.array([x.state == T for x in f])
    present = np.array([x.target_present for x in f])
    usable = np.array([x.target_face_usable for x in f])
    on_target = np.array([x.points_to == pid and pid is not None for x in f])
    correct = held & on_target
    wrong = held & ~on_target & present
    ghost = held & ~present
    out = {'frames': n}

    confirmed_correct = np.nonzero(conf & on_target)[0]
    first = int(confirmed_correct[0]) if confirmed_correct.size else None
    out['time_to_first_confirm_s'] = None if first is None else first / fps
    after = np.arange(n) >= (first if first is not None else n)

    scope = present & after
    out['coverage'] = float(correct[scope].mean()) if scope.any() else None
    loss = scope & ~usable
    out['coverage_during_face_loss'] = float(correct[loss].mean()) if loss.any() else None
    out['face_loss_seconds'] = float(loss.sum() / fps)

    out['wrong_hold_frames'] = int(wrong.sum())
    out['wrong_confirmed_frames'] = int((wrong & conf).sum())
    out['wrong_hold_rate'] = float(wrong.sum() / held.sum()) if held.any() else 0.0
    out['ghost_seconds'] = float(ghost.sum() / fps)
    out['max_ghost_run_s'] = _longest_run(ghost, fps)

    switches, last = 0, None
    for x, h in zip(f, held):
        if h and x.points_to is not None:
            if last is not None and x.points_to != last:
                switches += 1
            last = x.points_to
    out['id_switches'] = switches

    latencies, i = [], 0
    while i < n:
        if usable[i] and not usable[max(i - int(reacquire_gap * fps), 0):i].any() and i >= int(reacquire_gap * fps):
            j = i
            while j < n and not (conf[j] and on_target[j]):
                j += 1
            if j < n:
                latencies.append((j - i) / fps)
            i = j + 1
        else:
            i += 1
    out['reacquire_s'] = latencies

    for name, mask in (('confirmed', conf), ('temporary', temp)):
        out[f'{name}_precision'] = float((mask & on_target).sum() / mask.sum()) if mask.any() else None
        out[f'{name}_seconds'] = float(mask.sum() / fps)
    out['unknown_seconds'] = float((~held).sum() / fps)
    right_conf = [x.confidence for x, c in zip(f, correct) if c]
    wrong_conf = [x.confidence for x, w in zip(f, wrong | ghost) if w]
    out['mean_conf_correct'] = float(np.mean(right_conf)) if right_conf else None
    out['mean_conf_wrong'] = float(np.mean(wrong_conf)) if wrong_conf else None
    errs = [x.aim_error for x, c in zip(f, correct) if c and x.aim_error is not None]
    out['aim_error_frac_width'] = float(np.mean(errs)) if errs else None
    ms = np.array([x.ms for x in f])
    out['ms_mean'], out['ms_p95'], out['ms_max'] = float(ms.mean()), float(np.percentile(ms, 95)), float(ms.max())
    out['verifications_per_frame'] = float(np.mean([x.verifications for x in f]))
    return out
