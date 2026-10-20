# Whole-person tracking evaluation (5 seeds per scenario, detector every 1 frame(s))

`three_state` = Confirmed/Temporary/Unknown tracker with exponential decay. `face_lock` = Part 2 style lock that exists only while the target face is recognised.

| scenario | system | coverage | coverage w/o face | wrong-hold | ghost s | ID sw. | CONF precision | TEMP precision | reacquire s |
|---|---|---|---|---|---|---|---|---|---|
| head_turn | three_state | 100% | 100% | 0% | 0.00 | 0 | 100% | 100% | 0.25 |
| head_turn | face_lock | 63% | 0% | 0% | 0.00 | 0 | 100% | - | 0.04 |
| long_turn | three_state | 72% | 54% | 0% | 0.00 | 0 | 100% | 100% | 0.12 |
| long_turn | face_lock | 41% | 0% | 0% | 0.00 | 0 | 100% | - | 0.01 |
| occlusion | three_state | 93% | 71% | 7% | 0.00 | 0 | 100% | 61% | 0.01 |
| occlusion | face_lock | 76% | 0% | 0% | 0.00 | 0 | 100% | - | 0.00 |
| long_occlusion | three_state | 46% | 16% | 4% | 0.00 | 0 | 100% | 87% | 0.00 |
| long_occlusion | face_lock | 35% | 0% | 0% | 0.00 | 0 | 100% | - | 0.00 |
| path_crossing | three_state | 100% | 97% | 0% | 0.00 | 0 | 100% | 100% | 0.00 |
| path_crossing | face_lock | 86% | 0% | 0% | 0.00 | 0 | 100% | - | 0.00 |
| crossing_same_colour | three_state | 100% | 97% | 0% | 0.00 | 0 | 100% | 100% | 0.00 |
| crossing_same_colour | face_lock | 86% | 0% | 0% | 0.00 | 0 | 100% | - | 0.00 |
| leave_return | three_state | 98% | 70% | 0% | 1.57 | 0 | 100% | 54% | 0.00 |
| leave_return | face_lock | 92% | 0% | 0% | 0.00 | 0 | 100% | - | 0.00 |
| strangers_only | three_state | - | - | 0% | 0.29 | 0 | 0% | 0% | - |
| strangers_only | face_lock | - | - | 0% | 0.52 | 0 | 0% | - | - |
| crowd | three_state | 80% | 52% | 0% | 0.00 | 0 | 100% | 100% | 0.07 |
| crowd | face_lock | 60% | 1% | 0% | 0.00 | 0 | 100% | - | 0.00 |

## CPU cost of the tracking logic (per frame, includes appearance + Kalman + association)

| scenario | mean ms | p95 ms | max ms | embedder calls/frame (three_state) | (face_lock) |
|---|---|---|---|---|---|
| head_turn | 0.49 | 0.77 | 2.8 | 0.08 | 1.63 |
| long_turn | 0.48 | 0.74 | 1.6 | 0.07 | 1.39 |
| occlusion | 0.47 | 0.68 | 1.3 | 0.08 | 1.53 |
| long_occlusion | 0.34 | 0.61 | 1.2 | 0.06 | 1.30 |
| path_crossing | 0.57 | 0.73 | 46.6 | 0.10 | 1.83 |
| crossing_same_colour | 0.64 | 1.03 | 2.7 | 0.22 | 1.83 |
| leave_return | 0.46 | 0.78 | 1.4 | 0.07 | 1.50 |
| strangers_only | 0.74 | 1.16 | 2.7 | 0.19 | 1.82 |
| crowd | 1.76 | 3.18 | 15.9 | 0.28 | 2.81 |

Definitions: *coverage* = fraction of frames (after first confirmation) where the target is present and the tracker points at it while CONFIRMED or TEMPORARY. *wrong-hold* = fraction of held frames pointing at someone else. *ghost s* = seconds holding a lock while the target is not in the scene. *precision* = fraction of frames in that state that point at the target.
