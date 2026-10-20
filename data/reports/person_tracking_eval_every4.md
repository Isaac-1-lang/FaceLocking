# Whole-person tracking evaluation (5 seeds per scenario, detector every 4 frame(s))

`three_state` = Confirmed/Temporary/Unknown tracker with exponential decay. `face_lock` = Part 2 style lock that exists only while the target face is recognised.

| scenario | system | coverage | coverage w/o face | wrong-hold | ghost s | ID sw. | CONF precision | TEMP precision | reacquire s |
|---|---|---|---|---|---|---|---|---|---|
| head_turn | three_state | 100% | 100% | 0% | 0.00 | 0 | 100% | 100% | 0.19 |
| head_turn | face_lock | 64% | 1% | 0% | 0.00 | 0 | 100% | - | 0.03 |
| long_turn | three_state | 71% | 52% | 0% | 0.00 | 0 | 100% | 100% | 0.17 |
| long_turn | face_lock | 42% | 0% | 0% | 0.00 | 0 | 100% | - | 0.03 |
| occlusion | three_state | 100% | 100% | 0% | 0.00 | 0 | 100% | 100% | 0.03 |
| occlusion | face_lock | 76% | 0% | 0% | 0.00 | 0 | 100% | - | 0.00 |
| long_occlusion | three_state | 44% | 14% | 7% | 0.00 | 0 | 100% | 76% | 0.00 |
| long_occlusion | face_lock | 35% | 0% | 0% | 0.00 | 0 | 100% | - | 0.00 |
| path_crossing | three_state | 99% | 90% | 0% | 0.00 | 0 | 100% | 98% | 0.00 |
| path_crossing | face_lock | 87% | 0% | 0% | 0.00 | 0 | 100% | - | 0.00 |
| crossing_same_colour | three_state | 99% | 90% | 0% | 0.00 | 0 | 100% | 98% | 0.00 |
| crossing_same_colour | face_lock | 87% | 0% | 0% | 0.00 | 0 | 100% | - | 0.00 |
| leave_return | three_state | 98% | 65% | 0% | 1.53 | 0 | 100% | 61% | 0.00 |
| leave_return | face_lock | 93% | 0% | 0% | 0.00 | 0 | 100% | - | 0.00 |
| strangers_only | three_state | - | - | 0% | 1.31 | 0 | 0% | 0% | - |
| strangers_only | face_lock | - | - | 0% | 0.72 | 0 | 0% | - | - |
| crowd | three_state | 79% | 51% | 0% | 0.00 | 0 | 100% | 100% | 0.14 |
| crowd | face_lock | 59% | 1% | 0% | 0.00 | 0 | 100% | - | 0.01 |

## CPU cost of the tracking logic (per frame, includes appearance + Kalman + association)

| scenario | mean ms | p95 ms | max ms | embedder calls/frame (three_state) | (face_lock) |
|---|---|---|---|---|---|
| head_turn | 0.46 | 0.68 | 1.2 | 0.08 | 1.64 |
| long_turn | 0.42 | 0.64 | 1.5 | 0.07 | 1.40 |
| occlusion | 0.42 | 0.62 | 1.1 | 0.08 | 1.52 |
| long_occlusion | 0.34 | 0.67 | 1.9 | 0.06 | 1.30 |
| path_crossing | 0.60 | 0.88 | 1.7 | 0.10 | 1.84 |
| crossing_same_colour | 0.63 | 0.83 | 60.4 | 0.22 | 1.84 |
| leave_return | 0.75 | 1.53 | 4.8 | 0.07 | 1.50 |
| strangers_only | 1.03 | 1.89 | 3.6 | 0.19 | 1.82 |
| crowd | 1.17 | 1.90 | 25.4 | 0.28 | 2.80 |

Definitions: *coverage* = fraction of frames (after first confirmation) where the target is present and the tracker points at it while CONFIRMED or TEMPORARY. *wrong-hold* = fraction of held frames pointing at someone else. *ghost s* = seconds holding a lock while the target is not in the scene. *precision* = fraction of frames in that state that point at the target.
