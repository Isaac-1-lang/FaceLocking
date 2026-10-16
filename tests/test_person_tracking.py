import itertools
import unittest
from unittest.mock import Mock

import numpy as np

from src.person_tracking import detectors
from src.person_tracking.assignment import linear_assignment
from src.person_tracking.centering import CenteringController
from src.person_tracking.face_body import associate_faces, body_from_face, face_body_score
from src.person_tracking.identity import IdentityBelief, IdentityConfig, IdentityState as S
from src.person_tracking.kalman import BoxKalman
from src.person_tracking.pipeline import NeedleAdapter, PersonLockPipeline
from src.person_tracking.records import Detection, FaceObs, TargetEstimate, iou
from src.person_tracking.sim import metrics, runner, scenarios
from src.person_tracking.sim.sensors import SensorConfig
from src.person_tracking.sim.world import World
from src.person_tracking.tracker import TrackerConfig, WholePersonTracker
from src.landmarks import Face


class IdentityModelTests(unittest.TestCase):
    def belief(self):
        return IdentityBelief(IdentityConfig(half_life=2.0))

    def test_exponential_decay_halves_each_half_life(self):
        b = self.belief()
        b.confidence = 0.8
        b.advance(2.0)
        self.assertAlmostEqual(b.confidence, 0.4)
        b.advance(4.0)
        self.assertAlmostEqual(b.confidence, 0.2)

    def test_decay_rate_multiplier_and_time_monotonic(self):
        b = self.belief()
        b.confidence = 1.0
        b.advance(1.0, rate_multiplier=2.0)
        self.assertAlmostEqual(b.confidence, 0.5)
        with self.assertRaises(ValueError):
            b.advance(0.5)

    def test_weakest_passing_match_cannot_confirm_alone_but_two_can(self):
        b = self.belief()
        b.observe_match(0.45, 0.45)
        self.assertEqual(b.state, S.UNKNOWN)             # evidence building is not a claim
        b.observe_match(0.45, 0.45)
        self.assertEqual(b.state, S.CONFIRMED)

    def test_strong_match_confirms_immediately(self):
        b = self.belief()
        b.observe_match(0.75, 0.45)
        self.assertEqual(b.state, S.CONFIRMED)

    def test_temporary_only_reachable_from_confirmed(self):
        b = self.belief()
        b.observe_match(0.5, 0.45)                       # 0.55: between unknown and confirm
        self.assertEqual(b.state, S.UNKNOWN)
        b.observe_match(0.9, 0.45)
        self.assertEqual(b.state, S.CONFIRMED)
        b.advance(2.0)                                   # ~0.5
        b.advance(3.0)
        self.assertEqual(b.state, S.TEMPORARY)
        b.advance(20.0)
        self.assertEqual(b.state, S.UNKNOWN)

    def test_hysteresis_and_recovery_needs_a_face(self):
        b = self.belief()
        b.observe_match(0.9, 0.45)
        b.advance(1.0)                                   # ~0.71 -> still confirmed
        b.advance(1.4)
        self.assertEqual(b.state, S.CONFIRMED)           # between drop and confirm: stays
        b.advance(2.5)
        self.assertEqual(b.state, S.TEMPORARY)
        b.advance(2.6)                                   # decay alone never re-confirms
        self.assertNotEqual(b.state, S.CONFIRMED)

    def test_other_identity_needs_two_strikes_to_clear(self):
        b = self.belief()
        b.observe_match(0.9, 0.45)
        b.observe_other_identity()
        self.assertEqual(b.state, S.TEMPORARY)
        b.observe_other_identity()
        self.assertEqual(b.state, S.UNKNOWN)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            IdentityConfig(half_life=0)
        with self.assertRaises(ValueError):
            IdentityConfig(unknown_threshold=0.6, drop_threshold=0.5)
        with self.assertRaises(ValueError):
            IdentityConfig(evidence_floor=0.8)


class MathTests(unittest.TestCase):
    def test_assignment_matches_brute_force(self):
        rng = np.random.default_rng(3)
        for shape in ((3, 3), (2, 4), (4, 2), (1, 3), (5, 5)):
            cost = rng.random(shape)
            pairs = linear_assignment(cost)
            n = min(shape)
            self.assertEqual(len(pairs), n)
            best = min(sum(cost[r, c] for r, c in zip(rows, cols))
                       for rows in itertools.combinations(range(shape[0]), n)
                       for cols in itertools.permutations(range(shape[1]), n))
            self.assertAlmostEqual(sum(cost[r, c] for r, c in pairs), best)
        self.assertEqual(linear_assignment(np.zeros((0, 3))), [])

    def test_kalman_learns_velocity_and_coasts_with_damping(self):
        kf = BoxKalman([100, 100, 200, 400])
        for i in range(1, 30):
            kf.predict(1 / 15)
            kf.update([100 + 4 * i, 100, 200 + 4 * i, 400])
        self.assertAlmostEqual(kf.velocity[0] * (1 / 15), 4, delta=0.7)
        before = kf.center[0]
        for _ in range(60):
            kf.predict(1 / 15, coasting=True)
        self.assertGreater(kf.center[0], before)
        self.assertLess(abs(kf.velocity[0]), 0.4 * 4 * 15)   # damped to < 40% of the walking speed
        self.assertGreater(kf.uncertainty, 0.1)           # and honestly less certain

    def test_kalman_rejects_wild_measurement(self):
        kf = BoxKalman([100, 100, 200, 400])
        for _ in range(10):
            kf.predict(1 / 15)
            kf.update([100, 100, 200, 400])
        self.assertFalse(kf.update([600, 300, 700, 600]))
        self.assertAlmostEqual(kf.center[0], 150, delta=2)


class FaceBodyTests(unittest.TestCase):
    body = np.array([100.0, 100.0, 300.0, 600.0])

    def test_face_on_head_scores_high_and_off_body_zero(self):
        self.assertGreater(face_body_score([170, 100, 230, 170], self.body), 0.7)
        self.assertEqual(face_body_score([170, 400, 230, 470], self.body), 0.0)    # on the legs
        self.assertEqual(face_body_score([500, 100, 560, 170], self.body), 0.0)    # beside

    def test_pseudo_body_is_consistent_with_its_face(self):
        face = np.array([170.0, 100.0, 230.0, 180.0])
        self.assertGreater(face_body_score(face, body_from_face(face)), 0.8)

    def test_overlapping_bodies_flag_ambiguous_face(self):
        bodies = [[100, 100, 300, 600], [130, 100, 330, 600]]
        pairs, ambiguous = associate_faces([[170, 100, 230, 170]], bodies)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(ambiguous, {0})
        _, ambiguous = associate_faces([[170, 100, 230, 170]], [[100, 100, 300, 600], [600, 100, 800, 600]])
        self.assertEqual(ambiguous, set())

    def test_yaw_ratio(self):
        pts = np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], dtype=float)
        face = FaceObs([20, 30, 100, 110], pts)
        self.assertAlmostEqual(face.yaw_ratio, 0.0)
        pts[2, 0] = 75
        self.assertAlmostEqual(FaceObs([20, 30, 100, 110], pts).yaw_ratio, 0.375)


def person_box(cx, top=100, w=200, h=500):
    return Detection([cx - w / 2, top, cx + w / 2, top + h])


def face_at(cx, top=100, size=70, name='Isaac', score=0.8):
    obs = FaceObs([cx - size / 2, top, cx + size / 2, top + size * 1.2])
    obs.meta = (name, score)
    return obs


class TrackerLogicTests(unittest.TestCase):
    """Scripted observations through the real tracker (no models, no ground-truth leakage)."""

    def setUp(self):
        self.tracker = WholePersonTracker(TrackerConfig('Isaac', min_face_size=40))
        self.t = 0.0

    def step(self, persons, faces=(), dt=1 / 15, frame=None):
        self.t += dt
        return self.tracker.update(self.t, persons, list(faces), frame,
                                   recognize=lambda f: f.meta)

    def run_steps(self, n, persons, faces=()):
        est = None
        for _ in range(n):
            est = self.step(persons, faces)
        return est

    def test_acquire_from_unknown_only_on_target_face(self):
        persons = [person_box(200), person_box(600)]
        faces = [face_at(200, name='Unknown', score=0.2), face_at(600, name='Alice', score=0.8)]
        est = self.run_steps(15, persons, faces)
        self.assertEqual(est.state, S.UNKNOWN)          # strangers and other enrolled people never lock
        persons.append(person_box(1000))
        est = self.run_steps(4, persons, faces + [face_at(1000)])
        self.assertEqual(est.state, S.CONFIRMED)
        self.assertEqual(est.evidence, 'face')
        self.assertGreater(est.body_box[0], 800)

    def test_face_loss_keeps_body_then_decays_to_unknown(self):
        self.run_steps(10, [person_box(300)], [face_at(300)])
        est = self.step([person_box(300)])
        self.assertEqual(est.evidence, 'body')
        states = []
        for _ in range(15 * 12):
            est = self.step([person_box(300)])
            states.append(est.state)
        self.assertIn(S.TEMPORARY, states)
        self.assertEqual(states[-1], S.UNKNOWN)
        self.assertLess(states.index(S.TEMPORARY), states.index(S.UNKNOWN))
        self.assertNotIn(S.CONFIRMED, states[states.index(S.TEMPORARY):])   # never re-promoted by body alone

    def test_face_returns_and_reconfirms_same_track(self):
        self.run_steps(10, [person_box(300)], [face_at(300)])
        first = self.tracker.estimate.track_id
        self.run_steps(15 * 5, [person_box(300)])
        self.assertEqual(self.tracker.estimate.state, S.TEMPORARY)
        est = self.run_steps(3, [person_box(300)], [face_at(300)])
        self.assertEqual((est.state, est.track_id), (S.CONFIRMED, first))

    def test_known_other_person_on_target_track_removes_lock(self):
        self.run_steps(10, [person_box(300)], [face_at(300)])
        self.run_steps(30, [person_box(300)], [face_at(300, name='Alice')])
        self.assertEqual(self.tracker.estimate.state, S.UNKNOWN)

    def test_only_one_track_can_hold_the_target(self):
        persons = [person_box(250), person_box(750)]
        self.run_steps(10, persons, [face_at(250)])
        self.assertEqual(self.tracker.estimate.state, S.CONFIRMED)
        self.run_steps(10, persons, [face_at(750)])              # face evidence moves the lock
        holders = [t for t in self.tracker.tracks if t.state != S.UNKNOWN]
        self.assertEqual(len(holders), 1)
        self.assertGreater(self.tracker.estimate.body_box[0], 500)

    def test_detector_skipped_frames_do_not_count_as_loss(self):
        self.run_steps(10, [person_box(300)], [face_at(300)])
        for _ in range(4):
            est = self.step(None)                                # detector skipped, no face
        self.assertEqual(est.state, S.CONFIRMED)
        self.assertEqual(len(self.tracker.tracks), 1)

    def test_face_only_mode_builds_body_from_face(self):
        est = None
        for _ in range(6):
            est = self.step(None, [face_at(300)])
        self.assertEqual(est.state, S.CONFIRMED)
        self.assertEqual(len([t for t in self.tracker.tracks if t.hits >= 2]), 1)

    def test_verification_budget_per_frame(self):
        self.tracker = WholePersonTracker(TrackerConfig('Isaac', min_face_size=40, max_verifications=1))
        persons = [person_box(150), person_box(500), person_box(850)]
        faces = [face_at(150, name='Unknown', score=0.1), face_at(500, name='Unknown', score=0.1),
                 face_at(850, name='Unknown', score=0.1)]
        self.step(persons, faces)
        self.step(persons, faces)
        self.assertLessEqual(self.tracker.verifications, 1)

    def test_turned_or_tiny_faces_are_not_sent_to_the_embedder(self):
        calls = []
        tracker = WholePersonTracker(TrackerConfig('Isaac', min_face_size=60))
        tiny = FaceObs([280, 100, 320, 150])
        turned = FaceObs([270, 100, 340, 180],
                         np.array([[290, 120], [320, 120], [338, 135], [292, 160], [318, 160]], dtype=float))
        self.assertGreater(turned.yaw_ratio, 0.5)
        for i in range(10):
            tracker.update(i / 15, [person_box(300)], [tiny], recognize=lambda f: calls.append(1))
            tracker.update((i + 0.5) / 15, [person_box(300)], [turned], recognize=lambda f: calls.append(1))
        self.assertEqual(calls, [])

    def test_ghost_track_dies_and_identity_requires_a_new_match(self):
        self.run_steps(10, [person_box(300)], [face_at(300)])
        self.run_steps(15 * 10, [])
        self.assertEqual(self.tracker.estimate.state, S.UNKNOWN)
        self.assertEqual(self.tracker.estimate.confidence, 0.0)

    def test_leaving_the_frame_decays_faster_than_staying(self):
        def seconds_until_unknown(walks_out):
            tracker = WholePersonTracker(TrackerConfig('Isaac', min_face_size=40))
            t, x = 0.0, 400.0
            for i in range(60):                          # walk (or stand), face verified
                t += 1 / 15
                x += 20 if walks_out else 0
                tracker.update(t, [person_box(x)], [face_at(x)], recognize=lambda f: f.meta,
                               frame_shape=(720, 1280, 3))
            vanished = t
            for _ in range(15 * 20):
                t += 1 / 15
                tracker.update(t, [], [], frame_shape=(720, 1280, 3))
                if tracker.estimate.state == S.UNKNOWN:
                    break
            return t - vanished
        self.assertLess(seconds_until_unknown(True), 0.75 * seconds_until_unknown(False))


class CenteringTests(unittest.TestCase):
    def estimate(self, state, face=True, conf=0.9, evidence='face'):
        return TargetEstimate(state, conf, 1, np.array([400.0, 50, 600, 650]),
                              np.array([470.0, 60, 530, 140]) if face else None, evidence)

    def test_unknown_gives_no_command(self):
        c = CenteringController()
        self.assertIsNone(c.update(TargetEstimate(S.UNKNOWN, 0.0), (720, 1280), 0.0))

    def test_signs_deadzone_and_gain(self):
        c = CenteringController(smooth_tau=0)
        aim = c.update(self.estimate(S.CONFIRMED), (720, 1280), 0.0)
        self.assertEqual(aim.horizontal, 'LEFT')
        self.assertLess(aim.error_x, 0)
        self.assertEqual(aim.gain, 1.0)
        centred = CenteringController(smooth_tau=0).update(
            TargetEstimate(S.CONFIRMED, 0.9, 1, np.array([540.0, 50, 740, 650]), None, 'body'), (720, 1280), 0.0)
        self.assertEqual(centred.horizontal, 'CENTER')
        weak = CenteringController(smooth_tau=0).update(self.estimate(S.TEMPORARY, False, 0.3, 'body'),
                                                        (720, 1280), 0.0)
        self.assertLess(weak.gain, 1.0)
        self.assertGreaterEqual(weak.gain, 0.35)

    def test_no_jump_when_face_disappears(self):
        c = CenteringController(smooth_tau=0)
        for i in range(30):
            c.update(self.estimate(S.CONFIRMED), (720, 1280), i / 15)
        with_face = c.point.copy()
        aim = c.update(self.estimate(S.CONFIRMED, face=False, evidence='body'), (720, 1280), 2.1)
        self.assertLess(np.linalg.norm(aim.point - with_face), 5.0)
        self.assertEqual(aim.source, 'body-head')

    def test_smoothing_follows_with_time_constant(self):
        c = CenteringController(smooth_tau=0.5)
        c.update(self.estimate(S.CONFIRMED), (720, 1280), 0.0)
        moved = TargetEstimate(S.CONFIRMED, 0.9, 1, np.array([800.0, 50, 1000, 650]),
                               np.array([870.0, 60, 930, 140]), 'face')
        before = c.point[0]
        aim = c.update(moved, (720, 1280), 0.5)
        self.assertGreater(aim.point[0], before)
        self.assertLess(aim.point[0], 900)


class DetectorTests(unittest.TestCase):
    def test_decode_yolov8_layout_and_letterbox_inverse(self):
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        out[0, :4, 0] = [320, 320, 100, 200]        # cx, cy, w, h in 640 letterboxed space
        out[0, 4, 0] = 0.9                          # person score
        out[0, 4:, 1] = 0.0
        out[0, :4, 2] = [100, 100, 20, 20]
        out[0, 5, 2] = 0.95                         # a different class: ignored
        boxes = detectors.decode_yolo(out, ratio=0.5, pad=(0, 80))
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0].box, [540.0, 280.0, 740.0, 680.0])

    def test_decode_yolov5_objectness_layout(self):
        out = np.zeros((1, 25200, 85), dtype=np.float32)
        out[0, 0, :4] = [100, 100, 50, 100]
        out[0, 0, 4], out[0, 0, 5] = 0.9, 0.8
        out[0, 1, :4] = [300, 300, 50, 100]
        out[0, 1, 4], out[0, 1, 5] = 0.2, 0.9       # low objectness
        boxes = detectors.decode_yolo(out, ratio=1.0, pad=(0, 0))
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].score, 0.72, places=2)

    def test_scheduler_returns_none_between_runs(self):
        inner = Mock()
        inner.detect.return_value = [1]
        s = detectors.Scheduled(inner, every=3)
        self.assertEqual([s.detect(None) for _ in range(6)], [[1], None, None, [1], None, None])
        with self.assertRaises(ValueError):
            detectors.Scheduled(inner, every=0)

    def test_hog_runs_on_cpu_and_returns_list(self):
        found = detectors.HogPersonDetector().detect(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(found, [])

    def test_factory_validation(self):
        self.assertIsNone(detectors.make_person_detector('none'))
        with self.assertRaises(ValueError):
            detectors.make_person_detector('yolo')
        with self.assertRaises(ValueError):
            detectors.make_person_detector('bogus')


class PipelineTests(unittest.TestCase):
    def test_pipeline_uses_expensive_verifier_sparingly_and_adapts_to_needle(self):
        face = Face(np.array([270.0, 100.0, 80.0, 96.0]),
                    np.array([[285, 125], [320, 125], [302, 140], [288, 165], [318, 165]], dtype=float), 0.99)
        detector = Mock()
        detector.detect.return_value = [face]
        verifier = Mock(return_value=('Isaac', 0.8))
        person = Mock()
        person.detect.return_value = [Detection([200, 100, 400, 600])]
        pipeline = PersonLockPipeline(TrackerConfig('Isaac', min_face_size=40), detector, verifier, person)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        results = [pipeline.process(frame, i / 15) for i in range(30)]
        self.assertEqual(results[-1].estimate.state, S.CONFIRMED)
        self.assertLess(verifier.call_count, 12)                 # not one ArcFace call per frame
        publisher = Mock()
        adapter = NeedleAdapter(publisher, 'Isaac')
        adapter.publish(results[-1])
        proxy, aim = publisher.publish.call_args[0]
        self.assertEqual((proxy.state.name, proxy.target_name), ('LOCKED', 'Isaac'))
        self.assertIs(aim, results[-1].aim)


class SimulationRegressionTests(unittest.TestCase):
    """End-to-end behaviour on the scripted real-world situations, with fixed seeds."""

    def run_scenario(self, name, seed=0, **kwargs):
        scenario = scenarios.get(name)
        for key, value in kwargs.items():
            setattr(scenario, key, value)
        return metrics.score(runner.run(scenario, seed, 'three_state', SensorConfig(), None, World(scenario, seed)))

    def test_head_turn_keeps_target_without_a_face(self):
        m = self.run_scenario('head_turn')
        self.assertGreaterEqual(m['coverage_during_face_loss'], 0.95)
        self.assertEqual(m['wrong_hold_frames'], 0)
        self.assertGreater(m['temporary_seconds'], 1.0)          # it actually used the TEMPORARY state

    def test_long_face_loss_decays_to_unknown_then_recovers(self):
        m = self.run_scenario('long_turn')
        self.assertGreater(m['unknown_seconds'], 2.0)
        self.assertEqual(m['wrong_hold_frames'], 0)
        self.assertTrue(m['reacquire_s'])

    def test_crossing_never_transfers_identity(self):
        for name in ('path_crossing', 'crossing_same_colour'):
            m = self.run_scenario(name)
            self.assertEqual(m['wrong_confirmed_frames'], 0, name)
            self.assertEqual(m['id_switches'], 0, name)
            self.assertGreaterEqual(m['coverage'], 0.9, name)

    def test_no_lock_when_target_absent(self):
        m = self.run_scenario('strangers_only', lookalike_rate=0.0)
        self.assertEqual(m['ghost_seconds'], 0.0)
        self.assertEqual(m['confirmed_seconds'] + m['temporary_seconds'], 0.0)

    def test_leaving_the_scene_does_not_leave_a_long_ghost_lock(self):
        m = self.run_scenario('leave_return')
        self.assertLess(m['max_ghost_run_s'], 4.0)

    def test_confidence_is_informative(self):
        m = self.run_scenario('occlusion')
        self.assertEqual(m['wrong_confirmed_frames'], 0)
        if m['mean_conf_wrong'] is not None:
            self.assertLess(m['mean_conf_wrong'], m['mean_conf_correct'])

    def test_cpu_budget_of_tracking_logic(self):
        m = self.run_scenario('crowd')
        self.assertLess(m['ms_mean'], 15.0)                      # ample margin for 15+ FPS on a laptop CPU
        self.assertLessEqual(m['verifications_per_frame'], 1.0)

    def test_skipping_person_detector_still_tracks(self):
        scenario = scenarios.get('head_turn')
        run = runner.run(scenario, 0, 'three_state', SensorConfig(detect_every=4), None, World(scenario, 0))
        m = metrics.score(run)
        self.assertGreaterEqual(m['coverage_during_face_loss'], 0.9)
        self.assertEqual(m['wrong_hold_frames'], 0)


class RecordTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(iou([0, 0, 1, 1], [5, 5, 6, 6]), 0.0)


if __name__ == '__main__':
    unittest.main()
