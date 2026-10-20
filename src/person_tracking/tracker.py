"""Whole-person tracker that keeps the target through temporary face loss.

Per frame the caller supplies person boxes (or ``None`` when the person
detector was skipped to save CPU), face observations, and optionally a
``recognize(face) -> (name, similarity)`` callback. The callback is expensive
(ArcFace), so the tracker decides *which* faces deserve it, within a
per-frame budget. Everything else here is NumPy on a handful of boxes.
"""
from dataclasses import dataclass, field, replace
 
import numpy as np

from . import appearance
from .assignment import linear_assignment
from .face_body import associate_faces, body_from_face
from .identity import IdentityBelief, IdentityConfig, IdentityState
from .kalman import BoxKalman
from .records import (Detection, TargetEstimate, box_center, box_size, iou,
                      overlap_min)

_BIG = 1e6


@dataclass(frozen=True)
class TrackerConfig:
    target: str
    match_threshold: float = 0.45      # cosine similarity a face must reach (as in Part 1)
    min_face_size: float = 36.0        # px; smaller faces are not sent to the embedder
    yaw_limit: float = 0.5             # skip faces turned beyond this (see FaceObs.yaw_ratio)
    verify_interval: float = 1.0       # seconds between re-verifications of a held target
    probe_interval: float = 0.4        # seconds between probes of tracks that are not the target
    other_interval: float = 2.0        # ...of tracks already recognised as someone else
    max_verifications: int = 1         # embedder calls per frame (CPU budget)
    min_hits: int = 2                  # measurements before a track is trusted
    max_age: float = 1.5               # seconds an UNKNOWN track coasts before deletion
    coast_grace: float = 0.4           # seconds without a measurement before a track is 'coasting'
    max_identity_age: float = 20.0     # hard cap for identity-bearing tracks
    gate_distance: float = 0.45        # centre distance / body height always allowed
    contest_overlap: float = 0.25      # body overlap (of the smaller box) that counts as crossing
    contest_clear_time: float = 0.3    # seconds apart before a crossing is considered over
    contest_decay_multiplier: float = 2.5
    coast_decay_multiplier: float = 1.5
    exit_decay_multiplier: float = 4.0  # coasting track whose predicted centre left the image
    appearance_interval: float = 0.4
    min_appearance: float = 0.3        # colour similarity below which a box cannot be the same person
    identity: IdentityConfig = field(default_factory=IdentityConfig)

    def with_identity(self, **kwargs):
        return replace(self, identity=replace(self.identity, **kwargs))


class Track:
    def __init__(self, track_id, box, t, source, config):
        self.id = track_id
        self.kf = BoxKalman(box)
        self.belief = IdentityBelief(config.identity, now=t)
        self.hits = 1
        self.born = t
        self.gap = 0.0               # seconds since the last measurement (body or face-derived)
        self.last_measure_t = t
        self.source = source         # source of the latest measurement
        self.face = None             # FaceObs mapped to this track on the current frame
        self.face_ambiguous = False
        self.face_t = None           # last time any face was mapped here
        self.hist_ref = None
        self.hist_t = -1e9
        self.contested = False
        self.partners = set()
        self.clear_since = None
        self.next_verify = 0.0
        self.last_verify = -1e9
        self.other_name = None

    @property
    def box(self):
        return self.kf.box

    @property
    def state(self):
        return self.belief.state


class WholePersonTracker:
    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.t = None
        self._next_id = 1
        self.estimate = TargetEstimate(IdentityState.UNKNOWN, 0.0, reason='Waiting for target')
        self.events = []             # (track_id, old_state, new_state, reason) of the last update
        self.verifications = 0       # recognize() calls on the last update

    # ------------------------------------------------------------------ public
    def update(self, t, persons=None, faces=(), frame=None, recognize=None, frame_shape=None):
        cfg = self.config
        shape = frame.shape if frame is not None else frame_shape
        self.events = []
        self.verifications = 0
        dt = 0.0 if self.t is None else max(t - self.t, 0.0)
        self.t = t if self.t is None else max(t, self.t)
        t = self.t
        for tr in self.tracks:
            tr.face, tr.face_ambiguous = None, False
            tr.gap = t - tr.last_measure_t
            tr.kf.predict(dt, coasting=tr.gap > cfg.coast_grace)
        before = {tr.id: tr.state for tr in self.tracks}

        detections = self._detections(persons, faces)
        matched = self._associate(detections, t, frame)
        self._spawn(detections, matched, t)
        self._map_faces(faces)
        self._update_contests(t, frame)
        self._update_appearance(t, frame)
        for tr in self.tracks:
            multiplier = 1.0
            if tr.contested:
                multiplier = cfg.contest_decay_multiplier
            elif tr.gap > cfg.coast_grace:
                multiplier = cfg.coast_decay_multiplier
                if shape is not None and self._outside(tr, shape):
                    multiplier = cfg.exit_decay_multiplier
            tr.belief.advance(t, multiplier)
        if recognize is not None:
            self._verify(t, recognize)
        self._enforce_unique()
        self._prune(t)
        for tr in self.tracks:
            if before.get(tr.id) not in (None, tr.state):
                self.events.append((tr.id, before[tr.id], tr.state, 'identity update'))
        self.estimate = self._estimate()
        return self.estimate

    def track_by_id(self, track_id):
        return next((tr for tr in self.tracks if tr.id == track_id), None)

    @staticmethod
    def _complete(tr, box):
        cx, cy, w, h = tr.kf.x[:4]
        x1, y1, x2, y2 = box
        if x2 - x1 < 0.9 * w:
            cx = float(np.clip(cx, x2 - w / 2, x1 + w / 2))
        else:
            cx, w = (x1 + x2) / 2, x2 - x1
        if y2 - y1 < 0.9 * h:
            cy = float(np.clip(cy, y2 - h / 2, y1 + h / 2))
        else:
            cy, h = (y1 + y2) / 2, y2 - y1
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @staticmethod
    def _outside(tr, shape, margin=0.1):
        cx, cy = tr.kf.center
        height, width = shape[:2]
        return not (-margin * width <= cx <= (1 + margin) * width
                    and -margin * height <= cy <= (1 + margin) * height)

    # ------------------------------------------------------------ measurements
    def _detections(self, persons, faces):
        """Person boxes plus a pseudo-body for every face no person box explains."""
        detections = [Detection(p.box, p.score, p.source) if isinstance(p, Detection)
                      else Detection(p) for p in (persons or [])]
        if faces:
            explained = set()
            if detections:
                explained = set(associate_faces([f.box for f in faces],
                                                [d.box for d in detections])[0])
            detections += [Detection(body_from_face(f.box), 0.5, 'face')
                           for i, f in enumerate(faces) if i not in explained]
        return detections

    def _associate(self, detections, t, frame):
        cfg = self.config
        matched = set()
        if self.tracks and detections:
            hists = [None] * len(detections)
            if frame is not None and len(self.tracks) > 1:
                hists = [appearance.torso_histogram(frame, d.box) for d in detections]
            cost = np.full((len(self.tracks), len(detections)), _BIG)
            for ti, tr in enumerate(self.tracks):
                tbox, th = tr.box, max(tr.kf.x[3], 1.0)
                gate = cfg.gate_distance + 0.5 * min(tr.gap, 2.0)
                for di, det in enumerate(detections):
                    overlap = iou(tbox, det.box)
                    dist = float(np.linalg.norm(box_center(tbox) - box_center(det.box)) / th)
                    if overlap <= 0.05 and dist > gate:
                        continue
                    c = (1.0 - overlap) + 0.7 * min(dist, 2.0)
                    sim = appearance.similarity(hists[di], tr.hist_ref)
                    if sim is not None and not tr.contested:
                        if sim < cfg.min_appearance:
                            continue         # clothing colours that different: not this person
                        c += 0.6 * (1.0 - sim)
                    cost[ti, di] = c
            for ti, di in linear_assignment(cost):
                if cost[ti, di] >= _BIG:
                    continue
                tr, det = self.tracks[ti], detections[di]
                dw, dh = box_size(det.box)
                ratio = min(dw / max(tr.kf.x[2], 1.0), dh / max(tr.kf.x[3], 1.0))
                top = max(dw / max(tr.kf.x[2], 1.0), dh / max(tr.kf.x[3], 1.0))
                matched.add(di)
                if tr.contested and top > 1.2:
                    continue                         # two people merged into one box: ignore it
                noise = 3.0 if det.source == 'face' else 1.0
                box = det.box
                if ratio < 0.9 and det.source == 'person':
                    # Truncated by an occluder or the image edge: the true body *contains* this
                    # sliver. Using the sliver's own centre would bias position and velocity, so
                    # take the full-size box closest to the prediction that still contains it.
                    box = self._complete(tr, det.box)
                    noise = 2.0
                elif top > 1.25:
                    noise = max(noise, 4.0)
                if not tr.kf.update(box, noise):
                    continue                         # inconsistent with the motion model: ignore
                tr.hits += 1
                tr.gap = 0.0
                tr.last_measure_t = t
                tr.source = det.source
        return matched

    def _spawn(self, detections, matched, t):
        for di, det in enumerate(detections):
            if di in matched:
                continue
            # A box that mostly coincides with a live track is a split/merged detection.
            if any(iou(tr.box, det.box) > 0.35 or overlap_min(tr.box, det.box) > 0.6
                   for tr in self.tracks):
                continue
            self.tracks.append(Track(self._next_id, det.box, t, det.source, self.config))
            self._next_id += 1

    def _map_faces(self, faces):
        live = [tr for tr in self.tracks if tr.hits >= 1]
        if not faces or not live:
            return
        pairs, ambiguous = associate_faces([f.box for f in faces], [tr.box for tr in live])
        for fi, bi in pairs.items():
            tr = live[bi]
            tr.face, tr.face_ambiguous, tr.face_t = faces[fi], fi in ambiguous, self.t

    # --------------------------------------------------------- crossings (swaps)
    def _trusted(self, tr):
        return tr.hits >= self.config.min_hits and tr.gap < 1.0

    @staticmethod
    def _near(a, b, threshold):
        """Do two people overlap or nearly touch? Boxes are grown 15% per side because a
        partly hidden person is detected as only the visible sliver."""
        def grown(tr):
            x1, y1, x2, y2 = tr.box
            pad = 0.15 * (x2 - x1)
            return np.array([x1 - pad, y1, x2 + pad, y2])
        return overlap_min(grown(a), grown(b)) > threshold

    def _update_contests(self, t, frame):
        cfg = self.config
        now = {}
        trusted = [tr for tr in self.tracks if self._trusted(tr)]
        for i, a in enumerate(trusted):
            for b in trusted[i + 1:]:
                if self._near(a, b, cfg.contest_overlap):
                    now.setdefault(a.id, set()).add(b.id)
                    now.setdefault(b.id, set()).add(a.id)
        for tr in self.tracks:
            if tr.id in now:
                tr.contested, tr.clear_since = True, None
                tr.partners |= now[tr.id]
            elif tr.contested:
                tr.clear_since = t if tr.clear_since is None else tr.clear_since
                if t - tr.clear_since >= cfg.contest_clear_time:
                    self._resolve_contest(tr, frame)
                    tr.contested, tr.clear_since = False, None
                    tr.partners.clear()

    def _resolve_contest(self, tr, frame):
        """After two people separate, decide whether this track still follows the same person.

        Colour is compared with this track's signature and its partners' signatures
        recorded *before* they overlapped. A better match with a partner means the
        tracker swapped them: the identity claim is withdrawn, never transferred.
        """
        if tr.state == IdentityState.UNKNOWN and tr.belief.confidence == 0.0:
            return
        cur = None if frame is None else appearance.torso_histogram(frame, tr.box)
        own = appearance.similarity(cur, tr.hist_ref)
        others = [appearance.similarity(cur, p.hist_ref) for pid in tr.partners
                  if (p := self.track_by_id(pid)) is not None]
        others = [s for s in others if s is not None]
        if own is None or not others:
            tr.belief.penalize(0.6)
        elif max(others) > own + 0.08:
            tr.belief.reset()
            self.events.append((tr.id, None, IdentityState.UNKNOWN, 'swap suspected after crossing'))
        elif own >= 0.6 and own > max(others) + 0.08:
            tr.belief.penalize(0.9)
        else:
            tr.belief.penalize(0.5)     # colours cannot tell the people apart

    def _update_appearance(self, t, frame):
        if frame is None:
            return
        for tr in self.tracks:
            if (tr.contested or tr.gap > 0 or t - tr.hist_t < self.config.appearance_interval
                    or tr.hits < self.config.min_hits):
                continue
            # Do not learn colours from a box that is even partly hidden behind someone.
            if any(o is not tr and self._near(o, tr, 0.05) for o in self.tracks):
                continue
            tr.hist_ref = appearance.blend(tr.hist_ref, appearance.torso_histogram(frame, tr.box))
            tr.hist_t = t

    # ------------------------------------------------------------ identity logic
    def _verify(self, t, recognize):
        cfg = self.config
        candidates = []
        for tr in self.tracks:
            face = tr.face
            if face is None or tr.face_ambiguous:
                continue
            if face.size < cfg.min_face_size or face.yaw_ratio > cfg.yaw_limit:
                continue
            if t < tr.next_verify:
                continue
            held = 0 if tr.state != IdentityState.UNKNOWN else 1
            candidates.append((held, tr.last_verify, tr))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for _, _, tr in candidates[:cfg.max_verifications]:
            name, score = recognize(tr.face)
            self.verifications += 1
            tr.last_verify = t
            if name == cfg.target:
                tr.other_name = None
                tr.belief.observe_match(score, cfg.match_threshold, t)
                if tr.state == IdentityState.CONFIRMED:
                    tr.next_verify = t + cfg.verify_interval
                    self._supersede(tr)
                else:
                    tr.next_verify = t      # still building evidence: confirm on the next frame
            elif name in (None, 'Unknown'):
                tr.belief.observe_unmatched()
                tr.next_verify = t + cfg.probe_interval
            elif name == 'Too small':
                tr.next_verify = t + cfg.probe_interval
            else:
                tr.other_name = name
                tr.belief.observe_other_identity()
                tr.next_verify = t + cfg.other_interval

    def _supersede(self, winner):
        """Face evidence outranks inference: other tracks cannot also be the target."""
        for tr in self.tracks:
            if tr is not winner and tr.belief.confidence > 0.0 and tr.state != IdentityState.UNKNOWN:
                tr.belief.reset()
                self.events.append((tr.id, None, IdentityState.UNKNOWN, 'superseded by face match'))

    def _enforce_unique(self):
        holders = [tr for tr in self.tracks if tr.state != IdentityState.UNKNOWN]
        if len(holders) > 1:
            keep = max(holders, key=lambda tr: (tr.belief.confidence, tr.belief.last_evidence_t or 0))
            for tr in holders:
                if tr is not keep:
                    tr.belief.reset()

    def _prune(self, t):
        cfg = self.config
        keep = []
        for tr in self.tracks:
            if tr.hits < cfg.min_hits:
                limit = 0.3
            elif tr.state == IdentityState.UNKNOWN:
                limit = cfg.max_age
            else:
                limit = cfg.max_identity_age
            if tr.gap <= limit:
                keep.append(tr)
        self.tracks = keep

    # ------------------------------------------------------------------ output
    def _estimate(self):
        holders = [tr for tr in self.tracks if tr.state != IdentityState.UNKNOWN]
        if not holders:
            best = max((tr.belief.confidence for tr in self.tracks), default=0.0)
            return TargetEstimate(IdentityState.UNKNOWN, float(best),
                                  reason='No track is verified as the target')
        tr = max(holders, key=lambda x: x.belief.confidence)
        face_visible = tr.face is not None and not tr.face_ambiguous
        if face_visible:
            evidence = 'face'
        elif tr.gap <= self.config.coast_grace:
            evidence = 'body'
        else:
            evidence = 'predicted'
        since = tr.belief.seconds_since_face
        if tr.state == IdentityState.CONFIRMED:
            reason = ('Face verified' if since is not None and since < 0.05
                      else f'Verified {since:.1f}s ago' if since is not None else 'Confirmed')
        else:
            reason = f'Face not verified for {since:.1f}s: following body, confidence decaying' \
                if since is not None else 'Following body'
        if tr.contested:
            reason += ' | crossing in progress'
        if evidence == 'predicted':
            reason += ' | not currently detected'
        return TargetEstimate(
            tr.state, float(tr.belief.confidence), tr.id, tr.box.copy(),
            tr.face.box.copy() if face_visible else None, evidence, reason, since)
