"""The honest three-state identity model with exponential confidence decay.

CONFIRMED  a face matched the target recently enough that we still trust it.
TEMPORARY  the target *was* confirmed, the face is no longer observed, and the
           body is followed on motion/appearance alone while confidence decays.
UNKNOWN    no trustworthy claim: never confirmed, ruled out, or decayed away.

Rules that keep the model honest:

* Confidence only *rises* through positive face evidence (a recognised match).
  Following a body, however well, never raises it.
* Between face observations it decays exponentially, ``c(t) = c0 * 0.5**(t/T)``
  with half-life ``T``; decay is faster while the track is contested by another
  person or is coasting without any measurement.
* TEMPORARY can only be entered from CONFIRMED. A fresh track building up
  evidence stays UNKNOWN until it crosses ``confirm_threshold``.
* Hysteresis (``confirm`` > ``drop`` > ``unknown``) prevents flicker.
"""
from dataclasses import dataclass
from enum import Enum


class IdentityState(Enum):
    CONFIRMED = 'CONFIRMED'
    TEMPORARY = 'TEMPORARY'
    UNKNOWN = 'UNKNOWN'


@dataclass(frozen=True)
class IdentityConfig:
    half_life: float = 3.0            # seconds for confidence to halve with no face evidence
    confirm_threshold: float = 0.70   # confidence needed to become/return to CONFIRMED
    drop_threshold: float = 0.50      # CONFIRMED -> TEMPORARY below this
    unknown_threshold: float = 0.20   # TEMPORARY -> UNKNOWN below this
    evidence_floor: float = 0.55      # weight of a just-passing face match
    evidence_saturation: float = 0.80  # similarity at which a match carries full weight
    unmatched_factor: float = 0.80    # multiplier on an evaluated, unrecognised frontal face
    other_identity_factor: float = 0.25  # multiplier when the face is a different enrolled person

    def __post_init__(self):
        if self.half_life <= 0:
            raise ValueError('half_life must be positive')
        if not 0 < self.unknown_threshold < self.drop_threshold < self.confirm_threshold <= 1:
            raise ValueError('Need 0 < unknown < drop < confirm <= 1')
        if not 0 < self.evidence_floor <= 1:
            raise ValueError('evidence_floor must be in (0, 1]')
        # A single just-passing match must NOT confirm on its own: two agreeing
        # observations are required, which suppresses one-frame false matches.
        if self.evidence_floor >= self.confirm_threshold:
            raise ValueError('evidence_floor must be below confirm_threshold')


class IdentityBelief:
    """Confidence that one track is the target, plus its derived discrete state."""

    def __init__(self, config=None, now=0.0):
        self.config = config or IdentityConfig()
        self.confidence = 0.0
        self.state = IdentityState.UNKNOWN
        self.t = now
        self.last_evidence_t = None   # last time a face matched the target
        self.other_strikes = 0

    @property
    def seconds_since_face(self):
        return None if self.last_evidence_t is None else self.t - self.last_evidence_t

    def advance(self, t, rate_multiplier=1.0):
        """Apply exponential decay up to time ``t``."""
        dt = t - self.t
        if dt > 0:
            self.confidence *= 0.5 ** (rate_multiplier * dt / self.config.half_life)
            self.t = t
        elif dt < 0:
            raise ValueError('time went backwards')
        self._refresh()

    def observe_match(self, similarity, threshold, t=None):
        """A face on this track matched the target with cosine ``similarity``."""
        cfg = self.config
        span = max(cfg.evidence_saturation - threshold, 1e-6)
        weight = min(1.0, cfg.evidence_floor + (1 - cfg.evidence_floor)
                     * max(0.0, (similarity - threshold) / span))
        self.confidence += (1.0 - self.confidence) * weight
        self.last_evidence_t = self.t if t is None else t
        self.other_strikes = 0
        self._refresh()

    def observe_unmatched(self):
        """A usable frontal face was evaluated and is not (confidently) the target."""
        self.confidence *= self.config.unmatched_factor
        self._refresh()

    def observe_other_identity(self):
        """A usable face matched a different enrolled person. Two strikes clear the belief."""
        self.other_strikes += 1
        self.confidence *= self.config.other_identity_factor
        self._refresh()

    def penalize(self, factor):
        self.confidence *= factor
        self._refresh()

    def reset(self):
        self.confidence = 0.0
        self.last_evidence_t = None
        self.other_strikes = 0
        self._refresh()

    def _refresh(self):
        cfg = self.config
        c = self.confidence
        if c >= cfg.confirm_threshold:
            self.state = IdentityState.CONFIRMED
        elif self.state == IdentityState.CONFIRMED and c >= cfg.drop_threshold:
            pass                                    # hysteresis: stay confirmed
        elif self.state in (IdentityState.CONFIRMED, IdentityState.TEMPORARY) \
                and c >= cfg.unknown_threshold:
            self.state = IdentityState.TEMPORARY
        else:
            self.state = IdentityState.UNKNOWN
