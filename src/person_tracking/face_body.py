"""Spatial face-to-body mapping.

A face belongs to the body whose *head region* it falls in. The head is
expected near the horizontal centre of the box and in its top part. The score
is a smooth Gaussian prior (so partial/truncated detector boxes still work)
combined with a plausibility check on face-to-body size.
"""
import numpy as np

from .assignment import linear_assignment
from .records import as_box, box_center

HEAD_FRACTION = 0.085      # expected face-centre height below the body top, as a body-height fraction
FACE_TO_BODY_HEIGHT = 0.135  # typical face-box height / body height for a standing person
BODY_ASPECT = 0.38         # typical body width / height


def head_anchor(body_box, fraction=HEAD_FRACTION):
    """Expected head centre of a body box (x centre, near the top)."""
    body_box = as_box(body_box)
    return np.array([(body_box[0] + body_box[2]) / 2.0,
                     body_box[1] + fraction * (body_box[3] - body_box[1])])


def body_from_face(face_box):
    """Pseudo whole-body box for a face when the person detector gave nothing.

    The box may extend beyond the image: that is deliberate, it describes the
    body as it would be, and the filter treats it as a noisy measurement.
    """
    face_box = as_box(face_box)
    cx, cy = box_center(face_box)
    fh = max(face_box[3] - face_box[1], 1.0)
    height = fh / FACE_TO_BODY_HEIGHT
    top = face_box[1] - 0.15 * fh
    half_w = BODY_ASPECT * height / 2.0
    return np.array([cx - half_w, top, cx + half_w, top + height])


def face_body_score(face_box, body_box):
    """Likelihood-like score in [0, 1] that ``face_box`` is the head of ``body_box``."""
    face_box, body_box = as_box(face_box), as_box(body_box)
    bw = max(body_box[2] - body_box[0], 1.0)
    bh = max(body_box[3] - body_box[1], 1.0)
    fx, fy = box_center(face_box)
    fh = max(face_box[3] - face_box[1], 1.0)
    # Hard rejects: the face centre must be inside the (slightly grown) box's upper half.
    if not (body_box[0] - 0.1 * bw <= fx <= body_box[2] + 0.1 * bw):
        return 0.0
    if not (body_box[1] - 0.1 * bh <= fy <= body_box[1] + 0.5 * bh):
        return 0.0
    ax, ay = head_anchor(body_box)
    # Sigmas: loose horizontally (turned shoulders), tight vertically.
    gauss = np.exp(-0.5 * (((fx - ax) / (0.20 * bw)) ** 2
                           + ((fy - ay) / (0.08 * bh)) ** 2))
    ratio = fh / bh
    lo, hi = 0.05, 0.30
    size_term = 1.0 if lo <= ratio <= hi else float(np.exp(-abs(np.log(ratio / (
        lo if ratio < lo else hi))) * 2.0))
    return float(gauss * size_term)


def associate_faces(faces, bodies, min_score=0.12, ambiguity_ratio=0.6):
    """Globally optimal one-to-one face->body assignment.

    Returns ``(pairs, ambiguous)``: ``pairs`` maps face index -> body index;
    ``ambiguous`` is the set of face indices whose runner-up body scored within
    ``ambiguity_ratio`` of the winner (e.g. two overlapping people), which the
    caller must not use as identity evidence.
    """
    if not faces or not bodies:
        return {}, set()
    scores = np.array([[face_body_score(f, b) for b in bodies] for f in faces])
    pairs = {}
    for fi, bi in linear_assignment(1.0 - scores):
        if scores[fi, bi] >= min_score:
            pairs[fi] = bi
    ambiguous = set()
    for fi, bi in pairs.items():
        others = np.delete(scores[fi], bi)
        if others.size and others.max() >= ambiguity_ratio * scores[fi, bi]:
            ambiguous.add(fi)
    return pairs, ambiguous
