"""Evaluate held-out images against the enrolled gallery."""
import csv
from pathlib import Path
import cv2
from .align import align_face
from .database import FaceDatabase
from .workflow import models, parser, run


def main():
    cli = parser(__doc__)
    cli.add_argument("--manifest", required=True, help="CSV columns: path,label; Unknown for unenrolled people")
    cli.add_argument("--threshold", type=float, default=0.45)
    cli.add_argument("--margin", type=float, default=0.05)
    args = cli.parse_args()
    if not -1 <= args.threshold <= 1 or not 0 <= args.margin <= 2:
        cli.error("Threshold must be [-1,1] and margin [0,2]")
    detector, embedder = models(args)
    db = FaceDatabase(args.db, embedder.signature)
    if not db.people:
        cli.error("Enroll faces before evaluation")
    manifest = Path(args.manifest).resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label"}.issubset(reader.fieldnames or []):
            cli.error("Manifest must have path,label columns")
        rows = list(reader)
    if not rows:
        cli.error("Manifest is empty")
    correct = failures = false_accepts = false_rejects = misidentified = known = unknown = 0
    for row in rows:
        if not row.get("path") or not row.get("label"):
            cli.error("Every row needs a path and label")
        label = row["label"]
        if label != "Unknown" and label not in db.people:
            cli.error(f"Label {label!r} is not enrolled; use Unknown for unenrolled people")
        known += label != "Unknown"
        unknown += label == "Unknown"
        path = manifest.parent / row["path"]
        frame = cv2.imread(str(path))
        faces = [] if frame is None else detector.detect(frame)
        if len(faces) != 1:
            failures += 1
            print(f"FAIL {path.name}: expected one face, found {len(faces)}")
            continue
        prediction, score = db.match(embedder.embed(align_face(frame, faces[0].points)),
                                     args.threshold, args.margin)
        correct += prediction == label
        false_accepts += label == "Unknown" and prediction != "Unknown"
        false_rejects += label != "Unknown" and prediction == "Unknown"
        misidentified += label != "Unknown" and prediction not in (label, "Unknown")
        print(f"{path.name}: expected={label}, predicted={prediction}, cosine={score:.3f}")
    processed = len(rows) - failures
    print(f"Total={len(rows)} processed={processed} detection/read failures={failures}")
    print(f"Correct/total={correct}/{len(rows)} ({correct/len(rows):.1%})")
    print(f"Unknown false accepts={false_accepts}; known false rejects={false_rejects}; wrong identities={misidentified}")
    print(f"Known inputs={known}; unknown inputs={unknown}. Failures are excluded from error counts above.")


if __name__ == "__main__":
    run(main)
