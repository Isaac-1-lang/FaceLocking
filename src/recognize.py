"""Recognize enrolled faces from a webcam with unknown rejection."""
import time
import cv2
from .align import align_face
from .database import FaceDatabase
from .landmarks import draw_face
from .workflow import camera, exiting, models, parser, read_frame, run


def main():
    cli = parser(__doc__)
    cli.add_argument("--threshold", type=float, default=0.45)
    cli.add_argument("--margin", type=float, default=0.05)
    args = cli.parse_args()
    if not -1 <= args.threshold <= 1 or not 0 <= args.margin <= 2:
        cli.error("Threshold must be [-1,1] and margin [0,2]")
    detector, embedder = models(args)
    db = FaceDatabase(args.db, embedder.signature)
    if not db.people:
        cli.error('No faces enrolled. Run python -m src.enroll --name "Your Name" first.')
    window = "FaceX recognition"
    print("Recognition running. Q or Escape quits. Scores are cosine similarity, not probabilities.")
    with camera(args.camera) as cap:
        while True:
            start = time.perf_counter()
            frame = read_frame(cap)
            results = []
            for face in detector.detect(frame):
                if min(face.box[2:4]) < 70:
                    results.append((face, "Too small", 0.0))
                    continue
                try:
                    vector = embedder.embed(align_face(frame, face.points))
                    name, score = db.match(vector, args.threshold, args.margin)
                except ValueError:
                    name, score = "Unknown", 0.0
                results.append((face, name, score))
            # Draw only after inference so overlays never enter another face crop.
            for face, name, score in results:
                color = (0, 200, 0) if name not in ("Unknown", "Too small") else (0, 100, 255)
                draw_face(frame, face, f"{name} {score:.2f}", color)
            fps = 1 / max(time.perf_counter() - start, 1e-6)
            cv2.putText(frame, f"FaceX | {fps:.1f} FPS | Q: quit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window, frame)
            if exiting(window, cv2.waitKey(1) & 0xff):
                break


if __name__ == "__main__":
    run(main)
