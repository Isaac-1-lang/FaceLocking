"""Enroll one consenting person from a webcam; SPACE captures a sample."""
import time
import cv2
from .align import align_face
from .database import FaceDatabase
from .landmarks import draw_face
from .workflow import camera, exiting, models, parser, read_frame, run


def main():
    cli = parser(__doc__)
    cli.add_argument("--name", required=True)
    cli.add_argument("--samples", type=int, default=8)
    cli.add_argument("--replace", action="store_true")
    cli.add_argument("--min-face-size", type=int, default=90)
    cli.add_argument("--min-sharpness", type=float, default=40.0)
    args = cli.parse_args()
    if args.samples < 3 or args.min_face_size < 1 or not 0 <= args.min_sharpness < float("inf"):
        cli.error("Use at least 3 samples, positive face size, and finite nonnegative sharpness")
    name = args.name.strip()
    FaceDatabase.validate_name(name)
    detector, embedder = models(args)
    db = FaceDatabase(args.db, embedder.signature)
    if name in db.people and not args.replace:
        cli.error("Name already enrolled. Use --replace to replace it.")
    samples, last_capture = [], -float("inf")
    window = "FaceX enrollment"
    message = "SPACE: capture | Q: cancel"
    print("Look at the camera. Vary expression and head angle slightly between captures.")
    with camera(args.camera) as cap:
        while len(samples) < args.samples:
            frame = read_frame(cap)
            faces = detector.detect(frame)
            aligned = None
            quality = "Show exactly one face"
            if len(faces) == 1:
                face = faces[0]
                x, y, w, h = face.box
                if min(w, h) < args.min_face_size:
                    quality = "Move closer"
                elif x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                    quality = "Keep your whole face in frame"
                else:
                    aligned = align_face(frame, face.points)
                    sharpness = cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    quality = "Ready" if sharpness >= args.min_sharpness else "Too blurry: hold still / improve light"
                    if quality != "Ready":
                        aligned = None
            for face in faces:
                draw_face(frame, face)
            cv2.putText(frame, f"{len(samples)}/{args.samples} | {quality}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(frame, message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(window, frame)
            if aligned is not None:
                cv2.imshow("Aligned face", aligned)
            key = cv2.waitKey(1) & 0xff
            if exiting(window, key):
                print("Enrollment cancelled; database unchanged.")
                return
            if key == 32 and aligned is not None and time.monotonic() - last_capture >= 0.5:
                samples.append(embedder.embed(aligned))
                last_capture = time.monotonic()
                print(f"Captured {len(samples)}/{args.samples}")
    db.enroll(name, samples, replace=args.replace)
    print(f"Enrolled {name!r} with {len(samples)} samples in {db.path}")


if __name__ == "__main__":
    run(main)
