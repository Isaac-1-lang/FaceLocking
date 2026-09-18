"""Least-squares similarity alignment using all five detected landmarks."""
import cv2
import numpy as np

TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                     [56.0252, 71.7366], [41.5493, 92.3655],
                     [70.7299, 92.2041]], dtype=np.float32)


def similarity_transform(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Expected five finite (x, y) landmark coordinates")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("Degenerate facial landmarks")
    # x' = a*x - b*y + tx; y' = b*x + a*y + ty.
    design = np.zeros((10, 4))
    design[0::2, 0] = points[:, 0]
    design[0::2, 1] = -points[:, 1]
    design[0::2, 2] = 1
    design[1::2, 0] = points[:, 1]
    design[1::2, 1] = points[:, 0]
    design[1::2, 3] = 1
    a, b, tx, ty = np.linalg.lstsq(design, TEMPLATE.ravel(), rcond=None)[0]
    if a*a + b*b < 1e-12:
        raise ValueError("Invalid alignment scale")
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float32)


def align_face(frame, points):
    return cv2.warpAffine(frame, similarity_transform(points), (112, 112),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def main():
    from .landmarks import FaceDetector, draw_face
    from .workflow import camera, exiting, parser, read_frame

    args = parser("Preview 5-point face alignment").parse_args()
    detector = FaceDetector(args.detector)
    window = "Camera"

    with camera(args.camera) as cap:
        while True:
            frame = read_frame(cap)
            faces = detector.detect(frame)

            for face in faces:
                draw_face(frame, face)

            if len(faces) == 1:
                aligned = align_face(frame, faces[0].points)
                cv2.imshow("Aligned Face 112x112", aligned)

            cv2.imshow(window, frame)

            if exiting(window, cv2.waitKey(1) & 0xff):
                break


if __name__ == "__main__":
    from .workflow import run
    run(main)
