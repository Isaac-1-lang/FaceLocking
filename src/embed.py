"""ArcFace ONNX inference with explicit, model-dependent preprocessing."""
import hashlib
import cv2
import numpy as np
import onnxruntime as ort
from .config import EMBEDDER, require_model


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a finite one-dimensional vector")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Embedding has zero or invalid norm")
    return vector / norm


class ArcFace:
    def __init__(self, model=EMBEDDER, preprocessing="raw"):
        path = require_model(model)
        if preprocessing not in ("raw", "normalized"):
            raise ValueError("Preprocessing must be raw or normalized")
        with path.open("rb") as stream:
            self.fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        self.preprocessing = preprocessing
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("Expected one float32 ArcFace input")
        self.input = inputs[0]
        shape = self.input.shape
        if len(shape) != 4 or any(isinstance(v, int) and v != expected
                                  for v, expected in zip(shape, (1, 3, 112, 112))):
            raise ValueError(f"Expected NCHW [1,3,112,112] ArcFace input, got {shape}")

    @property
    def signature(self):
        return {"sha256": self.fingerprint, "preprocessing": self.preprocessing,
                "alignment": "arcface-112-five-point-v1"}

    def embed(self, aligned):
        if aligned.shape != (112, 112, 3):
            raise ValueError("ArcFace requires an aligned 112x112 BGR image")
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Model Zoo ResNet100 includes its input normalization in the graph.
        if self.preprocessing == "normalized":
            rgb = (rgb - 127.5) / 127.5
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        output = self.session.run(None, {self.input.name: tensor})[0]
        vector = normalize(output.reshape(-1))
        if vector.size != 512:
            raise ValueError(f"Expected 512 ArcFace features, got {vector.size}")
        return vector


def main():
    from .align import align_face
    from .workflow import camera, exiting, models, parser, read_frame
    args = parser("Preview ArcFace embedding shape and L2 norm").parse_args()
    detector, embedder = models(args)
    window = "FaceX embedding"
    with camera(args.camera) as cap:
        while True:
            frame = read_frame(cap)
            faces = detector.detect(frame)
            message = "Show exactly one face"
            if len(faces) == 1:
                aligned = align_face(frame, faces[0].points)
                vector = embedder.embed(aligned)
                message = f"Embedding: {vector.size} | L2 norm: {np.linalg.norm(vector):.4f}"
                cv2.imshow("Aligned face", aligned)
            cv2.putText(frame, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window, frame)
            if exiting(window, cv2.waitKey(1) & 0xff):
                break


if __name__ == "__main__":
    from .workflow import run
    run(main)
