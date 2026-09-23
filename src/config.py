"""Project-relative defaults, independent of the current working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / "models" / "detector_yunet.onnx"
EMBEDDER = ROOT / "models" / "embedder_arcface.onnx"
DATABASE = ROOT / "data" / "db" / "face_db.json"
LOGS = ROOT / "data" / "logs"


def require_model(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 1024:
        raise ValueError(f"Missing or empty model: {path}. Run python init_project.py --download-models")
    return path
