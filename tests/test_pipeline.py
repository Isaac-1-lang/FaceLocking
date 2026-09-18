import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.align import TEMPLATE, align_face, similarity_transform
from src.database import FaceDatabase
from src.embed import normalize


class AlignmentTests(unittest.TestCase):
    def test_recovers_rotation_scale_translation(self):
        angle = 0.3
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        source = TEMPLATE @ rotation.T * 1.8 + [40, 20]
        matrix = similarity_transform(source)
        actual = np.c_[source, np.ones(5)] @ matrix.T
        np.testing.assert_allclose(actual, TEMPLATE, atol=1e-4)

    def test_crop_and_invalid_points(self):
        frame = np.zeros((200, 200, 3), np.uint8)
        self.assertEqual(align_face(frame, TEMPLATE).shape, (112, 112, 3))
        for points in (np.zeros((5, 2)), np.zeros((4, 2)), np.full((5, 2), np.nan)):
            with self.assertRaises(ValueError):
                similarity_transform(points)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "faces.json"
        self.signature = {"sha256": "test", "preprocessing": "raw"}
        self.a, self.b = np.eye(512, dtype=np.float32)[:2]

    def test_roundtrip_unknown_and_replacement(self):
        db = FaceDatabase(self.path, self.signature)
        self.assertEqual(db.match(self.a)[0], "Unknown")
        db.enroll("Alice", [self.a]*3)
        db = FaceDatabase(self.path, self.signature)
        self.assertEqual(db.match(self.a)[0], "Alice")
        self.assertEqual(db.match(self.b)[0], "Unknown")
        with self.assertRaises(ValueError):
            db.enroll("Alice", [self.b])
        db.enroll("Alice", [self.b], replace=True)
        self.assertEqual(db.match(self.b)[0], "Alice")
        with self.assertRaises(ValueError):
            FaceDatabase(self.path, {"sha256": "other"})

    def test_ambiguous_and_malformed(self):
        db = FaceDatabase(self.path, self.signature)
        db.enroll("Alice", [self.a])
        db.enroll("Bob", [self.a])
        self.assertEqual(db.match(self.a)[0], "Unknown")
        self.path.write_text("{bad", encoding="utf-8")
        with self.assertRaises(ValueError):
            FaceDatabase(self.path, self.signature)

    def test_invalid_embeddings(self):
        for vector in (np.zeros(512), np.full(512, np.nan), np.ones((1, 512))):
            with self.assertRaises(ValueError):
                normalize(vector)
        with self.assertRaises(ValueError):
            FaceDatabase.validate_vector(np.ones(12))


if __name__ == "__main__":
    unittest.main()
