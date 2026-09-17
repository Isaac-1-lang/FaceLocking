import argparse

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from src.align import align_face_5pt
from src.face_signals import FaceSignalExtractor

from src.recognize import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    HaarFaceMesh5pt,
    load_db_npz,
)