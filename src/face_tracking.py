import argparse

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from src.align import align_face
from src.database import FaceDatabase
from src.embed import ArcFace
from src.landmarks import FaceDetector
