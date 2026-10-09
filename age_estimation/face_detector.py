"""Phát hiện khuôn mặt bằng MTCNN."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from mtcnn import MTCNN

from .config import DetectorConfig


@dataclass(frozen=True)
class FaceBox:
    """Bounding box (x1, y1, x2, y2) đã mở rộng margin và cắt theo biên ảnh."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y1:self.y2, self.x1:self.x2]


class FaceDetector:
    """Wrapper quanh MTCNN, trả về khuôn mặt có độ tin cậy cao nhất."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._detector = MTCNN()

    def detect_best_face(self, image_rgb: np.ndarray) -> Optional[FaceBox]:
        faces = self._detector.detect_faces(image_rgb)
        if not faces:
            return None

        best_face = max(faces, key=lambda f: f["confidence"])
        x, y, w, h = best_face["box"]
        h_img, w_img = image_rgb.shape[:2]

        # Mở rộng Bounding Box cho tự nhiên
        margin = int(self.config.box_margin_ratio * max(w, h))
        return FaceBox(
            x1=max(0, x - margin),
            y1=max(0, y - margin),
            x2=min(w_img, x + w + margin),
            y2=min(h_img, y + h + margin),
            confidence=float(best_face["confidence"]),
        )
