"""Pipeline dự đoán: phát hiện mặt -> crop/resize -> dự đoán tuổi -> vẽ kết quả."""
from dataclasses import dataclass
from typing import Dict, Sequence

import cv2
import numpy as np

from .face_detector import FaceDetector
from .model import AgeEstimationModel
from .visualizer import ImageAnnotator


@dataclass
class PredictionResult:
    annotated_image: np.ndarray
    face_crop: np.ndarray
    estimated_age: float
    group_probs: Dict[str, float]
    face_found: bool


class AgePredictor:
    """Điều phối FaceDetector, AgeEstimationModel và ImageAnnotator."""

    def __init__(
        self,
        detector: FaceDetector,
        model: AgeEstimationModel,
        annotator: ImageAnnotator,
        age_groups: Sequence[str],
    ):
        self.detector = detector
        self.model = model
        self.annotator = annotator
        self.age_groups = list(age_groups)

    def predict(self, image) -> PredictionResult:
        img_rgb = np.array(image)

        # 1. Phát hiện khuôn mặt (không có mặt -> dùng toàn bộ ảnh)
        box = self.detector.detect_best_face(img_rgb)
        face_crop = box.crop(img_rgb) if box else img_rgb

        # 2. Resize khuôn mặt về kích thước đầu vào của mô hình
        size = self.model.config.image_size
        face_resized = cv2.resize(face_crop, (size, size))

        # 3. Dự đoán độ tuổi
        estimated_age, probs = self.model.predict(face_resized)

        # 4. Vẽ kết quả lên ảnh
        if box:
            annotated = self.annotator.draw_face(img_rgb, box, estimated_age)
        else:
            annotated = self.annotator.draw_no_face(img_rgb)

        return PredictionResult(
            annotated_image=annotated,
            face_crop=face_resized,
            estimated_age=round(estimated_age, 1),
            group_probs=self._group_probabilities(probs),
            face_found=box is not None,
        )

    def _group_probabilities(self, probs: np.ndarray) -> Dict[str, float]:
        """Gom các mốc tuổi thành các nhóm 10 năm; nhóm cuối gồm toàn bộ phần còn lại."""
        last = len(self.age_groups) - 1
        group_probs = {}
        for i, name in enumerate(self.age_groups):
            end = len(probs) if i == last else (i + 1) * 10
            group_probs[name] = round(float(np.sum(probs[i * 10:end])), 4)
        return group_probs
