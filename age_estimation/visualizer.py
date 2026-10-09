"""Vẽ kết quả (bounding box, nhãn) lên ảnh."""
import cv2
import numpy as np

from .face_detector import FaceBox

GREEN = (0, 255, 0)
RED = (255, 0, 0)


class ImageAnnotator:
    """Các thao tác vẽ lên ảnh RGB. Luôn trả về bản sao, không sửa ảnh gốc.

    Lưu ý: cv2.putText chỉ hỗ trợ ký tự ASCII nên nhãn trên ảnh viết không dấu.
    """

    @staticmethod
    def draw_face(image_rgb: np.ndarray, box: FaceBox, age: float) -> np.ndarray:
        annotated = image_rgb.copy()
        cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), GREEN, 3)
        cv2.putText(annotated, f"Tuoi: {age:.1f}", (box.x1, max(30, box.y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2, cv2.LINE_AA)
        return annotated

    @staticmethod
    def draw_no_face(image_rgb: np.ndarray) -> np.ndarray:
        annotated = image_rgb.copy()
        cv2.putText(annotated, "Khong tim thay mat (Dung anh goc)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA)
        return annotated
