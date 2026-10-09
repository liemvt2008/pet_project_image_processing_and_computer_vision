"""Mô hình ước lượng tuổi: MobileNetV3-Large + SoftArgmax Head."""
from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import ModelConfig
from .layers import SoftArgmaxLayer


class AgeEstimationModel:
    """Đóng gói việc xây dựng, nạp trọng số và dự đoán của mô hình."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.weights_loaded = False
        self._model = self._build()
        self._load_weights()

    def _build(self) -> tf.keras.Model:
        cfg = self.config
        shape = (cfg.image_size, cfg.image_size, 3)
        has_weights = cfg.weights_path.exists()

        inputs = tf.keras.Input(shape=shape, name="input_image")

        # Preprocessing chuẩn của MobileNetV3 (Scale về [-1, 1])
        x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)

        # Backbone MobileNetV3-Large (đóng băng)
        backbone = tf.keras.applications.MobileNetV3Large(
            input_shape=shape,
            include_top=False,
            weights=None if has_weights else "imagenet",
            pooling="avg",
        )
        backbone.trainable = False

        x = backbone(x, training=False)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(cfg.dropout_rates[0])(x)
        x = tf.keras.layers.Dense(cfg.dense_units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(cfg.dropout_rates[1])(x)

        # Output Head
        age_probs = tf.keras.layers.Dense(cfg.max_age, activation="softmax", name="age_probs")(x)
        expected_age = SoftArgmaxLayer(max_age=cfg.max_age, name="expected_age")(age_probs)

        return tf.keras.Model(
            inputs=inputs,
            outputs=[expected_age, age_probs],
            name="MobileNetV3_SoftArgmax",
        )

    def _load_weights(self) -> None:
        path = self.config.weights_path
        if not path.exists():
            print(f"⚠️ Không tìm thấy '{path}'. Mô hình sẽ chạy với trọng số ngẫu nhiên.")
            return
        try:
            self._model.load_weights(str(path))
            self.weights_loaded = True
            print(f"✅ Đã nạp thành công file trọng số: {path}")
        except Exception as e:
            print(f"⚠️ Không thể nạp weights từ {path}: {e}")

    def predict(self, face_rgb: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Dự đoán tuổi từ ảnh khuôn mặt RGB đã resize về (image_size, image_size).
        Trả về (tuổi kỳ vọng, vector xác suất max_age mốc).
        """
        batch = np.expand_dims(face_rgb, axis=0).astype(np.float32)
        pred_age, pred_probs = self._model.predict(batch, verbose=0)
        return float(pred_age[0][0]), pred_probs[0]
