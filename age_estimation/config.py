"""Cấu hình tập trung cho toàn bộ ứng dụng."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_AGE_GROUPS: Tuple[str, ...] = (
    "0-9 tuổi", "10-19 tuổi", "20-29 tuổi", "30-39 tuổi", "40-49 tuổi",
    "50-59 tuổi", "60-69 tuổi", "70-79 tuổi", "80-89 tuổi", "90+ tuổi",
)


@dataclass(frozen=True)
class ModelConfig:
    weights_path: Path = PROJECT_ROOT / "models" / "best_mobilenet_softmax_lr_plateau.h5"
    image_size: int = 224
    max_age: int = 101
    dense_units: int = 128
    dropout_rates: Tuple[float, float] = (0.3, 0.2)


@dataclass(frozen=True)
class DetectorConfig:
    # Tỉ lệ mở rộng bounding box (10%) cho khuôn mặt tự nhiên hơn
    box_margin_ratio: float = 0.1


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    age_groups: Tuple[str, ...] = DEFAULT_AGE_GROUPS
    title: str = "Demo Dự đoán Độ tuổi - MobileNetV3 UTKFace"
