"""Entry point: khởi tạo các thành phần và chạy Gradio Web App.

Ví dụ:
    python main.py
    python main.py --port 8080 --share
    python main.py --weights path/to/weights.h5
"""
import argparse
from pathlib import Path

from age_estimation import (
    AgeEstimationModel, AgePredictor, AppConfig, DetectorConfig,
    FaceDetector, ImageAnnotator, ModelConfig, ServerConfig,
)
from age_estimation.ui import GradioApp


def parse_args() -> argparse.Namespace:
    defaults = AppConfig()
    parser = argparse.ArgumentParser(description="Demo Dự đoán Độ tuổi - MobileNetV3 UTKFace")
    parser.add_argument("--weights", type=Path, default=defaults.model.weights_path,
                        help="Đường dẫn file trọng số .h5")
    parser.add_argument("--host", default=defaults.server.host, help="Địa chỉ server (0.0.0.0 để mở trong LAN)")
    parser.add_argument("--port", type=int, default=defaults.server.port, help="Cổng server")
    parser.add_argument("--share", action="store_true", help="Tạo public link Gradio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig(
        model=ModelConfig(weights_path=args.weights),
        detector=DetectorConfig(),
        server=ServerConfig(host=args.host, port=args.port, share=args.share),
    )

    print("⌛ Đang khởi tạo MTCNN Face Detector & MobileNetV3 Model...")
    predictor = AgePredictor(
        detector=FaceDetector(config.detector),
        model=AgeEstimationModel(config.model),
        annotator=ImageAnnotator(),
        age_groups=config.age_groups,
    )
    print("🚀 Khởi tạo hoàn tất!")

    GradioApp(predictor, config).launch()


if __name__ == "__main__":
    main()
