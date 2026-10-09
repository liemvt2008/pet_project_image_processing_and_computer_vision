"""Giao diện Gradio Web App."""
import gradio as gr

from .config import AppConfig
from .predictor import AgePredictor


class GradioApp:
    """Xây dựng giao diện và gắn sự kiện với AgePredictor."""

    def __init__(self, predictor: AgePredictor, config: AppConfig):
        self.predictor = predictor
        self.config = config
        self.theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")
        self.demo = self._build()

    def _on_submit(self, input_image):
        if input_image is None:
            return None, None, 0.0, {}
        result = self.predictor.predict(input_image)
        return result.annotated_image, result.face_crop, result.estimated_age, result.group_probs

    def _build(self) -> gr.Blocks:
        size = self.config.model.image_size
        n_groups = len(self.config.age_groups)

        with gr.Blocks(title=self.config.title) as demo:
            gr.Markdown(
                """
                # 👤 Demo Dự đoán Độ tuổi từ Ảnh Khuôn mặt (Facial Age Estimation)
                ### Mô hình: **MobileNetV3-Large + SoftArgmax Output Head** | Bộ dữ liệu: **UTKFace**
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(
                        sources=["webcam", "upload"],
                        type="numpy",
                        label="📸 Đầu vào: Chụp Webcam hoặc Tải ảnh từ máy tính",
                    )
                    submit_btn = gr.Button("🔍 Dự đoán Độ tuổi", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output_annotated = gr.Image(label="🖼️ Kết quả phát hiện khuôn mặt & Tuổi dự đoán")
                    with gr.Row():
                        output_crop = gr.Image(label=f"✂️ Khuôn mặt đã Crop ({size}x{size})")
                        output_age = gr.Number(label="🎯 Tuổi dự đoán (Expected Age)", precision=1)
                    output_group_probs = gr.Label(
                        label=f"📊 Phân bố xác suất {n_groups} Nhóm tuổi", num_top_classes=5
                    )

            submit_btn.click(
                fn=self._on_submit,
                inputs=[input_img],
                outputs=[output_annotated, output_crop, output_age, output_group_probs],
            )
        return demo

    def launch(self) -> None:
        server = self.config.server
        self.demo.launch(
            server_name=server.host,
            server_port=server.port,
            share=server.share,
            theme=self.theme,
        )
