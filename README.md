# Facial Age Estimation — MobileNetV3 + SoftArgmax (UTKFace)

Demo web app that estimates a person's age from a face photo (webcam or upload).

**Pipeline:** MTCNN face detection → crop (+10% margin) → resize 224×224 →
MobileNetV3-Large backbone → 101-way softmax → **SoftArgmax** expected age
`E[y] = Σ k·P(k)` → probabilities grouped into 10 age ranges.

## Project structure

```
pet_project_image_processing_and_computer_vision/
├── main.py                  # Entry point (CLI args, wiring of components)
├── requirements.txt
├── models/
│   └── best_mobilenet_softmax_lr_plateau.h5   # Trained weights
└── age_estimation/
    ├── config.py            # AppConfig / ModelConfig / DetectorConfig / ServerConfig
    ├── layers.py            # SoftArgmaxLayer (custom Keras layer)
    ├── model.py             # AgeEstimationModel: build, load weights, predict
    ├── face_detector.py     # FaceDetector (MTCNN wrapper) + FaceBox
    ├── visualizer.py        # ImageAnnotator: draw bounding box & labels
    ├── predictor.py         # AgePredictor: end-to-end pipeline → PredictionResult
    └── ui.py                # GradioApp: web interface
```

## Setup

Requires **Python 3.10 – 3.12**.

```bash
git clone <your-repo-url>
cd pet_project_image_processing_and_computer_vision

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Then open http://127.0.0.1:7860 in your browser.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--weights PATH` | `models/best_mobilenet_softmax_lr_plateau.h5` | Weights file |
| `--host HOST` | `127.0.0.1` | Use `0.0.0.0` to allow access from other machines on the LAN |
| `--port PORT` | `7860` | Server port |
| `--share` | off | Create a temporary public Gradio link |

```bash
python main.py --host 0.0.0.0 --port 8080
python main.py --share
```

## Use as a library

```python
import cv2
from age_estimation import (AppConfig, AgeEstimationModel, AgePredictor,
                            FaceDetector, ImageAnnotator)

cfg = AppConfig()
predictor = AgePredictor(FaceDetector(cfg.detector), AgeEstimationModel(cfg.model),
                         ImageAnnotator(), cfg.age_groups)

img = cv2.cvtColor(cv2.imread("face.jpg"), cv2.COLOR_BGR2RGB)
result = predictor.predict(img)
print(result.estimated_age, result.group_probs)
```
