from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "db" / "embeddings.json"
MODEL_DIR = ROOT / "models" / "antispoof"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_SPOOF_MODEL = "old"      # "old" or "new"

# --- old: MiniFASNetV2 (scale 2.7) + V1SE (scale 4.0), 80x80, PyTorch ---
OLD_MODELS = [
    ("2.7_80x80_MiniFASNetV2.pth", 2.7, "MiniFASNetV2"),
    ("4_0_0_80x80_MiniFASNetV1SE.pth", 4.0, "MiniFASNetV1SE"),
]

# --- new: MiniFASNetV2-SE INT8, 128x128, ONNX (facenox/face-antispoof-onnx) ---
NEW_MODEL = MODEL_DIR / "best_model_quantized.onnx"
NEW_PAD = 1.5
NEW_SCALE_255 = True
NEW_REAL_INDEX = 0

LIVE_THRESHOLD = {"old": 0.50, "new": 0.50}   # P(real), per model

DET_SIZE = (640, 640)
MATCH_THRESHOLD = 0.40
CAMERA_INDEX = 1