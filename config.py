from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "db" / "embeddings.json"
MODEL_DIR = ROOT / "models" / "antispoof"
OUTPUT_DIR = ROOT / "outputs"

# (weights file, crop scale, architecture) -- scale comes from the file name
SPOOF_MODELS = [
    ("2.7_80x80_MiniFASNetV2.pth", 2.7, "MiniFASNetV2"),
    ("4_0_0_80x80_MiniFASNetV1SE.pth", 4.0, "MiniFASNetV1SE"),
]

DET_SIZE = (640, 640)
MATCH_THRESHOLD = 0.40   # cosine similarity; tune on your own data
LIVE_THRESHOLD = 0.50    # mean "real" probability; tune on your own data