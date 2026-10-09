"""Compare both anti-spoof models on a folder of images.
Usage: python check_antispoof.py data/spoof_phone"""
import sys
import cv2
from pathlib import Path
from src.face_engine import FaceEngine
from src.antispoof import load_antispoof

engine = FaceEngine()
models = {m: load_antispoof(m) for m in ("old", "new")}
print(f"{'file':32} {'old P(real)':>12} {'new P(real)':>12}")
for f in sorted(Path(sys.argv[1]).iterdir()):
    img = cv2.imread(str(f))
    face = engine.largest_face(img) if img is not None else None
    if face is None:
        continue
    bbox = tuple(int(v) for v in face.bbox)
    s = {m: models[m].predict(img, bbox)[1] for m in models}
    print(f"{f.name:32} {s['old']:12.3f} {s['new']:12.3f}")