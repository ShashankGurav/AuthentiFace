import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master"
NEW = "https://raw.githubusercontent.com/facenox/face-antispoof-onnx/main"

FILES = {
    f"{OLD}/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth":
        ROOT / "models/antispoof/2.7_80x80_MiniFASNetV2.pth",
    f"{OLD}/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth":
        ROOT / "models/antispoof/4_0_0_80x80_MiniFASNetV1SE.pth",
    f"{OLD}/src/model_lib/MiniFASNet.py":
        ROOT / "src/minifasnet_arch.py",
    f"{NEW}/models/best_model_quantized.onnx":
        ROOT / "models/antispoof/best_model_quantized.onnx",
}

for url, dest in FILES.items():
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("skip", dest.name)
        continue
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print("downloaded", dest.name)