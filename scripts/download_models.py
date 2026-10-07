import requests
from pathlib import Path

BASE = "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master"
ROOT = Path(__file__).resolve().parent.parent

FILES = {
    f"{BASE}/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth":
        ROOT / "models/antispoof/2.7_80x80_MiniFASNetV2.pth",
    f"{BASE}/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth":
        ROOT / "models/antispoof/4_0_0_80x80_MiniFASNetV1SE.pth",
    f"{BASE}/src/model_lib/MiniFASNet.py":
        ROOT / "src/minifasnet_arch.py",
}

for url, dest in FILES.items():
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("skip", dest.name)
        continue
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print("downloaded", dest.name)