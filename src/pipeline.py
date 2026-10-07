from dataclasses import dataclass
from src.face_engine import FaceEngine
from src.antispoof import AntiSpoof
from src.matcher import Matcher
from src.database import load_gallery


@dataclass
class FaceResult:
    bbox: tuple
    live: bool
    live_score: float
    name: str | None   # None if spoof (recognition skipped)
    sim: float | None


class Pipeline:
    def __init__(self, use_gpu: bool = False):
        self.engine = FaceEngine(use_gpu)
        self.spoof = AntiSpoof("cuda" if use_gpu else "cpu")
        self.matcher = Matcher(*load_gallery())

    def process(self, img_bgr) -> list[FaceResult]:
        results = []
        for face in self.engine.detect(img_bgr):                 # 1. detect
            bbox = tuple(int(v) for v in face.bbox)
            live, live_score = self.spoof.predict(img_bgr, bbox)  # 2. spoof check
            if not live:                                          # spoof never reaches matching
                results.append(FaceResult(bbox, False, live_score, None, None))
                continue
            name, sim = self.matcher.match(face.normed_embedding)  # 3. recognize
            results.append(FaceResult(bbox, True, live_score, name, sim))
        return results