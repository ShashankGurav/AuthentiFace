import numpy as np
from insightface.app import FaceAnalysis
import config


class FaceEngine:
    """InsightFace: detection + 5-pt alignment + 512-d ArcFace embedding."""

    def __init__(self, use_gpu: bool = False):
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_gpu else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name="buffalo_l", providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=config.DET_SIZE)

    def detect(self, img_bgr: np.ndarray):
        """Returns list of insightface Face objects (bbox, kps, normed_embedding)."""
        return self.app.get(img_bgr)

    def largest_face(self, img_bgr: np.ndarray):
        faces = self.detect(img_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))