import cv2
import numpy as np
import torch
import config
from src import minifasnet_arch as arch


def _crop(img, bbox_xywh, scale, out_size=80):
    """Same crop logic as Silent-Face-Anti-Spoofing (CropImage)."""
    src_h, src_w = img.shape[:2]
    x, y, bw, bh = bbox_xywh
    scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
    nw, nh = bw * scale, bh * scale
    cx, cy = bw / 2 + x, bh / 2 + y
    l, t = cx - nw / 2, cy - nh / 2
    r, b = cx + nw / 2, cy + nh / 2
    if l < 0: r -= l; l = 0
    if t < 0: b -= t; t = 0
    if r > src_w - 1: l -= r - src_w + 1; r = src_w - 1
    if b > src_h - 1: t -= b - src_h + 1; b = src_h - 1
    crop = img[int(t):int(b) + 1, int(l):int(r) + 1]
    return cv2.resize(crop, (out_size, out_size))


class AntiSpoof:
    """MiniFASNetV2 (scale 2.7) + MiniFASNetV1SE (scale 4.0), softmax-averaged.
    Class index 1 = real face (as in the original repo)."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.models = []
        for fname, scale, arch_name in config.SPOOF_MODELS:
            net = getattr(arch, arch_name)(conv6_kernel=(5, 5), num_classes=3, img_channel=3)
            state = torch.load(config.MODEL_DIR / fname, map_location=self.device)
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            net.load_state_dict(state)
            self.models.append((net.to(self.device).eval(), scale))

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray, bbox_xyxy) -> tuple[bool, float]:
        """Returns (is_live, live_score in 0..1)."""
        x1, y1, x2, y2 = bbox_xyxy
        bbox = [x1, y1, x2 - x1, y2 - y1]
        probs = []
        for net, scale in self.models:
            patch = _crop(img_bgr, bbox, scale)
            # original repo feeds raw BGR 0-255 values, no normalization
            t = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
            p = torch.softmax(net(t), dim=1)[0].cpu().numpy()
            probs.append(p)
        mean = np.mean(probs, axis=0)
        live_score = float(mean[1])
        return live_score > config.LIVE_THRESHOLD, live_score