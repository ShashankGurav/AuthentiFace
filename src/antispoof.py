import cv2
import numpy as np
import config


# ------------------------------------------------------------------ old model
def _crop_old(img, bbox_xywh, scale, out_size=80):
    """Crop logic from Silent-Face-Anti-Spoofing."""
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


class OldAntiSpoof:
    """MiniFASNetV2 + MiniFASNetV1SE, softmax-averaged. Class index 1 = real."""
    name = "old"

    def __init__(self, device: str = "cpu"):
        import torch                                   # only needed for this model
        from src import minifasnet_arch as arch
        self.torch = torch
        self.device = torch.device(device)
        self.models = []
        for fname, scale, arch_name in config.OLD_MODELS:
            net = getattr(arch, arch_name)(conv6_kernel=(5, 5), num_classes=3, img_channel=3)
            state = torch.load(config.MODEL_DIR / fname, map_location=self.device)
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            net.load_state_dict(state)
            self.models.append((net.to(self.device).eval(), scale))

    def predict(self, img_bgr, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        bbox = [x1, y1, x2 - x1, y2 - y1]
        probs = []
        with self.torch.no_grad():
            for net, scale in self.models:
                patch = _crop_old(img_bgr, bbox, scale)
                t = self.torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
                probs.append(self.torch.softmax(net(t), dim=1)[0].cpu().numpy())
        score = float(np.mean(probs, axis=0)[1])
        return score >= config.LIVE_THRESHOLD["old"], score


# ------------------------------------------------------------------ new model
def _crop_new(img_rgb, bbox_xyxy, pad, size):
    x1, y1, x2, y2 = bbox_xyxy
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    s = int(max(w, h) * pad)
    nx1, ny1 = int(cx - s / 2), int(cy - s / 2)
    nx2, ny2 = nx1 + s, ny1 + s
    H, W = img_rgb.shape[:2]
    nx1, ny1, nx2, ny2 = max(0, nx1), max(0, ny1), min(W, nx2), min(H, ny2)
    crop = img_rgb[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        raise ValueError("empty crop")
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)


class NewAntiSpoof:
    """MiniFASNetV2-SE INT8 ONNX, 128x128 RGB, 2 logits."""
    name = "new"

    def __init__(self, device: str = "cpu"):
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(str(config.NEW_MODEL), providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = inp.shape[2] if isinstance(inp.shape[2], int) else 128

    def predict(self, img_bgr, bbox_xyxy):
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face = _crop_new(rgb, bbox_xyxy, config.NEW_PAD, self.size).astype(np.float32)
        except ValueError:
            return False, 0.0
        if config.NEW_SCALE_255:
            face /= 255.0
        out = self.sess.run(None, {self.input_name: np.transpose(face, (2, 0, 1))[None]})[0][0]
        out = np.asarray(out, dtype=np.float64)
        if not (np.all(out >= 0) and abs(out.sum() - 1) < 1e-3):   # logits -> softmax
            e = np.exp(out - out.max())
            out = e / e.sum()
        score = float(out[config.NEW_REAL_INDEX])
        return score >= config.LIVE_THRESHOLD["new"], score


# -------------------------------------------------------------------- factory
def load_antispoof(model: str | None = None, device: str = "cpu"):
    model = model or config.DEFAULT_SPOOF_MODEL
    if model == "old":
        return OldAntiSpoof(device)
    if model == "new":
        return NewAntiSpoof(device)
    raise ValueError(f"Unknown anti-spoof model '{model}' (use 'old' or 'new')")