import argparse
import time
import cv2
import config
from src.pipeline import Pipeline
from src.viz import draw

MODELS = {
    "old": "MiniFASNet V2 + V1SE ensemble (80x80, PyTorch)",
    "new": "MiniFASNetV2-SE INT8 (128x128, ONNX)",
}


def choose_model() -> str:
    print("\nSelect anti-spoof model:")
    print(f"  1) old - {MODELS['old']}")
    print(f"  2) new - {MODELS['new']}")
    while True:
        c = input("Enter 1 or 2: ").strip()
        if c in ("1", "old"):
            return "old"
        if c in ("2", "new"):
            return "new"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["old", "new"], help="anti-spoof model (asks if omitted)")
    ap.add_argument("--cam", type=int, default=config.CAMERA_INDEX, help="camera index")
    args = ap.parse_args()

    model = args.model or choose_model()
    print(f"Using anti-spoof model: {model} - {MODELS[model]}")
    pipe = Pipeline(spoof_model=model)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Camera {args.cam} not available")
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    win = f"Face Recognition + Anti-Spoofing [{model}]  (q / ESC quit, s = screenshot)"
    cv2.namedWindow(win)
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw(frame, pipe.process(frame))
            now = time.time()
            cv2.putText(frame, f"[{model}] FPS {1 / max(now - t0, 1e-6):.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            t0 = now
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                cv2.imwrite(str(config.OUTPUT_DIR / f"shot_{model}_{int(time.time())}.jpg"), frame)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()