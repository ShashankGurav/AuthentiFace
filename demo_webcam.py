import time
import cv2
from src.pipeline import Pipeline
from src.viz import draw


def main(cam=0):
    pipe = Pipeline()
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Camera not available")
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        draw(frame, pipe.process(frame))
        fps = 1 / max(time.time() - t0, 1e-6)
        t0 = time.time()
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Face Recognition + Anti-Spoofing (q to quit)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            cv2.imwrite(f"outputs/shot_{int(time.time())}.jpg", frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)