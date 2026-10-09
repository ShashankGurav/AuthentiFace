"""Enroll one person at a time: prompts for name, then builds embeddings
from photos in a folder (or captures 4 from the webcam)."""
import argparse
import cv2
from pathlib import Path
import config
from src.face_engine import FaceEngine
from src.database import save_person, load_gallery

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def from_folder(engine, folder):
    embs = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(p))
        face = engine.largest_face(img) if img is not None else None
        if face is None:
            print(f"  [skip] no face in {p.name}")
            continue
        embs.append(face.normed_embedding)
        print(f"  [ok]   {p.name}")
    return embs


def from_webcam(engine, n=4, cam=config.CAMERA_INDEX):
    """Returns n embeddings, or None if the user aborted."""
    win = "Enroll  (SPACE = capture, q / ESC = quit)"
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  Camera {cam} not available")
        return None
    cv2.namedWindow(win)
    embs, aborted = [], False
    try:
        while len(embs) < n:
            ok, frame = cap.read()
            if not ok:
                aborted = True
                break
            view = frame.copy()
            cv2.putText(view, f"Captured {len(embs)}/{n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF

            # quit: q / ESC / window X button
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                aborted = True
                break
            if key == 32:  # SPACE
                face = engine.largest_face(frame)
                if face is None:
                    print("  no face detected, try again")
                    continue
                embs.append(face.normed_embedding)
                print(f"  captured {len(embs)}/{n}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):          # let Windows actually close the window
            cv2.waitKey(1)
    return None if aborted else embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", action="store_true", help="capture samples from webcam")
    ap.add_argument("--cam", type=int, default=config.CAMERA_INDEX, help="camera index")
    args = ap.parse_args()

    engine = FaceEngine()
    try:
        while True:
            name = input("\nEnter person name (blank to finish): ").strip()
            if not name:
                break
            if args.webcam:
                embs = from_webcam(engine, cam=args.cam)
                if embs is None:
                    print("Capture aborted, nothing saved.")
                    continue
            else:
                folder = input("Folder with this person's photos: ").strip()
                embs = from_folder(engine, folder)
            if not embs:
                print("No valid embeddings, nothing saved.")
                continue
            save_person(name, embs)
            print(f"Saved '{name}' with {len(embs)} embeddings.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()

    _, labels = load_gallery()
    print(f"\nGallery now holds {len(labels)} embeddings, {len(set(labels))} people.")


if __name__ == "__main__":
    main()