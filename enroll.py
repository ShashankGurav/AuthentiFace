"""Enroll one person at a time: prompts for name, then builds embeddings
from 4 photos in a folder (or captures 4 from the webcam)."""
import argparse
import cv2
from pathlib import Path
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


def from_webcam(engine, n=4):
    cap = cv2.VideoCapture(0)
    embs = []
    print("SPACE = capture, q = abort")
    while len(embs) < n:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, f"Captured {len(embs)}/{n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Enroll", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == 32:
            face = engine.largest_face(frame)
            if face is None:
                print("  no face detected, try again")
                continue
            embs.append(face.normed_embedding)
    cap.release()
    cv2.destroyAllWindows()
    return embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", action="store_true", help="capture samples from webcam")
    args = ap.parse_args()

    engine = FaceEngine()
    while True:
        name = input("\nEnter person name (blank to finish): ").strip()
        if not name:
            break
        if args.webcam:
            embs = from_webcam(engine)
        else:
            folder = input("Folder with this person's photos: ").strip()
            embs = from_folder(engine, folder)
        if len(embs) < 1:
            print("No valid embeddings, nothing saved.")
            continue
        save_person(name, embs)
        print(f"Saved '{name}' with {len(embs)} embeddings.")

    matrix, labels = load_gallery()
    print(f"\nGallery now holds {len(labels)} embeddings, {len(set(labels))} people.")


if __name__ == "__main__":
    main()