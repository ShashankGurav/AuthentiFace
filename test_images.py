"""Usage: python test_images.py data/test --model old
          python test_images.py a.jpg b.jpg --model new"""
import argparse
import cv2
from pathlib import Path
import config
from src.pipeline import Pipeline
from src.viz import draw

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect(paths):
    files = []
    for a in paths:
        p = Path(a)
        files += [f for f in sorted(p.iterdir()) if f.suffix.lower() in EXTS] if p.is_dir() else [p]
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--model", choices=["old", "new"], default=config.DEFAULT_SPOOF_MODEL)
    args = ap.parse_args()

    pipe = Pipeline(spoof_model=args.model)
    out_dir = config.OUTPUT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model: {args.model}")
    print(f"{'file':35} {'live':6} {'live_score':10} {'identity':14} sim")
    for f in collect(args.paths):
        img = cv2.imread(str(f))
        if img is None:
            continue
        results = pipe.process(img)
        if not results:
            print(f"{f.name:35} no face detected")
        for r in results:
            print(f"{f.name:35} {str(r.live):6} {r.live_score:<10.3f} "
                  f"{str(r.name):14} {'' if r.sim is None else f'{r.sim:.3f}'}")
        cv2.imwrite(str(out_dir / f.name), draw(img, results))


if __name__ == "__main__":
    main()