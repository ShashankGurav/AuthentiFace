import json
import numpy as np
import config


def load_db() -> dict:
    if not config.DB_PATH.exists():
        return {"people": []}
    return json.loads(config.DB_PATH.read_text())


def save_person(name: str, embeddings: list[np.ndarray]) -> None:
    db = load_db()
    db["people"] = [p for p in db["people"] if p["name"] != name]  # re-enroll replaces
    db["people"].append({"name": name,
                         "embeddings": [e.astype(float).tolist() for e in embeddings]})
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.DB_PATH.write_text(json.dumps(db))


def load_gallery() -> tuple[np.ndarray, list[str]]:
    """Returns matrix (N, 512) float32 and a parallel list of N names (N = 12 here)."""
    db = load_db()
    mats, labels = [], []
    for p in db["people"]:
        for e in p["embeddings"]:
            mats.append(e)
            labels.append(p["name"])
    if not mats:
        return np.zeros((0, 512), np.float32), []
    return np.asarray(mats, dtype=np.float32), labels