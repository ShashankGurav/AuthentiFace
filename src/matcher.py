import numpy as np
import config


class Matcher:
    """N:1 matching: one probe vs all N gallery embeddings via one matrix product."""

    def __init__(self, matrix: np.ndarray, labels: list[str]):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) if len(matrix) else 1
        self.matrix = matrix / norms if len(matrix) else matrix
        self.labels = labels

    def match(self, probe: np.ndarray, threshold: float = None) -> tuple[str, float]:
        threshold = config.MATCH_THRESHOLD if threshold is None else threshold
        if len(self.labels) == 0:
            return "Unknown", 0.0
        probe = probe / np.linalg.norm(probe)
        sims = self.matrix @ probe          # cosine similarities, shape (N,)
        i = int(np.argmax(sims))
        score = float(sims[i])
        return (self.labels[i] if score >= threshold else "Unknown"), score