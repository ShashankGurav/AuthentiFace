import cv2


def draw(img, results):
    for r in results:
        x1, y1, x2, y2 = r.bbox
        if not r.live:
            color, text = (0, 0, 255), f"SPOOF {r.live_score:.2f}"
        elif r.name == "Unknown":
            color, text = (0, 255, 255), f"LIVE {r.live_score:.2f} | Unknown {r.sim:.2f}"
        else:
            color, text = (0, 200, 0), f"LIVE {r.live_score:.2f} | {r.name} {r.sim:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img