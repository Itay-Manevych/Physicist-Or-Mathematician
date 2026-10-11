"""Build the game's data: held-out test people only (the model never trained on them).

Writes game/data.json and game/img/*.jpg (small square portraits around the face).
"""
import json
import random

import pandas as pd
import requests

from .config import DATA, OUTPUTS, ROOT, USER_AGENT
from .faces import detector, main_face, open_image

GAME = ROOT / "game"
IMG_SIZE = 144  # shown at 2x with pixelated rendering


def portrait(img, box, size=IMG_SIZE):
    """Square crop around the face (~3x the face), shifted to stay inside the photo."""
    x1, y1, x2, y2 = box
    s = min(max(x2 - x1, y2 - y1) * 3, *img.size)
    cx, cy = (x1 + x2) / 2, y1 + (y2 - y1) * 0.8
    left = min(max(cx - s / 2, 0), img.size[0] - s)
    top = min(max(cy - s / 2, 0), img.size[1] - s)
    return img.crop((int(left), int(top), int(left + s), int(top + s))).resize((size, size))


def descriptions(qids):
    out = {}
    for i in range(0, len(qids), 50):
        r = requests.get("https://www.wikidata.org/w/api.php", headers={"User-Agent": USER_AGENT}, timeout=60, params={
            "action": "wbgetentities", "ids": "|".join(qids[i:i + 50]), "props": "descriptions|sitelinks",
            "languages": "en", "sitefilter": "enwiki", "format": "json"}).json()
        for q, e in r.get("entities", {}).items():
            title = e.get("sitelinks", {}).get("enwiki", {}).get("title")
            out[q] = {"bio": e.get("descriptions", {}).get("en", {}).get("value", ""),
                      "wiki": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else ""}
    return out


def build(per_class=80, seed=0):
    test = pd.read_csv(OUTPUTS / "test_predictions.csv")
    test = test[test.source == "wiki"]
    clean = pd.read_csv(DATA / "clean.csv")[["qid", "birth_year"]]
    test = test.merge(clean, on="qid")
    pool = pd.concat([g.sample(min(per_class, len(g)), random_state=seed) for _, g in test.groupby("label")])
    (GAME / "img").mkdir(parents=True, exist_ok=True)
    for old in (GAME / "img").glob("*.jpg"):
        old.unlink()
    mtcnn, people = detector(), []
    info = descriptions(pool.qid.tolist())
    for row in pool.itertuples():
        img = open_image(row.file)
        box = main_face(mtcnn, img, min_prob=0.9)
        if box is None:
            continue
        boxes, probs = mtcnn.detect(img)
        main_w = box[2] - box[0]
        if sum(pr > 0.9 and b[2] - b[0] > 0.5 * main_w for b, pr in zip(boxes, probs)) > 1:
            continue  # someone else is in the photo too: ambiguous who you are guessing
        k = len(people)
        portrait(img, box).save(GAME / "img" / f"{k:03d}.jpg", quality=88)
        file_title = row.img.rsplit("/", 1)[1]
        people.append({
            "id": k, "img": f"img/{k:03d}.jpg", "name": row.name, "born": int(row.birth_year),
            "answer": "phys" if row.label == "physicist" else "math", "p": round(float(row.p), 3),
            "bio": info.get(row.qid, {}).get("bio", ""), "wiki": info.get(row.qid, {}).get("wiki", ""),
            "source": f"https://commons.wikimedia.org/wiki/File:{file_title}",
        })
    random.Random(seed).shuffle(people)
    acc = sum((p["p"] > 0.5) == (p["answer"] == "phys") for p in people) / len(people)
    json.dump({"model_accuracy": round(acc, 3), "people": people}, open(GAME / "data.json", "w"), indent=1)
    print(f"{len(people)} fighters -> game/data.json  (model accuracy on them: {acc:.0%})")
