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


# visual concepts for "the machine noticed ..." (CLIP prompt -> what the game shows)
CONCEPTS = {
    "a person wearing a suit and tie": "a suit and tie", "a person wearing a sweater": "a sweater",
    "a person wearing a t-shirt": "a t-shirt", "a person with messy hair": "messy hair",
    "a person with long hair": "long hair", "a man with a beard": "a beard", "a bald man": "a shiny bald head",
    "a person wearing glasses": "glasses", "a smiling person": "a big smile", "a serious person": "a serious stare",
    "a confident person": "confidence", "a person in front of a blackboard": "a blackboard",
    "a person in a laboratory": "lab equipment", "a person giving a talk": "a lecture pose",
    "a professional headshot": "a studio headshot", "a candid snapshot": "a candid snapshot",
    "an old black and white photo": "an old black-and-white photo", "a person who looks like a hippie": "hippie vibes",
    "a nerdy person": "nerd energy", "a person outdoors in nature": "the great outdoors",
    "a person in front of bookshelves": "bookshelves", "a person wearing a hat": "a hat",
}


def concept_clues(clean, game_rows, train_mask, picks, k=2):
    """For each game fighter: up to k concepts that stand out in their photo AND lean the way the model picked."""
    import numpy as np
    import open_clip
    import torch
    from sklearn.metrics import roc_auc_score
    from .config import BACKBONES, DEVICE

    arch, pretrained = BACKBONES["clipH"]
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=DEVICE)
    tok = open_clip.get_tokenizer(arch)
    with torch.no_grad():
        T = model.encode_text(tok([f"a photo of {c}" if not c.startswith(("a professional", "a candid", "an old")) else c
                                   for c in CONCEPTS]).to(DEVICE)).float()
    T = (T / T.norm(dim=-1, keepdim=True)).cpu().numpy()
    del model
    X = np.load(DATA / "feats_clipH.npy")[clean.row.values] + np.load(DATA / "feats_clipH_full.npy")[clean.row.values]
    sims = X @ T.T
    sims -= sims.mean(1, keepdims=True)                    # relative to this photo's other concepts
    z = (sims - sims.mean(0)) / sims.std(0)                # how unusual for this photo vs everyone
    y = (clean.label == "physicist").values
    lean = np.array([roc_auc_score(y[train_mask], sims[train_mask, j]) - 0.5 for j in range(len(CONCEPTS))])
    names = list(CONCEPTS.values())
    out = []
    for r, pick in zip(game_rows, picks):
        direction = 1 if pick == "phys" else -1
        score = z[r] * np.clip(lean * direction, 0, None)  # stands out here AND points toward the pick
        best = [j for j in np.argsort(-score)[:k] if score[j] > 0.01 and z[r, j] > 0.5]
        out.append([names[j] for j in best])
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
            "id": k, "qid": row.qid, "img": f"img/{k:03d}.jpg", "name": row.name, "born": int(row.birth_year),
            "answer": "phys" if row.label == "physicist" else "math", "p": round(float(row.p), 3),
            "bio": info.get(row.qid, {}).get("bio", ""), "wiki": info.get(row.qid, {}).get("wiki", ""),
            "source": f"https://commons.wikimedia.org/wiki/File:{file_title}",
        })
    # "the machine noticed ..." clues, with concept directions learned on training people only
    clean_all = pd.read_csv(DATA / "clean.csv")
    split = pd.read_csv(OUTPUTS / "split.csv").set_index("group").test
    train_mask = ~clean_all.qid.map(split).fillna(True).astype(bool).values
    row_of = {q: i for i, q in enumerate(clean_all.qid)}
    picks = ["phys" if p["p"] >= 0.5 else "math" for p in people]
    clues = concept_clues(clean_all, [row_of[p["qid"]] for p in people], train_mask, picks)
    for p, c in zip(people, clues):
        p["clues"] = c
        del p["qid"]
    random.Random(seed).shuffle(people)
    acc = sum((p["p"] > 0.5) == (p["answer"] == "phys") for p in people) / len(people)
    json.dump({"model_accuracy": round(acc, 3), "people": people}, open(GAME / "data.json", "w"), indent=1)
    credits = ["# Photo credits", "",
               "Portraits are cropped from freely licensed images on Wikimedia Commons.",
               "Author and license for each photo are on its Commons page.", ""]
    credits += [f"- `{p['img']}` {p['name']}: {p['source']}" for p in sorted(people, key=lambda p: p["id"])]
    (GAME / "CREDITS.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    print(f"{len(people)} fighters -> game/data.json  (model accuracy on them: {acc:.0%})")
