"""Build the game's data: held-out test people only (the model never trained on them).

Writes game/data.json and game/img/*.jpg (small square portraits around the face).
"""
import json

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


def build():
    """Every held-out test person with a clear single-face photo becomes a fighter.

    Incremental: fighters already in game/data.json keep their id and portrait (saved games and dailies refer to
    ids); new people are appended with new ids. Model scores and clues are refreshed for everyone.
    """
    test = pd.read_csv(OUTPUTS / "test_predictions.csv")
    test = test[test.source == "wiki"].merge(pd.read_csv(DATA / "clean.csv")[["qid", "birth_year"]], on="qid")
    test["source_url"] = "https://commons.wikimedia.org/wiki/File:" + test.img.str.rsplit("/", n=1).str[1]
    old = json.load(open(GAME / "data.json"))["people"] if (GAME / "data.json").exists() else []
    by_source = {p["source"]: p for p in old}
    (GAME / "img").mkdir(parents=True, exist_ok=True)
    info = descriptions(test.qid.tolist())
    mtcnn, people, next_id = detector(), [], max([p["id"] for p in old], default=-1) + 1
    for row in test.sort_values("qid").itertuples():
        kept = by_source.get(row.source_url)
        if kept:
            k, img_path = kept["id"], kept["img"]
        else:
            img = open_image(row.file)
            box = main_face(mtcnn, img, min_prob=0.9)
            if box is None:
                continue
            boxes, probs = mtcnn.detect(img)
            main_w = box[2] - box[0]
            if sum(pr > 0.9 and b[2] - b[0] > 0.5 * main_w for b, pr in zip(boxes, probs)) > 1:
                continue  # someone else is in the photo too: ambiguous who you are guessing
            k, next_id = next_id, next_id + 1
            img_path = f"img/{k:03d}.jpg"
            portrait(img, box).save(GAME / img_path, quality=88)
        people.append({
            "id": k, "qid": row.qid, "img": img_path, "name": row.name, "born": int(row.birth_year),
            "answer": "phys" if row.label == "physicist" else "math", "p": round(float(row.p), 3),
            "bio": info.get(row.qid, {}).get("bio", ""), "wiki": info.get(row.qid, {}).get("wiki", ""),
            "source": row.source_url,
        })
    # "the machine noticed ..." clues, with concept directions learned on training people only
    clean_all = pd.read_csv(DATA / "clean.csv")
    split = pd.read_csv(OUTPUTS / "split.csv").set_index("group").test
    train_mask = ~clean_all.qid.map(split).fillna(True).astype(bool).values
    row_of = {q: i for i, q in enumerate(clean_all.qid)}
    picks = ["phys" if p["p"] >= 0.5 else "math" for p in people]
    for p, c in zip(people, concept_clues(clean_all, [row_of[p["qid"]] for p in people], train_mask, picks)):
        p["clues"] = c
    people.sort(key=lambda p: p["id"])
    acc = sum((p["p"] >= 0.5) == (p["answer"] == "phys") for p in people) / len(people)
    json.dump({"model_accuracy": round(acc, 3), "people": people}, open(GAME / "data.json", "w"), indent=0)
    credits = ["# Photo credits", "",
               "Portraits are cropped from freely licensed images on Wikimedia Commons.",
               "Author and license for each photo are on its Commons page.", ""]
    credits += [f"- `{p['img']}` {p['name']}: {p['source']}" for p in people]
    (GAME / "CREDITS.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    print(f"{len(people)} fighters ({len(people) - len(old)} new) -> game/data.json  (model accuracy on them: {acc:.0%})")
