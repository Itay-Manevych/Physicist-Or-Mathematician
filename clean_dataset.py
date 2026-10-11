"""Build the strict, photos-only dataset.

Removes (moves to data/excluded/<reason>/, nothing is deleted):
  not_strict   - Wikidata lists a non-physics/maths career (astronaut, politician, engineer, ...)
  both_fields  - tagged physicist AND mathematician
  painting / drawing / statue / plaque_grave / stamp_coin - not a photograph of the person (CLIP zero-shot)
  no_face      - no usable face found
Old black-and-white photos are kept.

  python clean_dataset.py            # dry run: counts + contact sheets in data/review/
  python clean_dataset.py --apply    # move files, write data/clean.csv
"""
import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image, ImageDraw
from labels import load_faces

PROMPTS = {
    "photo": ["a photograph of a person", "a black and white photograph of a person", "an old photograph of a man",
              "a portrait photo of a scientist", "a candid photo of a person", "a photo of a person giving a lecture"],
    "painting": ["a painting of a person", "an oil painting portrait", "a watercolor portrait"],
    "drawing": ["a pencil drawing of a person", "a sketch of a face", "an engraving of a person", "a caricature",
                "a cartoon drawing of a person", "a digital illustration of a person"],
    "statue": ["a statue of a person", "a bronze bust", "a marble sculpture of a head"],
    "plaque_grave": ["a gravestone", "a memorial plaque", "a commemorative plaque on a wall"],
    "stamp_coin": ["a postage stamp", "a postage stamp with a portrait"],
}
# second opinion on the face crop itself (what the classifier actually trains on)
FACE_PROMPTS = {
    "photo": ["a photograph of a face", "a black and white photograph of a face", "a close-up photo of a person"],
    "painting": ["a painted portrait of a face", "an oil painting of a face"],
    "drawing": ["a pencil drawing of a face", "a sketch of a face", "an engraving of a face", "a cartoon face"],
    "statue": ["the face of a bronze statue", "the face of a stone sculpture"],
    "stamp_coin": ["a face on a postage stamp"],
}
_clip = {}


def classify(feats, prompts=PROMPTS):
    if not _clip:
        _clip["model"], _, _ = open_clip.create_model_and_transforms("ViT-H-14-quickgelu", pretrained="dfn5b", device="cuda")
        _clip["tok"] = open_clip.get_tokenizer("ViT-H-14-quickgelu")
    model, tok = _clip["model"], _clip["tok"]
    cats, embs = [], []
    with torch.no_grad():
        for c, ps in prompts.items():
            t = model.encode_text(tok(ps).cuda()).float()
            embs.append(t / t.norm(dim=-1, keepdim=True))
            cats += [c] * len(ps)
    T = torch.cat(embs).cpu().numpy()
    sims = feats @ T.T * 100
    cat_scores = pd.DataFrame(sims, columns=cats).T.groupby(level=0).max().T
    probs = np.exp(cat_scores - cat_scores.max(1).values[:, None])
    probs = probs.div(probs.sum(1), axis=0)
    return probs


def contact_sheet(files, path, n=48, size=128, cols=8):
    files = list(files)[:n]
    if not files:
        return
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * size), "white")
    for k, f in enumerate(files):
        try:
            im = Image.open(f).convert("RGB")
            im.thumbnail((size, size))
            sheet.paste(im, ((k % cols) * size + (size - im.size[0]) // 2, (k // cols) * size + (size - im.size[1]) // 2))
        except Exception:
            ImageDraw.Draw(sheet).text(((k % cols) * size + 5, (k // cols) * size + 5), "?", fill="red")
    sheet.save(path)


def main(apply):
    df = load_faces()
    df["row"] = np.arange(len(df))  # row index into data/feats_*.npy
    probs = classify(np.load("data/feats_clipH_full.npy"))
    df["kind"] = probs.idxmax(1).values
    df["p_photo"] = probs["photo"].values
    # call it a non-photo only when that beats "photo" clearly
    df.loc[df.p_photo > 0.4, "kind"] = "photo"
    face_probs = classify(np.load("data/feats_clipH.npy"), FACE_PROMPTS)
    face_kind = face_probs.idxmax(1).values
    flag = (df.kind == "photo").values & (face_kind != "photo") & (face_probs["photo"].values < 0.4)
    df["face_flag"] = flag
    df.loc[flag, "kind"] = face_kind[flag]

    df["reason"] = ""
    df.loc[df.kind != "photo", "reason"] = df.kind
    # manual overrides: one file path per line in data/manual_exclude.txt
    if os.path.exists("data/manual_exclude.txt"):
        manual = {l.strip() for l in open("data/manual_exclude.txt") if l.strip() and not l.startswith("#")}
        df.loc[df.file.isin(manual), "reason"] = "manual"
    df.loc[df.label == "both", "reason"] = "both_fields"
    df.loc[(df.reason == "") & ~df.strict, "reason"] = "not_strict"

    meta = pd.read_csv("data/people.csv")
    no_face = meta[meta.downloaded & ~meta.file.isin(df.file)]

    print("kept:", (df.reason == "").sum(), df[df.reason == ""].label.value_counts().to_dict())
    print("excluded:", df[df.reason != ""].reason.value_counts().to_dict(), "| no_face:", len(no_face))
    os.makedirs("data/review", exist_ok=True)
    for r in ["painting", "drawing", "statue", "plaque_grave", "stamp_coin", "not_strict"]:
        contact_sheet(df[df.reason == r].sort_values("p_photo").file, f"data/review/excluded_{r}.jpg")
    contact_sheet(df[df.face_flag & df.strict & (df.label != "both")].file, "data/review/excluded_by_face_check.jpg")
    kept = df[df.reason == ""]
    least = kept.sort_values("p_photo")
    contact_sheet(least.file, "data/review/kept_least_photo_like.jpg")
    least.head(48)[["file", "name"]].to_csv("data/review/kept_least_photo_like.csv", index=False)
    contact_sheet(kept.sample(48, random_state=0).file, "data/review/kept_random.jpg")

    if apply:
        def move(f, reason):
            if isinstance(f, str) and os.path.exists(f):
                d = f"data/excluded/{reason}/{os.path.dirname(f).split('/')[-1]}"
                os.makedirs(d, exist_ok=True)
                shutil.move(f, f"{d}/{os.path.basename(f)}")
        for row in df[df.reason != ""].itertuples():
            for f in [row.file, row.face, "data/upper/" + os.path.basename(row.file)]:
                move(f, row.reason)
        for f in no_face.file:
            move(f, "no_face")
        kept.to_csv("data/clean.csv", index=False)
        print("moved excluded files to data/excluded/, wrote data/clean.csv")


if __name__ == "__main__":
    main("--apply" in sys.argv)
