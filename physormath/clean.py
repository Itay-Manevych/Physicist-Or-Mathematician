"""Build the strict, photos-only Wikidata set (data/clean.csv).

Excluded files are moved (never deleted) to data/excluded/<reason>/:
  not_strict    a non-physics/maths career on Wikidata (astronaut, politician, engineer, ...)
  both_fields   tagged physicist AND mathematician
  painting / drawing / statue / plaque_grave / stamp_coin   not a photograph (CLIP zero-shot)
  shared_photo  same photo used for two people
  manual        listed in data/manual_exclude.txt
  no_face       no usable face
Old black-and-white photos are kept. Contact sheets for review go to data/review/.
"""
import os
import shutil

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image, ImageDraw

from .config import BACKBONES, DATA, DEVICE
from .labels import load_faces

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
_text = {}


def classify(feats, prompts):
    """Zero-shot category probabilities for CLIP-H image embeddings."""
    key = id(prompts)
    if key not in _text:
        arch, pretrained = BACKBONES["clipH"]
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=DEVICE)
        tok = open_clip.get_tokenizer(arch)
        cats, embs = [], []
        with torch.no_grad():
            for c, ps in prompts.items():
                t = model.encode_text(tok(ps).to(DEVICE)).float()
                embs.append(t / t.norm(dim=-1, keepdim=True))
                cats += [c] * len(ps)
        _text[key] = (cats, torch.cat(embs).cpu().numpy())
        del model
    cats, T = _text[key]
    scores = pd.DataFrame(feats @ T.T * 100, columns=cats).T.groupby(level=0).max().T
    p = np.exp(scores - scores.max(1).values[:, None])
    return p.div(p.sum(1), axis=0)


def photo_kind(tag=""):
    """'photo' or the non-photo category, from the whole image and (as a second opinion) the face crop.

    Cached in data/photo_kind<tag>.npy (recomputed when the number of photos changes)."""
    cache = DATA / f"photo_kind{tag}.npy"
    n = np.load(DATA / f"feats_clipH_full{tag}.npy", mmap_mode="r").shape[0]
    if cache.exists():
        kind = np.load(cache, allow_pickle=True)
        if len(kind) == n:
            return kind
    kind = _photo_kind(tag)
    np.save(cache, kind)
    return kind


def _photo_kind(tag):
    full = classify(np.load(DATA / f"feats_clipH_full{tag}.npy"), PROMPTS)
    if not (DATA / f"feats_clipH{tag}.npy").exists():  # no face-crop CLIP-H for this source: whole-image check only
        kind = full.idxmax(1).values.copy()
        kind[full["photo"].values > 0.4] = "photo"
        return kind
    face = classify(np.load(DATA / f"feats_clipH{tag}.npy"), FACE_PROMPTS)
    kind = full.idxmax(1).values.copy()
    kind[full["photo"].values > 0.4] = "photo"
    face_kind = face.idxmax(1).values
    flag = (kind == "photo") & (face_kind != "photo") & (face["photo"].values < 0.4)
    kind[flag] = face_kind[flag]
    return kind


def contact_sheet(files, path, n=48, size=128, cols=8):
    files = list(files)[:n]
    if not files:
        return
    sheet = Image.new("RGB", (cols * size, ((len(files) + cols - 1) // cols) * size), "white")
    for k, f in enumerate(files):
        x, y = (k % cols) * size, (k // cols) * size
        try:
            im = Image.open(f).convert("RGB")
            im.thumbnail((size, size))
            sheet.paste(im, (x + (size - im.size[0]) // 2, y + (size - im.size[1]) // 2))
        except Exception:
            ImageDraw.Draw(sheet).text((x + 5, y + 5), "?", fill="red")
    sheet.save(path)


def clean(apply=False):
    df = load_faces()
    df["row"] = np.arange(len(df))  # row index into data/feats_*.npy
    df["kind"] = photo_kind()
    df["reason"] = np.where(df.kind != "photo", df.kind, "")
    manual_path = DATA / "manual_exclude.txt"
    if manual_path.exists():
        manual = {l.strip() for l in open(manual_path) if l.strip() and not l.startswith("#")}
        df.loc[df.file.isin(manual), "reason"] = "manual"
    df.loc[df.label == "both", "reason"] = "both_fields"
    df.loc[(df.reason == "") & ~df.strict, "reason"] = "not_strict"
    kept_files = df.file[df.reason == ""]
    df.loc[kept_files.index[kept_files.duplicated(keep=False)], "reason"] = "shared_photo"
    kept = df[df.reason == ""]
    people = pd.read_csv(DATA / "people.csv")
    no_face = people[people.downloaded & ~people.file.isin(df.file)].file

    print("kept:", len(kept), kept.label.value_counts().to_dict())
    print("excluded:", df[df.reason != ""].reason.value_counts().to_dict(), "| no_face:", len(no_face))
    (DATA / "review").mkdir(exist_ok=True)
    for r in ["painting", "drawing", "statue", "plaque_grave", "stamp_coin", "not_strict"]:
        contact_sheet(df[df.reason == r].file, DATA / "review" / f"excluded_{r}.jpg")
    contact_sheet(kept.sample(min(48, len(kept)), random_state=0).file, DATA / "review" / "kept_random.jpg")
    if not apply:
        print("dry run - rerun with --apply to move files")
        return

    keep_names = {os.path.basename(f) for f in kept.file}

    def move(f, reason):
        if isinstance(f, str) and os.path.exists(f) and os.path.basename(f) not in keep_names:
            d = DATA / "excluded" / reason / os.path.dirname(f).split("/")[-1]
            d.mkdir(parents=True, exist_ok=True)
            shutil.move(f, d / os.path.basename(f))

    for row in df[df.reason != ""].itertuples():
        for f in [row.file, row.face, "data/upper/" + os.path.basename(row.file)]:
            move(f, row.reason)
    for f in no_face:
        move(f, "no_face")
    # files excluded by an earlier run but kept now (e.g. after a label change) are moved back
    for row in kept.itertuples():
        for f in [row.file, row.face, "data/upper/" + os.path.basename(row.file)]:
            if not os.path.exists(f):
                hits = list((DATA / "excluded").glob(f"*/{os.path.dirname(f).split('/')[-1]}/{os.path.basename(f)}"))
                if hits:
                    shutil.move(hits[0], f)
    kept.to_csv(DATA / "clean.csv", index=False)
    print("moved excluded files to data/excluded/, wrote data/clean.csv")
