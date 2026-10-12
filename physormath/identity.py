"""Strict checks that an extra photo of a person really shows that person and is not a copy of one we have.

Extra photos (data/faces_<source>.csv, found via Commons categories, "depicts" tags, other Wikipedias) are only
used when every check passes:
  photo    CLIP says it is a real photograph (not a painting, drawing, statue, plaque ...)
  alone    one clear face: nobody else of similar size is in the picture, so there is no doubt who it shows
  same     FaceNet similarity to the main portrait >= STRONG, or >= WEAK and also >= WEAK to another photo of
           them that passed with STRONG (two independent confirmations)
  unique   no other person's main portrait is clearly more similar (lookalike margin)
  gender   CLIP's man/woman call does not contradict the main portrait
  new      not a copy of a photo we already have (re-upload, crop, recolouring)
  manual   not listed in data/identity_reject.txt (rejected by eye)
-> data/verified.csv; contact sheets of the closest calls -> outputs/identity/.
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

from .clean import classify, photo_kind
from .config import DATA, OUTPUTS
from .embed import resolve
from .faces import detector, open_image

SOURCES = ["extra", "depicts", "more", "wiki"]
STRONG, WEAK = 0.70, 0.62  # FaceNet cosine; different people exceed them in 0.03% / 0.13% of pairs
MARGIN = 0.05              # reject when another person's portrait is this much more similar
COPY_FACE, COPY_FULL, COPY_HASH = 0.97, 0.95, 6  # near-identical face / whole image / perceptual-hash bits
GENDER = {"man": ["a photo of a man", "a portrait photo of a man", "an old man", "a young man"],
          "woman": ["a photo of a woman", "a portrait photo of a woman", "an old woman", "a young woman"]}


def _cached(name, n, compute):
    path = DATA / name
    if path.exists():
        a = np.load(path)
        if len(a) == n:
            return a
    a = compute()
    np.save(path, a)
    return a


def crowd(tag):
    """Per photo: how many other confident faces are at least half the size of the main one."""
    df = pd.read_csv(DATA / f"faces{tag}.csv")

    def compute():
        mtcnn, out = detector(), []
        for f in tqdm(df.file, desc=f"counting faces{tag}"):
            try:
                boxes, probs = mtcnn.detect(open_image(resolve(f)))
            except Exception:
                boxes = None
            if boxes is None:
                out.append(0)
                continue
            i = int(np.argmax(probs))
            w = boxes[i][2] - boxes[i][0]
            out.append(sum(p > 0.9 and b[2] - b[0] > 0.5 * w for b, p in zip(boxes, probs)) - 1)
        return np.array(out)
    return _cached(f"crowd{tag}.npy", len(df), compute)


def dhash(tag):
    """64-bit difference hash of each whole image (survives resizing, recompression, mild recolouring)."""
    df = pd.read_csv(DATA / f"faces{tag}.csv")

    def one(f):
        try:
            a = np.asarray(open_image(resolve(f)).convert("L").resize((9, 8)), dtype=np.int16)
            return np.packbits(a[:, 1:] > a[:, :-1])
        except Exception:
            return np.zeros(8, np.uint8) + 255
    return _cached(f"dhash{tag}.npy", len(df), lambda: np.stack([one(f) for f in tqdm(df.file, desc=f"hashing{tag}")]))


def woman(tag):
    """CLIP-H zero-shot probability that the face (or, without face features, the whole photo) is a woman."""
    face = DATA / f"feats_clipH{tag}.npy"
    feats = np.load(face if face.exists() else DATA / f"feats_clipH_full{tag}.npy")
    return _cached(f"woman{tag}.npy", len(feats), lambda: classify(feats, GENDER)["woman"].values)


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _candidates(w):
    parts = []
    for src in SOURCES:
        tag = f"_{src}"
        if not all((DATA / f).exists() for f in [f"faces{tag}.csv", f"feats_facenet{tag}.npy", f"feats_clipH_full{tag}.npy"]):
            continue
        d = pd.read_csv(DATA / f"faces{tag}.csv")
        d = d.assign(source=src, i=np.arange(len(d)), photo=photo_kind(tag) == "photo", crowd=crowd(tag),
                     woman=woman(tag))
        keep = d.qid.isin(set(w.qid)).values
        parts.append((d[keep], np.load(DATA / f"feats_facenet{tag}.npy")[keep],
                      _unit(np.load(DATA / f"feats_clipH_full{tag}.npy")[keep]), dhash(tag)[keep]))
    c = pd.concat([p[0] for p in parts], ignore_index=True)
    return c, *(np.concatenate([p[k] for p in parts]) for k in (1, 2, 3))


def _bits(a, b):
    return np.unpackbits(np.bitwise_xor(a, b), axis=-1).sum(-1)


def verify():
    w = pd.read_csv(DATA / "clean.csv")
    R = np.load(DATA / "feats_facenet.npy")[w.row.values]
    main = {q: k for k, q in enumerate(w.qid)}
    main_full = _unit(np.load(DATA / "feats_clipH_full.npy")[w.row.values])
    main_hash, main_woman = dhash("")[w.row.values], woman("")[w.row.values]
    c, F, G, H = _candidates(w)
    k = c.qid.map(main).values

    S = F @ R.T
    c["sim"] = S[np.arange(len(c)), k]
    S[np.arange(len(c)), k] = -1
    c["lookalike"] = S.max(1)
    # expected gender: the main portrait's when CLIP is sure about it, otherwise Wikidata's
    mw = main_woman[k]
    wd = w.gender.values[k]
    expect = np.where(mw > 0.8, 1.0, np.where(mw < 0.2, 0.0, np.where(wd == "female", 1.0, np.where(wd == "male", 0.0, np.nan))))
    c["gender_clash"] = ((expect == 1) & (c.woman < 0.2)) | ((expect == 0) & (c.woman > 0.8))

    manual = set()
    if (DATA / "identity_reject.txt").exists():
        manual = {l.strip() for l in open(DATA / "identity_reject.txt") if l.strip() and not l.startswith("#")}
    c["reason"] = ""

    def fail(mask, why):
        c.loc[(c.reason == "") & mask, "reason"] = why
    fail(~c.photo, "not a photo")
    fail(c.crowd > 0, "other faces")
    fail(c.sim < WEAK, "different person")
    fail(c.lookalike > c.sim + MARGIN, "lookalike")
    fail(c.gender_clash, "gender")
    fail(c.face.isin(manual), "manual")
    # weak matches need a second confirmation from a strong photo of the same person
    for q, g in c[(c.reason == "")].groupby("qid"):
        strong = g.index[g.sim >= STRONG].values
        for i in g.index[g.sim < STRONG]:
            if not (F[strong] @ F[i] >= WEAK).any():
                c.loc[i, "reason"] = "unconfirmed"
    # copies: keep the best-matching version, drop anything identical to the main portrait or a kept photo;
    # a copy of a photo rejected by eye is rejected too
    def same_image(i, js):
        return any(F[j] @ F[i] >= COPY_FACE or G[j] @ G[i] >= COPY_FULL or _bits(H[j], H[i]) <= COPY_HASH for j in js)
    for q, g in c[c.reason == ""].sort_values("sim", ascending=False).groupby("qid", sort=False):
        m = main[q]
        bad = c.index[(c.qid == q) & (c.reason == "manual")]
        kept = []
        for i in g.index:
            if same_image(i, bad):
                c.loc[i, "reason"] = "manual"
            elif (R[m] @ F[i] >= COPY_FACE or main_full[m] @ G[i] >= COPY_FULL
                  or _bits(main_hash[m], H[i]) <= COPY_HASH or same_image(i, kept)):
                c.loc[i, "reason"] = "copy"
            else:
                kept.append(i)
    c["ok"] = c.reason == ""
    c.drop(columns=["photo"]).to_csv(DATA / "verified.csv", index=False)
    print(c.groupby("source").ok.agg(["sum", "size"]).rename(columns={"sum": "kept", "size": "candidates"}).to_string())
    print("rejected:", c.reason[~c.ok].value_counts().to_dict())
    print("people with extra photos:", c[c.ok].qid.nunique())
    sheets(c, w)
    return c


def verified(source):
    """Boolean per row of data/faces_<source>.csv: passed every check in data/verified.csv."""
    n = len(pd.read_csv(DATA / f"faces_{source}.csv"))
    if not (DATA / "verified.csv").exists():
        verify()
    v = pd.read_csv(DATA / "verified.csv")
    v = v[(v.source == source) & v.ok]
    ok = np.zeros(n, bool)
    ok[v.i.values[v.i.values < n]] = True
    return ok


def sheets(c, w, per=20):
    """Closest accepted calls (main portrait | accepted photo), lowest similarity first, for checking by eye."""
    out = OUTPUTS / "identity"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    faces = pd.read_csv(DATA / "faces.csv").face.values
    main_face = dict(zip(w.qid, faces[w.row.values]))
    ok = c[c.ok].sort_values("sim")
    ok = ok[ok.sim < 0.78]
    S = 120
    for n in range(0, len(ok), per):
        chunk = ok.iloc[n:n + per]
        sheet = Image.new("RGB", (S * 4, S * ((len(chunk) + 1) // 2)), "white")
        dr = ImageDraw.Draw(sheet)
        for j, r in enumerate(chunk.itertuples()):
            x, y = (j % 2) * S * 2, (j // 2) * S
            for t, p in enumerate([main_face[r.qid], r.face]):
                try:
                    sheet.paste(Image.open(resolve(p)).convert("RGB").resize((S - 4, S - 4)), (x + t * S, y))
                except Exception:
                    pass
            dr.text((x + S + 3, y + 2), f"{j} {r.sim:.2f}", fill="yellow")
        sheet.save(out / f"sheet_{n // per:03d}.png")
        chunk[["face", "qid", "sim"]].assign(slot=range(len(chunk))).to_csv(out / f"sheet_{n // per:03d}.csv", index=False)
    print(f"{len(ok)} closest accepts in {out} ({(len(ok) + per - 1) // per} sheets)")
