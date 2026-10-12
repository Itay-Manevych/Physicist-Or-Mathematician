"""Fine-tuning on a free Kaggle GPU (this PC's 6 GB GPU is too small to fine-tune MetaCLIP-2 well).

  python -m physormath kaggle pack     # data/kaggle/pom-data.zip: photos (resized) + meta.csv with the fixed split
  (upload the zip as a Kaggle dataset, run kaggle/finetune_metaclip2.ipynb, download predictions.csv)
  python -m physormath kaggle blend    # blend the fine-tuned predictions with the production model, report on test

Roles in meta.csv: "train" (fit), "val" (15% of the training people, held out to pick the epoch and the blend
weight), "test" (the fixed test people; their extra photos are included for multi-photo scoring). Training-only
sources (loose, older) are always "train". The notebook never uses test labels for any choice.
"""
import json
import shutil
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .config import DATA, OUTPUTS, ROOT
from .embed import resolve

PACK = DATA / "kaggle"
MAX_SIDE = 448
VAL_SHARE = 0.15


def _rows():
    from .train import extra_photos, load_sources, split, TRAIN_ONLY
    df, _ = load_sources()
    test = split(df)
    rows = df.assign(test=test)[["file", "face", "group", "label", "source", "test"]].copy()
    w = pd.read_csv(DATA / "clean.csv")
    test_groups = set(rows.group[rows.test])
    for source in ["depicts", "more"]:  # identity-verified extra photos
        got = extra_photos(source, w)
        if got is not None:
            d = got[0]
            rows = pd.concat([rows, pd.DataFrame({"file": d.file.values, "face": d.face.values, "group": d.qid.values, "label": d.label.values,
                                                  "source": source, "test": d.qid.isin(test_groups).values})],
                             ignore_index=True)
    rng = np.random.default_rng(0)
    strict_train = rows[~rows.test & ~rows.source.isin(TRAIN_ONLY)].group.unique()
    val_groups = set(rng.choice(strict_train, int(VAL_SHARE * len(strict_train)), replace=False))
    rows["role"] = np.where(rows.test, "test", np.where(rows.group.isin(val_groups) & ~rows.source.isin(TRAIN_ONLY), "val", "train"))
    rows["y"] = (rows.label == "physicist").astype(int)
    rows["main"] = rows.source.isin(["wiki", "new"])  # the person's main portrait (what single-photo scores use)
    return rows.drop_duplicates("file").reset_index(drop=True)


def pack():
    rows = _rows()
    out = PACK / "pom-data"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "faces").mkdir()
    keep = []
    for i, r in enumerate(rows.itertuples()):
        try:
            img = ImageOps.exif_transpose(Image.open(resolve(r.file))).convert("RGB")
            face = Image.open(resolve(r.face)).convert("RGB")  # the 224px face crop used locally
        except Exception:
            continue
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(out / "images" / f"{i}.jpg", quality=90)
        face.save(out / "faces" / f"{i}.jpg", quality=92)
        keep.append(i)
    meta = rows.loc[keep, ["group", "y", "role", "source", "main"]].assign(
        image=[f"images/{i}.jpg" for i in keep], face=[f"faces/{i}.jpg" for i in keep])
    meta.index.name = "row_id"
    meta.reset_index().to_csv(out / "meta.csv", index=False)
    rows.loc[keep].assign(row_id=keep).to_csv(PACK / "rows_local.csv", index=False)  # local copy with original paths
    zpath = PACK / "pom-data.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # JPEGs don't compress further
        for f in out.rglob("*"):
            z.write(f, f.relative_to(out))
    print(meta.groupby(["role", "source"]).size().unstack(fill_value=0).to_string())
    print(f"\n{len(meta)} photos -> {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


def blend():
    """Combine the Kaggle fine-tune with the production probe; blend weight chosen on the val people only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from .train import load_sources, split
    pred = pd.read_csv(PACK / "predictions.csv")            # row_id, logit (from the notebook)
    rows = pd.read_csv(PACK / "rows_local.csv").merge(pred, on="row_id")
    df, X = load_sources()
    test = split(df)
    df = df.assign(test=test, row=np.arange(len(df)))
    val_groups = set(rows.group[rows.role == "val"])
    fit = ~df.test & ~df.group.isin(val_groups)
    probe = LogisticRegression(max_iter=5000, C=0.3).fit(X[fit], df.y[fit])  # production probe, val people held out
    z_probe = pd.Series(probe.decision_function(X), index=df.file.values).groupby(level=0).mean()
    rows["z_probe"] = rows.file.map(z_probe)
    main = rows[rows.main & rows.z_probe.notna()]
    v, t = main[main.role == "val"], main[main.role == "test"]
    ws = np.linspace(0, 1, 21)
    zs = lambda d, w: w * d.logit + (1 - w) * d.z_probe
    w = max(ws, key=lambda w: roc_auc_score(v.y, zs(v, w)))
    report = {}
    for name, d in [("val", v), ("test", t)]:
        for label, z in [("probe", d.z_probe), ("finetune", d.logit), (f"blend w={w:.2f}", zs(d, w))]:
            report[f"{name} {label}"] = (round(roc_auc_score(d.y, z), 4), round(accuracy_score(d.y, z > 0), 4))
    for k, (auc, acc) in report.items():
        print(f"{k:24s} AUC {auc:.3f}  acc {acc:.1%}")
    json.dump({"blend_weight": w, **{k: list(v) for k, v in report.items()}}, open(OUTPUTS / "kaggle_blend.json", "w"), indent=2)
