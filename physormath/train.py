"""Train and evaluate with a fixed, leak-free train/test split.

Sources (all optional except wiki):
  wiki   cleaned Wikidata portraits (data/clean.csv)              group = person
  new    Wikidata people from physics/maths sub-fields             group = person
  extra  more Commons photos of wiki people (identity-verified)    group = that person
  uni    university + Simons headshots, balanced per university    group = university

Split: 20% of groups are held out as the TEST set (stratified by label). A test person's extra photos and a
test university's photos never appear in training. Regularisation is chosen by 5-fold grouped CV on TRAIN only;
the test set is scored once. The saved model is trained on TRAIN only, so the game can use test people.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from .clean import photo_kind
from .config import DATA, MODEL_FEATURES, OUTPUTS
from .labels import is_strict

SAME_PERSON = 0.55  # FaceNet cosine similarity for "same person"
CS = [0.03, 0.1, 0.3, 1, 3]
CV_TOLERANCE = 0.005  # prefer the simplest model within this much CV AUC of the best


def features(tag, rows=None):
    X = np.hstack([np.load(DATA / f"feats_{p}{tag}.npy") for p in MODEL_FEATURES])
    return X if rows is None else X[rows]


def load_sources():
    parts = []
    w = pd.read_csv(DATA / "clean.csv")
    w = w[w.occupations.map(is_strict)]
    parts.append((w.assign(source="wiki", group=w.qid), features("", w.row.values)))

    if (DATA / "feats_clipH_full_new.npy").exists():
        n = pd.read_csv(DATA / "faces_new.csv")
        ok = photo_kind("_new") == "photo"
        parts.append((n[ok].assign(source="new", group=n.qid[ok]), features("_new")[ok]))

    if (DATA / "feats_clipH_full_extra.npy").exists():
        e = pd.read_csv(DATA / "faces_extra.csv")
        ref = dict(zip(w.qid, np.load(DATA / "feats_facenet.npy")[w.row.values]))
        sim = np.array([ref[q] @ f if q in ref else -1 for q, f in zip(e.qid, np.load(DATA / "feats_facenet_extra.npy"))])
        ok = (sim > SAME_PERSON) & (photo_kind("_extra") == "photo")
        parts.append((e[ok].assign(source="extra", group=e.qid[ok]), features("_extra")[ok]))

    if (DATA / "feats_clipH_full_uni.npy").exists():
        u = pd.read_csv(DATA / "faces_uni.csv")
        ok = photo_kind("_uni") == "photo"
        u, X = u[ok].reset_index(drop=True), features("_uni")[ok]
        rng, idx = np.random.default_rng(0), []
        for _, g in u.groupby("university"):  # equal physicists/mathematicians per university; skip one-sided ones
            counts = g.label.value_counts()
            if len(counts) == 2:
                for lab in counts.index:
                    idx += list(rng.choice(g.index[g.label == lab], counts.min(), replace=False))
        idx = np.array(sorted(idx), dtype=int)
        parts.append((u.loc[idx].assign(source="uni", group="uni:" + u.loc[idx].university), X[idx]))

    cols = ["label", "source", "group", "name", "qid", "university", "face", "file", "img"]
    df = pd.concat([p[0][[c for c in cols if c in p[0]]] for p in parts], ignore_index=True)
    df["y"] = (df.label == "physicist").astype(int)
    return df, np.vstack([p[1] for p in parts])


def split(df):
    """Group-level 80/20 split, stratified by label; saved so every run uses the same test set."""
    path = OUTPUTS / "split.csv"
    groups = df.groupby("group").y.first()
    if path.exists():
        saved = pd.read_csv(path).set_index("group").test
        new = groups.index.difference(saved.index)  # groups added since the split was made
        if len(new):
            extra = _assign(groups[new])
            saved = pd.concat([saved, extra])
            saved.rename("test").reset_index().to_csv(path, index=False)
        test = saved
    else:
        test = _assign(groups)
        test.rename("test").reset_index().to_csv(path, index=False)
    return df.group.map(test).astype(bool).values


def _assign(groups):
    fold = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(groups.index, groups.values, groups.index))[1]
    return pd.Series(np.isin(np.arange(len(groups)), fold), index=groups.index, name="test")


def metrics(y, p):
    return {"auc": round(roc_auc_score(y, p), 3), "acc": round(accuracy_score(y, p > 0.5), 3), "n": int(len(y))}


def main():
    OUTPUTS.mkdir(exist_ok=True)
    df, X = load_sources()
    y, test = df.y.values, split(df)
    train = ~test
    print("photos per source (train / test):")
    print(df.assign(set=np.where(test, "test", "train")).groupby(["source", "set"]).size().unstack().to_string(), "\n")

    # choose C by grouped CV on TRAIN only
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    cv_auc = {}
    for C in CS:
        p = cross_val_predict(LogisticRegression(max_iter=5000, C=C), X[train], y[train], groups=df.group[train],
                              cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        cv_auc[C] = roc_auc_score(y[train], p)
    best = max(cv_auc.values())
    C = min(c for c in CS if cv_auc[c] >= best - CV_TOLERANCE)  # strongest regularisation that is ~as good
    model = LogisticRegression(max_iter=5000, C=C).fit(X[train], y[train])
    p = model.predict_proba(X)[:, 1]
    df["p"] = p

    # main evaluation: one photo per test person (their Wikidata portrait), then per-source breakdowns
    main_photo = df.source.isin(["wiki", "new"]).values
    rep = {"C": C, "cv_auc_by_C": {str(k): round(v, 3) for k, v in cv_auc.items()},
           "train": metrics(y[train & main_photo], p[train & main_photo]),
           "test": metrics(y[test & main_photo], p[test & main_photo])}
    print(f"C = {C} (grouped 5-fold CV on train: {rep['cv_auc_by_C']})")
    print(f"TRAIN  AUC {rep['train']['auc']:.3f}  acc {rep['train']['acc']:.1%}   (n={rep['train']['n']})")
    print(f"TEST   AUC {rep['test']['auc']:.3f}  acc {rep['test']['acc']:.1%}   (n={rep['test']['n']}, never seen in training)")
    gap = rep["train"]["acc"] - rep["test"]["acc"]
    level = "none" if gap < 0.02 else "mild" if gap < 0.05 else "strong"
    rep["overfitting"] = {"train_test_acc_gap": round(gap, 3), "level": level,
                          "cv_auc": round(cv_auc[C], 3), "test_auc": rep["test"]["auc"]}
    print(f"overfitting check: train-test accuracy gap {gap:+.1%} ({level}); "
          f"CV AUC on train {cv_auc[C]:.3f} vs TEST AUC {rep['test']['auc']:.3f}")

    # confidence: accuracy on the most confident test people; thresholds saved for predict.py
    t = test & main_photo
    margin = np.abs(p[t] - 0.5)
    rep["confident"] = {}
    for cov in [0.5, 0.25, 0.1]:
        k = margin >= np.quantile(margin, 1 - cov)
        rep["confident"][f"top_{int(cov * 100)}%"] = round(accuracy_score(y[t][k], p[t][k] > 0.5), 3)
    print("test accuracy on the most confident:", rep["confident"])

    # several photos per person: average over the main + extra photos of test people who have extras
    ext = (df.source == "extra").values
    if ext.any():
        with_extra = set(df.group[test & ext])
        tm = t & df.group.isin(with_extra).values
        avg = df[test & (main_photo | ext)].groupby("group").p.mean()
        rep["multi_photo"] = {"single": metrics(y[tm], p[tm]),
                              "averaged": metrics(y[tm], avg.reindex(df.group[tm]).values)}
        print(f"test people with several photos (n={tm.sum()}): one photo AUC {rep['multi_photo']['single']['auc']:.3f}"
              f" acc {rep['multi_photo']['single']['acc']:.1%} -> averaged AUC {rep['multi_photo']['averaged']['auc']:.3f}"
              f" acc {rep['multi_photo']['averaged']['acc']:.1%}")

    # held-out universities: staff headshots from universities never seen in training
    uni = (df.source == "uni").values
    if (uni & test).any():
        rep["held_out_universities"] = metrics(y[uni & test], p[uni & test])
        print(f"held-out universities ({df.university[uni & test].nunique()}): AUC "
              f"{rep['held_out_universities']['auc']:.3f}  acc {rep['held_out_universities']['acc']:.1%}")

    # learning curve on the fixed test set: does more training data still help?
    rng, rep["learning_curve"] = np.random.default_rng(0), {}
    tr_groups = df.group[train].unique()
    for frac in [0.1, 0.25, 0.5, 1.0]:
        g = set(rng.choice(tr_groups, int(len(tr_groups) * frac), replace=False))
        sub = train & df.group.isin(g).values
        pp = LogisticRegression(max_iter=5000, C=C).fit(X[sub], y[sub]).predict_proba(X[t])[:, 1]
        rep["learning_curve"][f"{frac:.0%}"] = round(roc_auc_score(y[t], pp), 3)
    print("learning curve (test AUC vs share of training data):", rep["learning_curve"])

    q50, q75 = np.quantile(margin, [0.5, 0.75])
    joblib.dump({"features": MODEL_FEATURES, "clf": model, "margins": {"leaning": q50, "confident": q75}},
                OUTPUTS / "model.joblib")
    df[test & main_photo].drop(columns=["y"]).to_csv(OUTPUTS / "test_predictions.csv", index=False)
    json.dump(rep, open(OUTPUTS / "report.json", "w"), indent=2)
    print("\nsaved outputs/model.joblib, outputs/report.json, outputs/test_predictions.csv")
