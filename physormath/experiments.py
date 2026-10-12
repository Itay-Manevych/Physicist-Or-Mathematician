"""Experiment bench for the main physicist-vs-mathematician model.

Same fixed person-level split as `train` (outputs/split.csv). Every experiment is scored by 5-fold grouped CV on the
TRAIN part only; the test set is scored too but only used to confirm finalists. Results append to
outputs/experiments.csv.

  python -m physormath experiment classifiers   # classifier variants on the current features
  python -m physormath experiment features      # which feature views help
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import DATA, OUTPUTS
from .labels import is_strict

SOURCES = {"": "wiki", "_new": "new", "_extra": "extra"}


def _feat(name, tag, rows=None):
    path = DATA / f"feats_{name}{tag}.npy"
    if not path.exists():
        return None
    X = np.load(path)
    return X if rows is None else X[rows]


def load(feature_names, with_extra=True):
    """wiki (+ new + verified extra photos) with the requested feature views; None if a view is missing."""
    from .clean import photo_kind
    from .identity import verified
    parts = []
    w = pd.read_csv(DATA / "clean.csv")
    w = w[w.occupations.map(is_strict)]
    Xs = [_feat(f, "", w.row.values) for f in feature_names]
    if any(x is None for x in Xs):
        return None
    parts.append((w.assign(source="wiki", group=w.qid), np.hstack(Xs)))
    n = pd.read_csv(DATA / "faces_new.csv")
    ok = photo_kind("_new") == "photo"
    Xs = [_feat(f, "_new") for f in feature_names]
    if all(x is not None for x in Xs):
        parts.append((n[ok].assign(source="new", group=n.qid[ok]), np.hstack(Xs)[ok]))
    if with_extra:
        e = pd.read_csv(DATA / "faces_extra.csv")
        ok = verified("extra")
        Xs = [_feat(f, "_extra") for f in feature_names]
        if all(x is not None for x in Xs):
            parts.append((e[ok].assign(source="extra", group=e.qid[ok]), np.hstack(Xs)[ok]))
    cols = ["label", "source", "group", "birth_year", "gender", "face", "file"]
    df = pd.concat([p[0].reindex(columns=cols) for p in parts], ignore_index=True)
    df["y"] = (df.label == "physicist").astype(int)
    split = pd.read_csv(OUTPUTS / "split.csv").set_index("group").test
    df["test"] = df.group.map(split).fillna(False).astype(bool)
    return df, np.vstack([p[1] for p in parts])


def scores(model, X, df):
    """Grouped 5-fold CV on train (main photos scored) + one fit on train scored on test main photos."""
    y, train, main = df.y.values, ~df.test.values, df.source.isin(["wiki", "new"]).values
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    idx = np.where(train)[0]
    for tr, va in cv.split(idx, y[idx], df.group.values[idx]):
        m = clone(model).fit(X[idx[tr]], y[idx[tr]])
        oof[idx[va]] = _proba(m, X[idx[va]])
    m = clone(model).fit(X[train], y[train])
    p = _proba(m, X)
    ev, te = train & main, df.test.values & main
    return {"cv_auc": roc_auc_score(y[ev], oof[ev]), "test_auc": roc_auc_score(y[te], p[te]),
            "test_acc": accuracy_score(y[te], p[te] > (0.0 if _is_margin(m) else 0.5))}


def _is_margin(m):
    return not hasattr(m, "predict_proba")


def _proba(m, X):
    return m.decision_function(X) if _is_margin(m) else m.predict_proba(X)[:, 1]


def log(name, res, seconds):
    row = {"experiment": name, **{k: round(v, 4) for k, v in res.items()}, "seconds": round(seconds)}
    path = OUTPUTS / "experiments.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"{name:48s} CV AUC {res['cv_auc']:.3f} | test AUC {res['test_auc']:.3f} acc {res['test_acc']:.1%}  ({seconds:.0f}s)", flush=True)


def run(name, model, X, df):
    t = time.time()
    log(name, scores(model, X, df), time.time() - t)


BASE = ["clipL", "clipH", "siglip2", "clipH_full"]


def classifiers():
    df, X = load(BASE)
    print(f"{len(df)} photos, {X.shape[1]} features\n")
    for C in [0.3, 1, 3]:
        run(f"logreg C={C}", LogisticRegression(max_iter=5000, C=C), X, df)
    for C in [0.01, 0.03, 0.1]:
        run(f"standardize + logreg C={C}", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C)), X, df)
    for C in [1, 3, 10]:
        run(f"SVM rbf C={C}", SVC(C=C, kernel="rbf", gamma="scale"), X, df)
    for alpha in [1e-3, 1e-2]:
        run(f"MLP 512 alpha={alpha}", MLPClassifier((512,), alpha=alpha, early_stopping=True, max_iter=200, random_state=0), X, df)


def features():
    for extra in [[], ["clipH_upper"], ["facenet"], ["clipH_upper", "facenet"]]:
        loaded = load(BASE + extra)
        if loaded is None:
            print("missing features:", extra)
            continue
        df, X = loaded
        run(f"logreg C=1 on base + {extra or 'nothing'}", LogisticRegression(max_iter=5000, C=1), X, df)
    df, X = load(BASE)
    by = df.birth_year.fillna(df.birth_year.median()).values[:, None]
    meta = np.hstack([(by - 1950) / 30, (df.gender == "female").values[:, None].astype(float)])
    run("logreg C=1 on base + birth year + gender", LogisticRegression(max_iter=5000, C=1), np.hstack([X, meta * 3]), df)


CANDIDATES = ["dinov2L", "eva02L", "mc2H", "mc2H378", "mc2bigG"]


def backbones():
    """Each candidate vision transformer added to the current features (face, and face + whole photo)."""
    found = []
    for b in CANDIDATES:
        for views in ([b], [b, f"{b}_full"]):
            loaded = load(BASE + views)
            if loaded is None:
                print(f"{b}: features not computed yet")
                break
            df, X = loaded
            run(f"logreg C=1 on base + {'+'.join(views)}", LogisticRegression(max_iter=5000, C=1), X, df)
            if len(views) == 2:
                found += views
        loaded = load([b, f"{b}_full"])
        if loaded is not None:
            run(f"logreg C=1 on {b} face + whole only", LogisticRegression(max_iter=5000, C=1), loaded[1], loaded[0])
    if found:
        df, X = load(BASE + found)
        for C in [0.3, 1]:
            run(f"logreg C={C} on base + all candidates", LogisticRegression(max_iter=5000, C=C), X, df)


def extract(names=None):
    """Embed candidate backbones (face crops + whole photos) for every training source."""
    from .embed import embed_faces, embed_full
    for b in names or CANDIDATES:
        for csv, tag in [("data/faces.csv", ""), ("data/faces_new.csv", "_new"), ("data/faces_extra.csv", "_extra")]:
            embed_faces(csv, tag, [b])
            embed_full(csv, tag, [b])


class BlockStack:
    """One logistic regression per feature block (backbone/view), combined by a logistic-regression meta model
    trained on their out-of-fold probabilities."""

    def __init__(self, sizes, C=1.0, meta_C=1.0):
        self.sizes, self.C, self.meta_C = sizes, C, meta_C

    def get_params(self, deep=False):
        return {"sizes": self.sizes, "C": self.C, "meta_C": self.meta_C}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def _blocks(self, X):
        edges = np.cumsum([0] + list(self.sizes))
        return [X[:, a:b] for a, b in zip(edges[:-1], edges[1:])]

    def fit(self, X, y):
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        blocks = self._blocks(X)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        oof = np.column_stack([cross_val_predict(LogisticRegression(max_iter=5000, C=self.C), B, y, cv=cv,
                                                 method="predict_proba")[:, 1] for B in blocks])
        self.models_ = [LogisticRegression(max_iter=5000, C=self.C).fit(B, y) for B in blocks]
        self.meta_ = LogisticRegression(max_iter=5000, C=self.meta_C).fit(_logit(oof), y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        P = np.column_stack([m.predict_proba(B)[:, 1] for m, B in zip(self.models_, self._blocks(X))])
        return self.meta_.predict_proba(_logit(P))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ensembles(names=None):
    """Concatenation vs. per-backbone stacking, on the given (or all available) feature views."""
    names = names or [n for n in BASE + ["clipH_upper"] + sum(([b, f"{b}_full"] for b in CANDIDATES), [])
                      if (DATA / f"feats_{n}.npy").exists()]
    df, X = load(names)
    sizes = [np.load(DATA / f"feats_{n}.npy", mmap_mode="r").shape[1] for n in names]
    print(f"views: {names}\n")
    run(f"concat logreg C=1 ({len(names)} views)", LogisticRegression(max_iter=5000, C=1), X, df)
    for C in [0.3, 1]:
        run(f"stacked per-view logreg C={C} ({len(names)} views)", BlockStack(sizes, C=C), X, df)


def multiphoto(names=None, C=1.0):
    """Single photo vs. averaging over all of a test person's verified photos (main + extra [+ depicts])."""
    from .identity import verified
    names = names or BASE
    df, X = load(names)
    y, train = df.y.values, ~df.test.values
    m = LogisticRegression(max_iter=5000, C=C).fit(X[train], y[train])
    logit = m.decision_function(X)
    rows = pd.DataFrame({"group": df.group, "y": y, "test": df.test, "src": df.source, "z": logit})
    # identity-verified depicts photos, if their features exist
    if all((DATA / f"feats_{n}_depicts.npy").exists() for n in names + ["facenet"]):
        dep = pd.read_csv(DATA / "faces_depicts.csv")
        ok = verified("depicts")
        Xd = np.hstack([np.load(DATA / f"feats_{n}_depicts.npy") for n in names])[ok]
        split = pd.read_csv(OUTPUTS / "split.csv").set_index("group").test
        d = dep[ok]
        rows = pd.concat([rows, pd.DataFrame({"group": d.qid.values, "y": (d.label == "physicist").astype(int).values,
                                              "test": d.qid.map(split).fillna(False).values, "src": "depicts",
                                              "z": m.decision_function(Xd)})], ignore_index=True)
    t = rows[rows.test]
    main = t[t.src.isin(["wiki", "new"])].set_index("group")
    avg = t.groupby("group").z.mean()
    count = t.groupby("group").size()
    yy = main.y
    print(f"test people: {len(main)} | with 2+ photos: {(count.reindex(main.index) > 1).sum()} "
          f"(photos per person, mean {count.reindex(main.index).mean():.2f})")
    for label, idx in [("all test people", main.index), ("people with 2+ photos", main.index[count.reindex(main.index) > 1])]:
        s, a = main.z.reindex(idx), avg.reindex(idx)
        print(f"  {label:24s} single photo AUC {roc_auc_score(yy[idx], s):.3f} acc {((s > 0) == yy[idx]).mean():.1%}"
              f"  ->  averaged AUC {roc_auc_score(yy[idx], a):.3f} acc {((a > 0) == yy[idx]).mean():.1%}")


def prepare(csv, tag):
    """Face crops + only the views the production model needs (and FaceNet for identity checks)."""
    from .config import MODEL_FEATURES
    from .embed import embed_faces, embed_full
    from .faces import crop_faces
    faces_csv = f"data/faces{tag}.csv"
    if not (DATA / f"faces{tag}.csv").exists():
        crop_faces(csv, faces_csv)
    face_views = ["facenet"] + [f for f in MODEL_FEATURES if not f.endswith("_full")]
    embed_faces(faces_csv, tag, face_views)
    embed_full(faces_csv, tag, [f[:-5] for f in MODEL_FEATURES if f.endswith("_full")])
