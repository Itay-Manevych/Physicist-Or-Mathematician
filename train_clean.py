"""Train + evaluate on the cleaned dataset (data/clean.csv: strict physicists/mathematicians, real photos only).

Reports 5-fold CV and held-out test metrics for several feature sets, confidence-filtered accuracy,
and saves the best model + out-of-fold predictions (used by the game).
"""
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/clean.csv")
rows = df.row.values
F = {n: np.load(f"data/feats_{n}.npy")[rows] for n in ["clipL", "clipH", "siglip2", "clipH_upper", "clipH_full", "facenet"]}
y = (df.label == "physicist").astype(int).values
cv = StratifiedKFold(5, shuffle=True, random_state=0)
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)

configs = {
    "face shape only (FaceNet)": ["facenet"],
    "face (3 CLIP models)": ["clipL", "clipH", "siglip2"],
    "upper body": ["clipH_upper"],
    "whole photo": ["clipH_full"],
    "face + whole photo": ["clipL", "clipH", "siglip2", "clipH_full"],
    "face + upper + whole photo": ["clipL", "clipH", "siglip2", "clipH_upper", "clipH_full"],
}


def fit_eval(X, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    best = None
    for C in [0.3, 1, 3, 10]:
        p = cross_val_predict(LogisticRegression(max_iter=5000, C=C), X[m], y[m], cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        auc = roc_auc_score(y[m], p)
        if best is None or auc > best[0]:
            best = (auc, C, p)
    return best


results, oof = {}, {}
print(f"clean dataset: {len(df)} people ({(y == 0).sum()} mathematicians, {y.sum()} physicists)\n")
for name, parts in configs.items():
    X = np.hstack([F[p] for p in parts])
    auc, C, p = fit_eval(X)
    tr_te = LogisticRegression(max_iter=5000, C=C).fit(X[idx_tr], y[idx_tr]).predict_proba(X[idx_te])[:, 1]
    results[name] = {"cv_auc": round(auc, 3), "cv_acc": round(((p > .5) == y).mean(), 3),
                     "test_auc": round(roc_auc_score(y[idx_te], tr_te), 3), "test_acc": round(((tr_te > .5) == y[idx_te]).mean(), 3), "C": C}
    oof[name] = p
    print(f"{name:28s} CV AUC {auc:.3f}  acc {((p > .5) == y).mean():.1%}   | held-out test AUC {results[name]['test_auc']:.3f}  acc {results[name]['test_acc']:.1%}", flush=True)

best = max([k for k in configs if "FaceNet" not in k], key=lambda k: results[k]["cv_auc"])
p = oof[best]
print(f"\nbest: {best}")

# confidence-filtered accuracy
conf = np.abs(p - .5)
sel = {}
for cov in [1.0, 0.5, 0.25, 0.1]:
    k = conf >= np.quantile(conf, 1 - cov)
    sel[f"top {cov:.0%}"] = round(((p[k] > .5) == y[k]).mean(), 3)
print("accuracy when only answering the most confident:", sel)

# confound check: photo style + metadata, and what the model adds on top
def style(r):
    im = Image.open(r.file); w, h = im.size
    a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=float) / 255
    return [w, h, w / h, (a.max(2) - a.min(2)).mean(), a.mean(), a.std(), *a.mean((0, 1)), r.birth_year, r.gender == "female"]
S = np.array([style(r) for r in df.itertuples()], dtype=float)
p_s = cross_val_predict(HistGradientBoostingClassifier(max_depth=4), S, y, cv=cv, method="predict_proba")[:, 1]
p_sm = cross_val_predict(HistGradientBoostingClassifier(max_depth=4), np.c_[S, p], y, cv=cv, method="predict_proba")[:, 1]
print(f"photo style + birth year + gender only: AUC {roc_auc_score(y, p_s):.3f}  -> plus model: {roc_auc_score(y, p_sm):.3f}")
results.update({"best": best, "selective_accuracy": sel, "style_meta_auc": round(roc_auc_score(y, p_s), 3)})

# academic subset (doctoral advisor or university employer on Wikidata)
acad = df.academic.astype(bool).values
a_auc, _, a_p = fit_eval(np.hstack([F[q] for q in configs[best]]), acad)
print(f"academic-affiliation subset (n={acad.sum()}): CV AUC {a_auc:.3f}  acc {((a_p > .5) == y[acad]).mean():.1%}")
results["academic_subset"] = {"n": int(acad.sum()), "cv_auc": round(a_auc, 3)}

X = np.hstack([F[q] for q in configs[best]])
joblib.dump({"parts": configs[best], "clf": LogisticRegression(max_iter=5000, C=results[best]["C"]).fit(X, y)}, "model_clean.joblib")
df.assign(p_oof=p).to_csv("oof_predictions_clean.csv", index=False)
json.dump(results, open("results_clean.json", "w"), indent=2)
