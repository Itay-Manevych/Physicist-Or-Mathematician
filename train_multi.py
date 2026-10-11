"""Compare backbones (and their combination) on the physicist-vs-mathematician task.

Same person-disjoint split as train.py. Reports test AUC, 5-fold CV AUC, and how much each model
adds on top of a photo-style + metadata baseline. Saves the best combination as model_multi.joblib.
"""
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

NAMES = ["facenet", "clipL", "clipH", "siglip2"]
Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]

df = pd.read_csv("data/faces.csv")
F = {n: np.load(f"data/feats_{n}.npy") for n in NAMES}
mask = df.label.isin(["physicist", "mathematician"]).values
both_df = df[~mask].reset_index(drop=True)
both_F = {n: f[~mask] for n, f in F.items()}
df = df[mask].reset_index(drop=True)
F = {n: f[mask] for n, f in F.items()}
y = (df.label == "physicist").astype(int).values
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def pick_C(X):
    best = max(Cs, key=lambda C: roc_auc_score(
        y[idx_tr], cross_val_predict(LogisticRegression(max_iter=5000, C=C), X[idx_tr], y[idx_tr], cv=cv,
                                     method="predict_proba", n_jobs=-1)[:, 1]))
    return best


def style_feats():
    rows = []
    for r in df.itertuples():
        im = Image.open(r.file)
        w, h = im.size
        a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=float) / 255
        f = np.asarray(Image.open(r.face).convert("RGB"), dtype=float) / 255
        rows.append([w, h, w / h, (a.max(2) - a.min(2)).mean(), a.mean(), a.std(),
                     (f.max(2) - f.min(2)).mean(), f.mean(), f.std(), *a.mean((0, 1)),
                     r.birth_year, r.gender == "female"])
    return np.array(rows, dtype=float)


S = style_feats()
p_base = cross_val_predict(HistGradientBoostingClassifier(max_depth=4), S, y, cv=cv, method="predict_proba")[:, 1]
results = {"style+metadata baseline (cv auc)": round(roc_auc_score(y, p_base), 3)}

configs = {n: [n] for n in NAMES}
configs["clip_all"] = ["clipL", "clipH", "siglip2"]
configs["everything"] = NAMES
oof = {}
for cname, parts in configs.items():
    X = np.hstack([F[p] for p in parts])
    C = pick_C(X)
    clf = LogisticRegression(max_iter=5000, C=C).fit(X[idx_tr], y[idx_tr])
    p = clf.predict_proba(X[idx_te])[:, 1]
    oof[cname] = cross_val_predict(LogisticRegression(max_iter=5000, C=C), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    p_comb = cross_val_predict(HistGradientBoostingClassifier(max_depth=4), np.c_[S, oof[cname]], y, cv=cv,
                               method="predict_proba")[:, 1]
    results[cname] = {
        "C": C,
        "test_auc": round(roc_auc_score(y[idx_te], p), 3),
        "test_acc": round(accuracy_score(y[idx_te], p > 0.5), 3),
        "cv_auc": round(roc_auc_score(y, oof[cname]), 3),
        "baseline+this (cv auc)": round(roc_auc_score(y, p_comb), 3),
    }
    print(cname, results[cname], flush=True)

best = max([c for c in configs if c != "facenet"], key=lambda c: results[c]["cv_auc"])
results["best"] = best
X = np.hstack([F[p] for p in configs[best]])
clf = LogisticRegression(max_iter=5000, C=results[best]["C"]).fit(X, y)
joblib.dump({"parts": configs[best], "clf": clf}, "model_multi.joblib")
both_df["p_physicist"] = clf.predict_proba(np.hstack([both_F[p] for p in configs[best]]))[:, 1]
both_df.sort_values("p_physicist").to_csv("both_scores_multi.csv", index=False)
df.assign(p_oof=oof[best], y=y).sort_values("p_oof").to_csv("oof_predictions_multi.csv", index=False)
json.dump(results, open("results_multi.json", "w"), indent=2)
print(json.dumps(results, indent=2))
