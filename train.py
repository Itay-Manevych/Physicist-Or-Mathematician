"""Train a physicist-vs-mathematician classifier on CLIP face embeddings, with confound checks."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv("data/faces.csv")
X = np.load("data/clip_feats.npy")
mask = df.label.isin(["physicist", "mathematician"]).values
both_df, both_X = df[~mask].reset_index(drop=True), X[~mask]
df, X = df[mask].reset_index(drop=True), X[mask]
y = (df.label == "physicist").astype(int).values
print("class counts:", df.label.value_counts().to_dict())

idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
results = {}

# --- main model: logistic regression probe on CLIP face embeddings
grid = GridSearchCV(
    LogisticRegression(max_iter=5000, class_weight="balanced"),
    {"C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X[idx_tr], y[idx_tr])
clf = grid.best_estimator_
p = clf.predict_proba(X[idx_te])[:, 1]
results["face_clip"] = {
    "C": grid.best_params_["C"],
    "auc": roc_auc_score(y[idx_te], p),
    "acc": accuracy_score(y[idx_te], p > 0.5),
    "balanced_acc": balanced_accuracy_score(y[idx_te], p > 0.5),
}

# --- confound baselines: how much can we predict WITHOUT looking at the face?
meta = pd.DataFrame({
    "birth_year": df.birth_year,
    "female": (df.gender == "female").astype(int),
})
gb = HistGradientBoostingClassifier(max_depth=3).fit(meta.iloc[idx_tr], y[idx_tr])
pm = gb.predict_proba(meta.iloc[idx_te])[:, 1]
results["metadata_only(birth_year+gender)"] = {"auc": roc_auc_score(y[idx_te], pm)}

# is the face model just reading age/era? check AUC within birth-year bands
te = df.iloc[idx_te].assign(p=p, y=y[idx_te])
bands = {}
for lo, hi in [(1900, 1940), (1940, 1960), (1960, 2010)]:
    s = te[(te.birth_year >= lo) & (te.birth_year < hi)]
    if s.y.nunique() == 2 and len(s) > 30:
        bands[f"{lo}-{hi}"] = {"n": len(s), "auc": roc_auc_score(s.y, s.p)}
results["face_clip_auc_by_birth_band"] = bands

# --- refit on everything and save
clf_full = LogisticRegression(max_iter=5000, class_weight="balanced", C=grid.best_params_["C"]).fit(X, y)
joblib.dump(clf_full, "model.joblib")

# --- fun: scores for the people Wikidata says are BOTH, and the extremes of the test set
both_df["p_physicist"] = clf_full.predict_proba(both_X)[:, 1]
both_df.sort_values("p_physicist").to_csv("both_scores.csv", index=False)
te.sort_values("p").to_csv("test_predictions.csv", index=False)

print(json.dumps(results, indent=2))
json.dump(results, open("results.json", "w"), indent=2)
