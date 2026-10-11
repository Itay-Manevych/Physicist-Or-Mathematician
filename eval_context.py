"""Does seeing more than the face help? Compare face / upper-body / full-photo features, on all people
and on the clean academic subset. 5-fold CV AUC with logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from labels import load_faces

df = load_faces()
F = {n: np.load(f"data/feats_{n}.npy") for n in ["clipL", "clipH", "siglip2", "clipH_upper", "clipH_full"]}
keep = df.label.isin(["physicist", "mathematician"]).values
y_all = (df.label == "physicist").astype(int).values
cv = StratifiedKFold(5, shuffle=True, random_state=0)
configs = {
    "face (clipH)": ["clipH"],
    "upper body (clipH)": ["clipH_upper"],
    "full photo (clipH)": ["clipH_full"],
    "face x3 backbones": ["clipL", "clipH", "siglip2"],
    "face + upper + full": ["clipH", "clipH_upper", "clipH_full"],
    "face x3 + upper + full": ["clipL", "clipH", "siglip2", "clipH_upper", "clipH_full"],
}
for subset_name, mask in [("ALL", keep), ("STRICT", keep & df.strict.values), ("STRICT ACADEMIC", keep & df.academic.values)]:
    y = y_all[mask]
    print(f"== {subset_name} (n={mask.sum()}) ==")
    for name, parts in configs.items():
        X = np.hstack([F[p][mask] for p in parts])
        best = 0
        for C in [0.3, 1, 3, 10]:
            p = cross_val_predict(LogisticRegression(max_iter=5000, C=C), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            auc = roc_auc_score(y, p)
            if auc > best:
                best, acc, bestC = auc, ((p > 0.5) == y).mean(), C
        print(f"  {name:26s} AUC {best:.3f}  acc {acc:.1%}  (C={bestC})", flush=True)
