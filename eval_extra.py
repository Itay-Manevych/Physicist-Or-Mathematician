"""Does having several photos per person help?

1. Verify extra photos show the same person (FaceNet cosine similarity to the main portrait).
2. Person-level 5-fold CV (all photos of a person in the same fold):
   A) train on main photos, predict from the main photo only
   B) train on main photos, average predictions over all of the person's photos
   C) train on main + extra photos, average predictions over all photos
Evaluated on people who have at least one verified extra photo (so A/B/C use the same people).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from labels import load_faces

PARTS = ["clipL", "clipH", "siglip2"]
SAME_PERSON = 0.55

main = load_faces()
keep = main.label.isin(["physicist", "mathematician"]).values & main.strict.values
Fm = {p: np.load(f"data/feats_{p}.npy")[keep] for p in PARTS + ["facenet"]}
main = main[keep].reset_index(drop=True)
extra = pd.read_csv("data/faces_extra.csv")
Fe = {p: np.load(f"data/feats_{p}_extra.npy") for p in PARTS + ["facenet"]}

# identity check
ref = dict(zip(main.qid, Fm["facenet"]))
sim = np.array([ref[q] @ f if q in ref else -1 for q, f in zip(extra.qid, Fe["facenet"])])
ok = sim > SAME_PERSON
print(f"extra photos: {len(extra)}, same person verified: {ok.sum()} ({ok.mean():.0%})")
extra, Fe = extra[ok].reset_index(drop=True), {p: f[ok] for p, f in Fe.items()}

Xm = np.hstack([Fm[p] for p in PARTS])
Xe = np.hstack([Fe[p] for p in PARTS])
ym = (main.label == "physicist").astype(int).values
lab = dict(zip(main.qid, ym))
ye = extra.qid.map(lab).values
has_extra = main.qid.isin(set(extra.qid)).values
print(f"people with >=1 verified extra photo: {has_extra.sum()}  (avg {len(extra) / max(has_extra.sum(), 1):.1f} extras each)")

pA, pB, pC = np.zeros(len(main)), np.zeros(len(main)), np.zeros(len(main))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xm, ym):
    tr_q, te_q = set(main.qid[tr]), set(main.qid[te])
    e_tr, e_te = extra.qid.isin(tr_q).values, extra.qid.isin(te_q).values
    clf = LogisticRegression(max_iter=5000, C=3).fit(Xm[tr], ym[tr])
    clf2 = LogisticRegression(max_iter=5000, C=3).fit(np.vstack([Xm[tr], Xe[e_tr]]), np.r_[ym[tr], ye[e_tr]])
    for model, out, use_extra in [(clf, pA, False), (clf, pB, True), (clf2, pC, True)]:
        p_main = model.predict_proba(Xm[te])[:, 1]
        if use_extra:
            pe = pd.Series(model.predict_proba(Xe[e_te])[:, 1], index=extra.qid[e_te].values)
            pe = pe.groupby(level=0).agg(["sum", "count"])
            s = pd.Series(p_main, index=main.qid[te].values)
            p_main = ((s + pe["sum"].reindex(s.index).fillna(0)) / (1 + pe["count"].reindex(s.index).fillna(0))).values
        out[te] = p_main

m = has_extra
for name, p in [("A) main photo only", pA), ("B) average over all photos", pB), ("C) + train on extras, average", pC)]:
    print(f"{name:34s} AUC {roc_auc_score(ym[m], p[m]):.3f}  acc {((p[m] > .5) == ym[m]).mean():.1%}   "
          f"| all strict people: AUC {roc_auc_score(ym, p):.3f}")
