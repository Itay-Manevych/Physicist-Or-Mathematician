"""Combine every data source and measure what each one adds.

Sources
  wiki   - cleaned Wikidata portraits (data/clean.csv)                     group = person
  new    - extra Wikidata people from physics/maths sub-fields               group = person
  extra  - more Commons photos of wiki people (identity-verified)           group = same person as wiki
  uni    - university department + Simons headshots (balanced per uni)      group = university

Evaluation
  A) Wikidata people, 5-fold CV by person: train on wiki only vs. wiki + all other sources
  B) Leave-one-university-out on uni photos: train on everything except that university
  C) Wikidata people with extra photos: single photo vs. averaged over all their photos
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from labels import is_strict
from clean_dataset import classify, PROMPTS, FACE_PROMPTS

PARTS = ["clipL", "clipH", "siglip2", "clipH_full"]
SAME_PERSON = 0.55
C = 1


def feats(tag, rows=None):
    X = np.hstack([np.load(f"data/feats_{p}{tag}.npy") for p in PARTS])
    return X if rows is None else X[rows]


def photo_ok(tag):
    """True where CLIP thinks the image (whole + face crop) is a real photograph."""
    full = classify(np.load(f"data/feats_clipH_full{tag}.npy"))
    face = classify(np.load(f"data/feats_clipH{tag}.npy"), FACE_PROMPTS)
    bad_full = (full.idxmax(1) != "photo").values & (full["photo"].values <= 0.4)
    bad_face = (face.idxmax(1) != "photo").values & (face["photo"].values < 0.4)
    return ~(bad_full | bad_face)


def load():
    parts = []
    w = pd.read_csv("data/clean.csv")
    keep = w.occupations.map(is_strict).values
    print(f"wiki: {keep.sum()} (dropped {(~keep).sum()} bio/geo-physicists etc. after tightening)")
    parts.append((w[keep].assign(source="wiki", group=w.qid[keep]), feats("", w.row.values[keep])))

    if os.path.exists("data/feats_clipH_full_new.npy"):
        n = pd.read_csv("data/faces_new.csv")
        ok = photo_ok("_new")
        print(f"new: {ok.sum()} of {len(n)} are real photos")
        parts.append((n[ok].assign(source="new", group=n.qid[ok]), feats("_new")[ok]))

    if os.path.exists("data/feats_clipH_full_extra.npy"):
        e = pd.read_csv("data/faces_extra.csv")
        ref = dict(zip(w.qid, np.load("data/feats_facenet.npy")[w.row.values]))
        fn = np.load("data/feats_facenet_extra.npy")
        same = np.array([ref[q] @ f if q in ref else -1 for q, f in zip(e.qid, fn)]) > SAME_PERSON
        ok = same & photo_ok("_extra")
        print(f"extra: {ok.sum()} of {len(e)} verified same person + real photo")
        parts.append((e[ok].assign(source="extra", group=e.qid[ok]), feats("_extra")[ok]))

    if os.path.exists("data/feats_clipH_full_uni.npy"):
        u = pd.read_csv("data/faces_uni.csv")
        ok = photo_ok("_uni")
        u, X = u[ok].reset_index(drop=True), feats("_uni")[ok]
        # balance: same number of physicists and mathematicians per university; drop one-department universities
        rng = np.random.default_rng(0)
        idx = []
        for uni, g in u.groupby("university"):
            counts = g.label.value_counts()
            if len(counts) < 2:
                continue
            n_min = counts.min()
            for lab in counts.index:
                idx += list(rng.choice(g.index[g.label == lab], n_min, replace=False))
        idx = np.array(sorted(idx))
        print(f"uni: {len(idx)} balanced photos from {u.loc[idx].university.nunique()} universities "
              f"(of {len(u)} real photos)")
        parts.append((u.loc[idx].assign(source="uni", group="uni:" + u.loc[idx].university), X[idx]))

    df = pd.concat([p[0][["label", "source", "group"] + [c for c in ["name", "qid", "university", "face", "img"] if c in p[0]]]
                    for p in parts], ignore_index=True)
    X = np.vstack([p[1] for p in parts])
    df["y"] = (df.label == "physicist").astype(int)
    return df, X


def main():
    df, X = load()
    y = df.y.values
    print("\ntotal:", len(df), df.groupby(["source", "label"]).size().unstack().to_string(), "\n")
    res = {}

    # A) Wikidata people: CV by person
    wiki = (df.source == "wiki").values
    people = df.group[wiki].unique()
    pl = df[wiki].groupby("group").y.first().reindex(people).values
    pA_only, pA_all = np.zeros(wiki.sum()), np.zeros(wiki.sum())
    wiki_groups = df.group[wiki].values
    for tr_p, te_p in StratifiedKFold(5, shuffle=True, random_state=0).split(people, pl):
        te_groups = set(people[te_p])
        test = wiki & df.group.isin(te_groups).values
        train_wiki = wiki & ~df.group.isin(te_groups).values
        train_all = ~df.group.isin(te_groups).values  # also excludes the test people's extra photos
        m1 = LogisticRegression(max_iter=5000, C=C).fit(X[train_wiki], y[train_wiki])
        m2 = LogisticRegression(max_iter=5000, C=C).fit(X[train_all], y[train_all])
        sel = np.isin(wiki_groups, list(te_groups))
        pA_only[sel] = m1.predict_proba(X[test])[:, 1]
        pA_all[sel] = m2.predict_proba(X[test])[:, 1]
    yw = y[wiki]
    for name, p in [("wiki only", pA_only), ("wiki + all sources", pA_all)]:
        print(f"A) Wikidata people, trained on {name:20s} AUC {roc_auc_score(yw, p):.3f}  acc {((p > .5) == yw).mean():.1%}")
        res[f"A_{name}"] = round(roc_auc_score(yw, p), 3)

    # B) leave-one-university-out
    if (df.source == "uni").any():
        rows = []
        for g in sorted(df.group[df.source == "uni"].unique()):
            test = (df.group == g).values
            for name, train in [("wiki+new+extra", (df.source != "uni").values),
                                ("everything else", ~test)]:
                m = LogisticRegression(max_iter=5000, C=C).fit(X[train], y[train])
                p = m.predict_proba(X[test])[:, 1]
                rows.append({"university": g[4:], "train": name, "n": test.sum(),
                             "auc": roc_auc_score(y[test], p), "acc": ((p > .5) == y[test]).mean()})
        r = pd.DataFrame(rows)
        print("\nB) leave-one-university-out (test = staff headshots of a university never seen in training)")
        print(r.pivot(index="university", columns="train", values="auc").round(3).assign(n=r.groupby("university").n.first()).to_string())
        for name, g in r.groupby("train"):
            w_auc = np.average(g.auc, weights=g.n)
            w_acc = np.average(g.acc, weights=g.n)
            print(f"   weighted mean, trained on {name:16s}: AUC {w_auc:.3f}  acc {w_acc:.1%}")
            res[f"B_{name}"] = round(w_auc, 3)

    # C) averaging over several photos of the same person (uses the A) wiki+all models' fold structure)
    if (df.source == "extra").any():
        ext = (df.source == "extra").values
        has = df.group[wiki].isin(set(df.group[ext])).values
        people = df.group[wiki].unique()
        p_single, p_avg = np.zeros(wiki.sum()), np.zeros(wiki.sum())
        for tr_p, te_p in StratifiedKFold(5, shuffle=True, random_state=0).split(people, pl):
            te_groups = set(people[te_p])
            train = ~df.group.isin(te_groups).values
            m = LogisticRegression(max_iter=5000, C=C).fit(X[train], y[train])
            test_all = df.group.isin(te_groups).values & (wiki | ext)
            p_all = pd.Series(m.predict_proba(X[test_all])[:, 1], index=df.group[test_all].values)
            avg = p_all.groupby(level=0).mean()
            sel = np.isin(wiki_groups, list(te_groups))
            p_single[sel] = m.predict_proba(X[wiki & df.group.isin(te_groups).values])[:, 1]
            p_avg[sel] = avg.reindex(wiki_groups[sel]).values
        print(f"\nC) {has.sum()} Wikidata people with extra photos:")
        for name, p in [("main photo only", p_single), ("averaged over all photos", p_avg)]:
            print(f"   {name:26s} AUC {roc_auc_score(yw[has], p[has]):.3f}  acc {((p[has] > .5) == yw[has]).mean():.1%}")
            res[f"C_{name}"] = round(roc_auc_score(yw[has], p[has]), 3)

    final = LogisticRegression(max_iter=5000, C=C).fit(X, y)
    joblib.dump({"parts": PARTS, "clf": final}, "model_all.joblib")
    df[wiki].assign(p_oof=pA_all).to_csv("oof_predictions_all.csv", index=False)
    json.dump(res, open("results_all.json", "w"), indent=2)


if __name__ == "__main__":
    main()
