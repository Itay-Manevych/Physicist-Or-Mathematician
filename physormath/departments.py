"""Guess the department: a separate 7-way mode (its own data, model and game files; the main game is untouched).

  python -m physormath dept fetch    # Wikidata chemists, biologists, CS, philosophers, economists -> data/dept/
  python -m physormath process data/dept/people.csv _dept
  python -m physormath dept train    # 7-way classifier, fixed person-level split -> outputs/dept/
  python -m physormath dept game     # game/dept/data.json + portraits (held-out test people only)

Physicists and mathematicians come from the main clean set and keep their main train/test assignment.
A person counts only if every Wikidata occupation belongs to ONE department (plus generic academic titles).
"""
import json
import random
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from .config import DATA, MODEL_FEATURES, OUTPUTS, ROOT

DEPT = DATA / "dept"
OUT = OUTPUTS / "dept"
GAME = ROOT / "game" / "dept"
CAP = 2500  # people per new department

GENERIC = (r"university teacher|professor|academic|researcher|scientist|lecturer|docent|research fellow|teacher|educator|"
           r"writer|author|non-fiction writer|essayist|professeur des universit.s")
DEPARTMENTS = {
    "physics": {"label": "PHYSICS"},
    "math": {"label": "MATHEMATICS"},
    "chem": {"label": "CHEMISTRY", "qids": ["Q593644", "Q16744668", "Q34463094", "Q85519878", "Q71116296", "Q43668003", "Q71126987"],
             "allowed": r"chemist|physical chemist|organic chemist|theoretical chemist|inorganic chemist|analytical chemist|electrochemist"},
    "bio": {"label": "BIOLOGY", "qids": ["Q864503", "Q350979", "Q2374149", "Q3126128", "Q3779582", "Q15839206", "Q15839134", "Q3055126",
                                          "Q3640160", "Q16063497", "Q15839136", "Q1225716"],
            "allowed": r"biologist|zoologist|botanist|geneticist|microbiologist|molecular biologist|ecologist|entomologist|"
                       r"marine biologist|evolutionary biologist|cell biologist|ornithologist"},
    "cs": {"label": "COMPUTER SCIENCE", "qids": ["Q82594", "Q15976092"],
           "allowed": r"computer scientist|artificial intelligence researcher"},
    "phil": {"label": "PHILOSOPHY", "qids": ["Q4964182", "Q5403434", "Q18930007"],
             "allowed": r"philosopher|ethicist|political philosopher"},
    "econ": {"label": "ECONOMICS", "qids": ["Q188094", "Q16773661"], "allowed": r"economist|econometrician"},
}
KEYS = list(DEPARTMENTS)


def is_strict_for(dept, occupations):
    allowed = re.compile(rf"^({DEPARTMENTS[dept]['allowed']}|{GENERIC})$", re.I)
    occs = [o for o in occupations.split("|") if o]
    return bool(occs) and all(allowed.match(o) for o in occs)


def fetch():
    from .wikidata import download_all, image_path, occupations, parse_people, people_query, resolve_thumbs, session, sparql
    (DEPT / "images").mkdir(parents=True, exist_ok=True)
    frames = []
    for key, d in DEPARTMENTS.items():
        if "qids" not in d:
            continue
        people = parse_people(sparql(people_query(d["qids"]))).assign(label=key)
        print(f"{key}: {len(people)} people with a photo", flush=True)
        frames.append(people)
    df = pd.concat(frames)
    df = df[~df.qid.duplicated(keep=False)]                      # listed under two departments' occupations
    df = df.merge(occupations(df.qid.tolist()), on="qid")
    df = df[[is_strict_for(k, o) for k, o in zip(df.label, df.occupations)]]
    df = df.sample(frac=1, random_state=0).groupby("label").head(CAP)  # up to CAP random people per department
    df["file"] = df.img.map(lambda u: image_path(u, "data/dept/images"))
    s = session()
    df["thumb"] = df.img.map(resolve_thumbs(s, df.img.unique()))
    df["downloaded"] = download_all(s, df)
    df.to_csv(DEPT / "people.csv", index=False)
    print("strict people per department (downloaded):", df[df.downloaded].label.value_counts().to_dict())


def load():
    """All 7 departments with features; physics/math reuse the main clean set and its train/test split."""
    from .clean import photo_kind
    from .train import features
    main = pd.read_csv(DATA / "clean.csv")
    main_split = pd.read_csv(OUTPUTS / "split.csv").set_index("group").test
    pm = main.assign(dept=np.where(main.label == "physicist", "physics", "math"), test=main.qid.map(main_split).fillna(False))
    X_pm = features("", pm.row.values)
    new = pd.read_csv(DATA / "faces_dept.csv")
    ok = photo_kind("_dept") == "photo"
    new, X_new = new[ok].reset_index(drop=True), features("_dept")[ok]
    new = new.assign(dept=new.label)
    split_path = OUT / "split.csv"
    if split_path.exists():
        test = pd.read_csv(split_path).set_index("qid").test
    else:  # 20% of each new department's people held out, fixed once
        rng = random.Random(0)
        test = pd.Series({q: rng.random() < 0.2 for q in new.qid.unique()}, name="test")
        OUT.mkdir(parents=True, exist_ok=True)
        test.rename_axis("qid").reset_index().to_csv(split_path, index=False)
    new["test"] = new.qid.map(test).fillna(False)
    cols = ["qid", "name", "dept", "test", "file", "img", "birth_year"]
    df = pd.concat([pm[cols], new[cols]], ignore_index=True)
    df["test"] = df.test.astype(bool)
    return df, np.vstack([X_pm, X_new])


def train():
    OUT.mkdir(parents=True, exist_ok=True)
    df, X = load()
    y = df.dept.map({k: i for i, k in enumerate(KEYS)}).values
    test = df.test.values
    train_ = ~test
    print("people per department (train / test):")
    print(df.groupby(["dept", "test"]).size().unstack().rename(columns={False: "train", True: "test"}).to_string(), "\n")
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    scores = {}
    for C in [0.1, 0.3, 1, 3]:
        p = cross_val_predict(LogisticRegression(max_iter=5000, C=C), X[train_], y[train_], groups=df.qid[train_],
                              cv=cv, method="predict_proba", n_jobs=-1)
        scores[C] = accuracy_score(y[train_], p.argmax(1))
    best = max(scores.values())
    C = min(c for c in scores if scores[c] >= best - 0.005)
    model = LogisticRegression(max_iter=5000, C=C).fit(X[train_], y[train_])
    P = model.predict_proba(X)
    pred = P.argmax(1)
    top2 = (np.argsort(-P, 1)[:, :2] == y[:, None]).any(1)
    rep = {
        "C": C, "cv_acc_by_C": {str(k): round(v, 3) for k, v in scores.items()},
        "chance": round(1 / len(KEYS), 3),
        "train_acc": round(accuracy_score(y[train_], pred[train_]), 3),
        "test_acc": round(accuracy_score(y[test], pred[test]), 3),
        "test_top2": round(top2[test].mean(), 3),
        "per_department_test_acc": {k: round(float((pred[test & (y == i)] == i).mean()), 3) for i, k in enumerate(KEYS)},
        "confusion_test": confusion_matrix(y[test], pred[test]).tolist(),
    }
    print(f"C = {C}  (grouped CV accuracy on train: {rep['cv_acc_by_C']})")
    print(f"chance {rep['chance']:.0%} | TRAIN acc {rep['train_acc']:.1%} | TEST acc {rep['test_acc']:.1%} | TEST top-2 {rep['test_top2']:.1%}")
    print("per department (test):", rep["per_department_test_acc"])
    cm = pd.DataFrame(rep["confusion_test"], index=[f"true {k}" for k in KEYS], columns=KEYS)
    print(cm.to_string())
    joblib.dump({"features": MODEL_FEATURES, "clf": model, "classes": KEYS}, OUT / "model.joblib")
    df.assign(**{f"p_{k}": P[:, i] for i, k in enumerate(KEYS)})[test].to_csv(OUT / "test_predictions.csv", index=False)
    json.dump(rep, open(OUT / "report.json", "w"), indent=2)


def game(per_dept=150):
    """game/dept/data.json + portraits: up to `per_dept` held-out test people per department."""
    from .faces import detector, main_face, open_image
    from .game import descriptions, portrait
    test = pd.read_csv(OUT / "test_predictions.csv")
    pool = test.sample(frac=1, random_state=0).groupby("dept").head(per_dept)
    (GAME / "img").mkdir(parents=True, exist_ok=True)
    info = descriptions(pool.qid.tolist())
    mtcnn, people = detector(), []
    for row in pool.sort_values("qid").itertuples():
        img = open_image(row.file)
        box = main_face(mtcnn, img, min_prob=0.9)
        if box is None:
            continue
        boxes, probs = mtcnn.detect(img)
        if sum(pr > 0.9 and b[2] - b[0] > 0.5 * (box[2] - box[0]) for b, pr in zip(boxes, probs)) > 1:
            continue
        k = len(people)
        portrait(img, box).save(GAME / "img" / f"{k:03d}.jpg", quality=88)
        people.append({
            "id": k, "img": f"dept/img/{k:03d}.jpg", "name": row.name, "born": int(row.birth_year), "answer": row.dept,
            "p": {d: round(float(getattr(row, f"p_{d}")), 3) for d in KEYS},
            "bio": info.get(row.qid, {}).get("bio", ""), "wiki": info.get(row.qid, {}).get("wiki", ""),
            "source": "https://commons.wikimedia.org/wiki/File:" + row.img.rsplit("/", 1)[1],
        })
    acc = sum(max(p["p"], key=p["p"].get) == p["answer"] for p in people) / len(people)
    meta = {"departments": {k: DEPARTMENTS[k]["label"] for k in KEYS}, "model_accuracy": round(acc, 3)}
    json.dump({**meta, "people": people}, open(GAME / "data.json", "w"), indent=0)
    credits = ["# Photo credits (Guess the department)", "", "Cropped from freely licensed images on Wikimedia Commons;",
               "author and license for each photo are on its Commons page.", ""]
    credits += [f"- `{p['img']}` {p['name']}: {p['source']}" for p in people]
    (GAME / "CREDITS.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    print(f"{len(people)} fighters -> game/dept/data.json (model accuracy on them: {acc:.0%})")
