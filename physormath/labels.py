"""Strict labels: a person counts only if every Wikidata occupation is physics, maths or generic academia."""
import re

import pandas as pd

from .config import DATA

ALLOWED = re.compile(
    r"^(mathematician|physicist|astrophysicist|nuclear physicist|theoretical physicist|particle physicist|"
    r"experimental physicist|quantum physicist|plasma physicist|statistical physicist|atomic physicist|"
    r"applied physicist|laser physicist|accelerator physicist|materials physicist|condensed matter physicist|"
    r"solid-state physicist|string theorist|cosmologist|topologist|logician|applied mathematician|number theorist|"
    r"professor of mathematics|geometer|differential geometer|algebraist|numerical analyst|graph theorist|"
    r"probability theorist|mathematical analyst|probabilist|combinatorialist|"
    r"university teacher|professor|academic|researcher|scientist|lecturer|docent|research fellow|"
    r"professeur des universit.s|teacher|mathematics teacher|physics teacher|educator)$", re.I)


def is_strict(occupations: str) -> bool:
    occs = [o for o in occupations.split("|") if o]
    return bool(occs) and all(ALLOWED.match(o) for o in occs)


def load_faces() -> pd.DataFrame:
    """data/faces.csv joined with occupations, plus `strict` and `academic` flags."""
    df = pd.read_csv(DATA / "faces.csv")
    occ = pd.read_csv(DATA / "occupations.csv").fillna("")
    df = df.merge(occ, on="qid", how="left").fillna({"occupations": "", "commons_category": ""})
    df["strict"] = df.occupations.map(is_strict)
    affiliated = df.has_advisor.fillna(False).astype(bool) | df.uni_employer.fillna(False).astype(bool)
    df["academic"] = df.strict & affiliated
    return df
