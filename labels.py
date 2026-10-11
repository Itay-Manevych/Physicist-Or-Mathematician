"""Label cleanup: keep people whose Wikidata occupations are only academic physics/maths."""
import re
import pandas as pd

OTHER_CAREERS = re.compile(
    r"politician|astronaut|engineer|inventor|business|entrepreneur|journalist|poet|physician|chemist|"
    r"computer scientist|economist|philosopher|astronomer|programmer|chess|military|officer|diplomat|translator|"
    r"historian|pedagogue|science communicator|novelist|priest|theologian|lawyer|musician|composer|painter|actor|"
    r"banker|biologist|geologist|meteorologist|cryptographer|statistician|science fiction|television|manager|executive",
    re.I)


# Every occupation a "strict" person has must be one of these (physics, maths, or generic academia).
ALLOWED = re.compile(
    r"^(mathematician|physicist|astrophysicist|nuclear physicist|theoretical physicist|particle physicist|"
    r"experimental physicist|quantum physicist|plasma physicist|statistical physicist|atomic physicist|applied physicist|"
    r"laser physicist|accelerator physicist|materials physicist|condensed matter physicist|solid-state physicist|"
    r"string theorist|cosmologist|topologist|logician|applied mathematician|number theorist|professor of mathematics|"
    r"geometer|differential geometer|algebraist|numerical analyst|graph theorist|probability theorist|"
    r"mathematical analyst|probabilist|combinatorialist|"
    r"university teacher|professor|academic|researcher|scientist|lecturer|docent|research fellow|"
    r"professeur des universit.s|teacher|mathematics teacher|physics teacher|educator)$", re.I)


def is_strict(occupations):
    occs = [o for o in occupations.split("|") if o]
    return bool(occs) and all(ALLOWED.match(o) for o in occs)


def load_faces():
    """data/faces.csv joined with occupations, plus `clean` / `strict` / `academic` flags.

    clean    = no obviously different career (blocklist)
    strict   = every Wikidata occupation is physics/maths/academic (allowlist)
    academic = strict AND has a doctoral advisor or a university employer on Wikidata
    """
    df = pd.read_csv("data/faces.csv")
    occ = pd.read_csv("data/occupations.csv").fillna("")
    df = df.merge(occ, on="qid", how="left").fillna({"occupations": "", "commons_category": ""})
    df["clean"] = ~df.occupations.str.contains(OTHER_CAREERS)
    df["strict"] = df.occupations.map(is_strict)
    has_affil = df.has_advisor.fillna(False).astype(bool) | df.uni_employer.fillna(False).astype(bool)
    df["academic"] = df.strict & has_affil
    return df
