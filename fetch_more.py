"""Add Wikidata people whose occupation is a physics/maths SUB-field (e.g. 'theoretical physicist', 'topologist')
but who were not in the first download. Same filters: born >= 1900, has an image, strict occupations only.

Writes data/new/people.csv + data/new/images/ and data/new/occupations.csv.
"""
import os
import time
import hashlib
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from fetch_data import HEADERS, SPARQL, MIN_BIRTH_YEAR, resolve_thumbs, download
from labels import is_strict

SUBFIELDS = {
    "physicist": ["Q752129", "Q16742096", "Q19350898", "Q2998308", "Q63017728", "Q29909337", "Q16745601",
                  "Q139379879", "Q27067722", "Q105627120", "Q109123470", "Q99284310", "Q118313857",
                  "Q106968355", "Q122820165"],
    "mathematician": ["Q14565331", "Q18663426", "Q102038373", "Q72271334", "Q97016341", "Q130302558",
                      "Q21772571", "Q23900107", "Q48999002", "Q118878454", "Q115653130"],
}
os.makedirs("data/new/images", exist_ok=True)


def query(occ_qids):
    values = " ".join(f"wd:{q}" for q in occ_qids)
    q = f"""
    SELECT ?p ?pLabel (SAMPLE(?img) AS ?image) (SAMPLE(?birth) AS ?b) (SAMPLE(?genderLabel) AS ?g)
           (GROUP_CONCAT(DISTINCT ?occLabel; separator="|") AS ?occs)
           (COUNT(DISTINCT ?adv) AS ?n_adv) (COUNT(DISTINCT ?emp) AS ?n_uni) WHERE {{
      VALUES ?o {{ {values} }}
      ?p wdt:P31 wd:Q5; wdt:P106 ?o; wdt:P18 ?img; wdt:P569 ?birth.
      FILTER(YEAR(?birth) >= {MIN_BIRTH_YEAR})
      ?p wdt:P106 ?occ. ?occ rdfs:label ?occLabel. FILTER(LANG(?occLabel) = "en")
      OPTIONAL {{ ?p wdt:P21 ?gender. ?gender rdfs:label ?genderLabel. FILTER(LANG(?genderLabel) = "en") }}
      OPTIONAL {{ ?p wdt:P184 ?adv. }}
      OPTIONAL {{ ?p wdt:P108 ?emp. ?emp wdt:P31/wdt:P279* wd:Q3918. }}
      ?p rdfs:label ?pLabel. FILTER(LANG(?pLabel) = "en")
    }} GROUP BY ?p ?pLabel"""
    r = requests.get(SPARQL, params={"query": q, "format": "json"}, headers=HEADERS, timeout=300)
    r.raise_for_status()
    return pd.DataFrame([{
        "qid": b["p"]["value"].rsplit("/", 1)[1], "name": b["pLabel"]["value"], "img": b["image"]["value"],
        "birth_year": int(b["b"]["value"][:4].lstrip("+-") or 0), "gender": b.get("g", {}).get("value", ""),
        "occupations": b["occs"]["value"], "has_advisor": int(b["n_adv"]["value"]) > 0,
        "uni_employer": int(b["n_uni"]["value"]) > 0} for b in r.json()["results"]["bindings"]])


def main():
    old = set(pd.read_csv("data/people.csv").qid)
    frames = []
    for label, qids in SUBFIELDS.items():
        df = query(qids).assign(label=label)
        print(label, len(df), "people with a sub-field occupation")
        frames.append(df)
        time.sleep(2)
    df = pd.concat(frames)
    df = df[~df.qid.isin(old)]
    df = df[~df.qid.duplicated(keep=False)]  # in both groups -> ambiguous
    df["strict"] = df.occupations.map(is_strict)
    print("new people:", len(df), "| strict:", df.strict.sum(), df[df.strict].label.value_counts().to_dict())
    df = df[df.strict]
    df["file"] = df.img.map(lambda u: "data/new/images/" + hashlib.md5(u.encode()).hexdigest() + ".jpg")
    s = requests.Session()
    s.headers.update(HEADERS)
    df["thumb"] = df.img.map(resolve_thumbs(s, df.img.unique()))
    with ThreadPoolExecutor(4) as ex:
        df["downloaded"] = list(tqdm(ex.map(lambda a: download(s, *a), zip(df.thumb, df.file)), total=len(df)))
    df.to_csv("data/new/people.csv", index=False)
    print("downloaded:", df.downloaded.sum(), df[df.downloaded].label.value_counts().to_dict())


if __name__ == "__main__":
    main()
