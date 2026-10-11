"""Collect labelled portraits from Wikidata / Wikimedia Commons (freely licensed)."""
import hashlib
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from tqdm import tqdm

from .config import DATA, MIN_BIRTH_YEAR, USER_AGENT
from .labels import is_strict

SPARQL = "https://query.wikidata.org/sparql"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OCCUPATION = {"physicist": "Q169470", "mathematician": "Q170790"}
# people listed only under a sub-field (e.g. 'theoretical physicist', 'topologist')
SUBFIELDS = {
    "physicist": ["Q752129", "Q16742096", "Q19350898", "Q2998308", "Q63017728", "Q29909337", "Q16745601",
                  "Q139379879", "Q27067722", "Q105627120", "Q109123470", "Q99284310", "Q118313857",
                  "Q106968355", "Q122820165"],
    "mathematician": ["Q14565331", "Q18663426", "Q102038373", "Q72271334", "Q97016341", "Q130302558",
                      "Q21772571", "Q23900107", "Q48999002", "Q118878454", "Q115653130"],
}


def session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def sparql(query):
    for attempt in range(4):
        r = requests.get(SPARQL, params={"query": query, "format": "json"},
                         headers={"User-Agent": USER_AGENT}, timeout=300)
        if r.ok:
            return r.json()["results"]["bindings"]
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()


def image_path(url, folder):
    return f"{folder}/{hashlib.md5(url.encode()).hexdigest()}.jpg"


def resolve_thumbs(s, commons_urls, width=500):
    """Commons file URL -> direct thumbnail URL, 50 titles per API call."""
    names = {u: "File:" + urllib.parse.unquote(u.rsplit("/", 1)[1]) for u in commons_urls}
    titles, by_title = list(set(names.values())), {}
    for i in tqdm(range(0, len(titles), 50), desc="resolving thumbnails"):
        chunk = titles[i:i + 50]
        q = s.get(COMMONS_API, params={"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url",
                                       "iiurlwidth": width, "titles": "|".join(chunk)}, timeout=60).json().get("query", {})
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        info = {p["title"]: p.get("imageinfo", [{}])[0].get("thumburl") for p in q.get("pages", {}).values()}
        for t in chunk:
            by_title[t] = info.get(norm.get(t, t))
        time.sleep(0.2)
    return {u: by_title.get(t) for u, t in names.items()}


def download(s, url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    if not isinstance(url, str):
        return False
    for attempt in range(5):
        try:
            r = s.get(url, timeout=60)
            if r.status_code == 429:  # rate limited: back off
                time.sleep(5 * (attempt + 1))
                continue
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
            return False
        except requests.RequestException:
            time.sleep(3)
    return False


def download_all(s, df, desc="downloading"):
    with ThreadPoolExecutor(4) as ex:
        return list(tqdm(ex.map(lambda a: download(s, *a), zip(df.thumb, df.file)), total=len(df), desc=desc))


def people_query(occ_qids):
    values = " ".join(f"wd:{q}" for q in occ_qids)
    return f"""
    SELECT ?p ?pLabel (SAMPLE(?img) AS ?image) (SAMPLE(?birth) AS ?b) (SAMPLE(?genderLabel) AS ?g) WHERE {{
      VALUES ?o {{ {values} }}
      ?p wdt:P31 wd:Q5; wdt:P106 ?o; wdt:P18 ?img; wdt:P569 ?birth.
      FILTER(YEAR(?birth) >= {MIN_BIRTH_YEAR})
      OPTIONAL {{ ?p wdt:P21 ?gender. ?gender rdfs:label ?genderLabel. FILTER(LANG(?genderLabel) = "en") }}
      OPTIONAL {{ ?p rdfs:label ?pLabel. FILTER(LANG(?pLabel) = "en") }}
    }} GROUP BY ?p ?pLabel"""


def parse_people(rows):
    return pd.DataFrame([{
        "qid": b["p"]["value"].rsplit("/", 1)[1], "img": b["image"]["value"],
        "name": b.get("pLabel", {}).get("value", b["p"]["value"].rsplit("/", 1)[1]),
        "birth_year": int(b["b"]["value"][:4].lstrip("+-") or 0), "gender": b.get("g", {}).get("value", "")}
        for b in rows])


def occupations(qids):
    """All occupations + academic signals (doctoral advisor, university employer, Commons category) per person."""
    out = []
    for i in tqdm(range(0, len(qids), 300), desc="occupations"):
        values = " ".join(f"wd:{q}" for q in qids[i:i + 300])
        rows = sparql(f"""
        SELECT ?p (GROUP_CONCAT(DISTINCT ?occLabel; separator="|") AS ?occs)
               (COUNT(DISTINCT ?adv) AS ?n_adv) (COUNT(DISTINCT ?emp) AS ?n_uni) (SAMPLE(?cat) AS ?commonscat) WHERE {{
          VALUES ?p {{ {values} }}
          OPTIONAL {{ ?p wdt:P106 ?occ. ?occ rdfs:label ?occLabel. FILTER(LANG(?occLabel) = "en") }}
          OPTIONAL {{ ?p wdt:P184 ?adv. }}
          OPTIONAL {{ ?p wdt:P108 ?emp. ?emp wdt:P31/wdt:P279* wd:Q3918. }}
          OPTIONAL {{ ?p wdt:P373 ?cat. }}
        }} GROUP BY ?p""")
        out += [{"qid": b["p"]["value"].rsplit("/", 1)[1], "occupations": b.get("occs", {}).get("value", ""),
                 "has_advisor": int(b["n_adv"]["value"]) > 0, "uni_employer": int(b["n_uni"]["value"]) > 0,
                 "commons_category": b.get("commonscat", {}).get("value", "")} for b in rows]
        time.sleep(1)
    return pd.DataFrame(out)


def fetch_people():
    """Step 1: everyone with occupation physicist / mathematician -> data/people.csv, data/occupations.csv."""
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    frames = {label: parse_people(sparql(people_query([qid]))) for label, qid in OCCUPATION.items()}
    both = set(frames["physicist"].qid) & set(frames["mathematician"].qid)
    df = pd.concat([f.assign(label=label) for label, f in frames.items()])
    df.loc[df.qid.isin(both), "label"] = "both"
    df = df.drop_duplicates("qid")
    df["file"] = df.img.map(lambda u: image_path(u, "data/images"))
    s = session()
    df["thumb"] = df.img.map(resolve_thumbs(s, df.img.unique()))
    df["downloaded"] = download_all(s, df)
    df.to_csv(DATA / "people.csv", index=False)
    occupations(df.qid.tolist()).to_csv(DATA / "occupations.csv", index=False)
    print(df.groupby("label").downloaded.sum())


def fetch_subfields():
    """Extra people listed only under a physics/maths sub-field -> data/new/people.csv."""
    (DATA / "new" / "images").mkdir(parents=True, exist_ok=True)
    old = set(pd.read_csv(DATA / "people.csv").qid)
    df = pd.concat([parse_people(sparql(people_query(q))).assign(label=label) for label, q in SUBFIELDS.items()])
    df = df[~df.qid.isin(old) & ~df.qid.duplicated(keep=False)]
    df = df.merge(occupations(df.qid.tolist()), on="qid")
    df = df[df.occupations.map(is_strict)]
    df["file"] = df.img.map(lambda u: image_path(u, "data/new/images"))
    s = session()
    df["thumb"] = df.img.map(resolve_thumbs(s, df.img.unique()))
    df["downloaded"] = download_all(s, df)
    df.to_csv(DATA / "new" / "people.csv", index=False)
    print("new strict people:", df[df.downloaded].label.value_counts().to_dict())


def fetch_extra(max_per_person=4):
    """More photos of people in the clean set, from their Commons category -> data/extra.csv."""
    (DATA / "extra").mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(DATA / "clean.csv")
    clean = clean[clean.commons_category.fillna("") != ""]
    s, jobs = session(), []
    for row in tqdm(clean.itertuples(), total=len(clean), desc="listing categories"):
        own = "File:" + urllib.parse.unquote(row.img.rsplit("/", 1)[1]).replace("_", " ")
        try:
            pages = s.get(COMMONS_API, params={
                "action": "query", "format": "json", "generator": "categorymembers",
                "gcmtitle": f"Category:{row.commons_category}", "gcmtype": "file", "gcmlimit": 30,
                "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 500}, timeout=60).json()
        except (requests.RequestException, ValueError):
            continue
        files = []
        for p in pages.get("query", {}).get("pages", {}).values():
            info = (p.get("imageinfo") or [{}])[0]
            if info.get("mime") in ("image/jpeg", "image/png") and info.get("thumburl") and p["title"] != own:
                files.append((p["title"], info["thumburl"]))
        for title, url in sorted(files)[:max_per_person]:
            jobs.append({"qid": row.qid, "label": row.label, "title": title, "thumb": url,
                         "file": image_path(title, "data/extra")})
        time.sleep(0.1)
    jobs = pd.DataFrame(jobs)
    jobs["downloaded"] = download_all(s, jobs)
    jobs.to_csv(DATA / "extra.csv", index=False)
    print(len(jobs), "extra photos for", jobs.qid.nunique(), "people")
