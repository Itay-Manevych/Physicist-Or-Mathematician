"""Collect labelled portraits from Wikidata / Wikimedia Commons (freely licensed)."""
import hashlib
import os
import re
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


def people_query(occ_qids, born_from=MIN_BIRTH_YEAR, born_before=9999):
    values = " ".join(f"wd:{q}" for q in occ_qids)
    return f"""
    SELECT ?p ?pLabel (SAMPLE(?img) AS ?image) (SAMPLE(?birth) AS ?b) (SAMPLE(?genderLabel) AS ?g) WHERE {{
      VALUES ?o {{ {values} }}
      ?p wdt:P31 wd:Q5; wdt:P106 ?o; wdt:P18 ?img; wdt:P569 ?birth.
      FILTER(YEAR(?birth) >= {born_from} && YEAR(?birth) < {born_before})
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


def fetch_depicts(max_per_person=8):
    """More photos per clean person: Commons files whose structured data says they depict that person (P180).
    Skips the main portrait and photos already in data/extra.csv -> data/depicts.csv, data/depicts/."""
    (DATA / "depicts").mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(DATA / "clean.csv")
    have = set(pd.read_csv(DATA / "extra.csv").title) if (DATA / "extra.csv").exists() else set()
    s, jobs = session(), []
    for row in tqdm(clean.itertuples(), total=len(clean), desc="searching depicts"):
        own = "File:" + urllib.parse.unquote(row.img.rsplit("/", 1)[1]).replace("_", " ")
        try:
            pages = s.get(COMMONS_API, params={
                "action": "query", "format": "json", "generator": "search", "gsrsearch": f"haswbstatement:P180={row.qid}",
                "gsrnamespace": 6, "gsrlimit": 20, "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 500},
                timeout=60).json()
        except (requests.RequestException, ValueError):
            continue
        files = []
        for p in pages.get("query", {}).get("pages", {}).values():
            info = (p.get("imageinfo") or [{}])[0]
            if (info.get("mime") in ("image/jpeg", "image/png") and info.get("thumburl")
                    and p["title"] != own and p["title"] not in have):
                files.append((p.get("index", 99), p["title"], info["thumburl"]))
        for _, title, url in sorted(files)[:max_per_person]:
            jobs.append({"qid": row.qid, "label": row.label, "title": title, "thumb": url,
                         "file": image_path(title, "data/depicts")})
        time.sleep(0.1)
    jobs = pd.DataFrame(jobs).drop_duplicates("file")
    jobs["downloaded"] = download_all(s, jobs)
    jobs.to_csv(DATA / "depicts.csv", index=False)
    print(len(jobs), "depicts photos for", jobs.qid.nunique(), "people")


def fetch_more_photos(max_per_person=8):
    """Deeper pass over each clean person's Commons category: up to `max_per_person` photos beyond the ones in
    data/extra.csv and data/depicts.csv -> data/more.csv, data/more/ (identity is verified later with FaceNet)."""
    (DATA / "more").mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(DATA / "clean.csv")
    clean = clean[clean.commons_category.fillna("") != ""]
    have = set()
    for f in ["extra.csv", "depicts.csv"]:
        if (DATA / f).exists():
            have |= set(pd.read_csv(DATA / f).title)
    s, jobs = session(), []
    for row in tqdm(clean.itertuples(), total=len(clean), desc="listing categories (deep)"):
        own = "File:" + urllib.parse.unquote(row.img.rsplit("/", 1)[1]).replace("_", " ")
        try:
            pages = s.get(COMMONS_API, params={
                "action": "query", "format": "json", "generator": "categorymembers",
                "gcmtitle": f"Category:{row.commons_category}", "gcmtype": "file", "gcmlimit": 60,
                "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 500}, timeout=60).json()
        except (requests.RequestException, ValueError):
            continue
        files = []
        for p in pages.get("query", {}).get("pages", {}).values():
            info = (p.get("imageinfo") or [{}])[0]
            if (info.get("mime") in ("image/jpeg", "image/png") and info.get("thumburl")
                    and p["title"] != own and p["title"] not in have):
                files.append((p["title"], info["thumburl"]))
        for title, url in sorted(files)[:max_per_person]:
            jobs.append({"qid": row.qid, "label": row.label, "title": title, "thumb": url,
                         "file": image_path(title, "data/more")})
        time.sleep(0.1)
    jobs = pd.DataFrame(jobs).drop_duplicates("file")
    jobs["downloaded"] = download_all(s, jobs)
    jobs.to_csv(DATA / "more.csv", index=False)
    print(len(jobs), "more photos for", jobs.qid.nunique(), "people")


SKIP_FILE = re.compile(r"signature|unterschrift|autograph|logo|flag|coat.of.arms|wappen|diagram|equation|formula|symbol|"
                       r"medal|stamp|coin|banknote|grave|tomb|plaque|statue|building|campus|institut|universit|crater|"
                       r"asteroid|street|cover|\b(map|karte|graph|plot|icon|bust|book|house)\b|\.(svg|gif|tiff?)$", re.I)


def fetch_wikipedia_photos(max_per_person=6):
    """Photos shown in each clean person's articles on every language Wikipedia (Commons files only, so freely
    licensed), beyond the ones we already have -> data/articles.csv, data/articles/. Identity is checked later (`verify`)."""
    (DATA / "articles").mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(DATA / "clean.csv")
    have = {"File:" + urllib.parse.unquote(u.rsplit("/", 1)[1]).replace("_", " ") for u in clean.img}
    for f in ["extra.csv", "depicts.csv", "more.csv"]:
        if (DATA / f).exists():
            have |= set(pd.read_csv(DATA / f).title)
    s = session()
    cache = DATA / "articles_used.csv"  # steps 1-2 take ~20 minutes: keep their result
    if cache.exists():
        used = pd.read_csv(cache)
    else:
        used = _article_images(s, clean.qid.tolist())
        used.to_csv(cache, index=False)
    # local wikis name the namespace in their own language ("Datei:", "Fichier:"); Commons wants "File:"
    used["title"] = "File:" + used.local.str.split(":", n=1).str[1].str.replace("_", " ")
    used = used[~used.title.str.contains(SKIP_FILE) & ~used.title.isin(have)]
    used = used.groupby(["qid", "title"]).size().rename("n_wikis").reset_index()
    # 3. keep files that live on Commons, and are photos (jpeg/png)
    titles, info = used.title.unique().tolist(), {}
    for i in tqdm(range(0, len(titles), 50), desc="commons files"):
        q = _post(s, COMMONS_API, {"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url|mime",
                                   "iiurlwidth": 500, "titles": "|".join(titles[i:i + 50])}).get("query", {})
        norm = {n["to"]: n["from"] for n in q.get("normalized", [])}
        for p in q.get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            if "missing" not in p and ii.get("mime") in ("image/jpeg", "image/png") and ii.get("thumburl"):
                info[norm.get(p["title"], p["title"])] = ii["thumburl"]
        time.sleep(0.2)
    used = used[used.title.isin(info)]
    # most widely used first: a photo shown in many languages' articles is most likely of the person
    used = used.sort_values(["qid", "n_wikis", "title"], ascending=[True, False, True]).groupby("qid").head(max_per_person)
    jobs = used.merge(clean[["qid", "label"]], on="qid")
    jobs["thumb"] = jobs.title.map(info)
    jobs["file"] = jobs.title.map(lambda t: image_path(t, "data/articles"))
    jobs = jobs[~jobs.file.duplicated(keep=False)]  # one photo on several people's pages: unclear who it shows
    jobs["downloaded"] = download_all(s, jobs)
    jobs.to_csv(DATA / "articles.csv", index=False)
    print(len(jobs), "Wikipedia photos for", jobs.qid.nunique(), "people")


def _post(s, url, data):
    """API call by POST (long title lists), with retries; {} if it keeps failing."""
    for attempt in range(4):
        try:
            r = s.post(url, data=data, timeout=60)
            if r.ok:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (attempt + 1))
    return {}


def _article_images(s, qids):
    """(qid, local file title) for every image shown in any language Wikipedia's article about each person."""
    # 1. every Wikipedia article about each person (Wikidata sitelinks)
    articles = []  # (qid, wiki, title)
    for i in tqdm(range(0, len(qids), 50), desc="sitelinks"):
        r = _post(s, "https://www.wikidata.org/w/api.php", {
            "action": "wbgetentities", "format": "json", "ids": "|".join(qids[i:i + 50]), "props": "sitelinks"})
        for q, e in r.get("entities", {}).items():
            for site, link in e.get("sitelinks", {}).items():
                if re.fullmatch(r"[a-z_]+wiki", site) and site not in ("commonswiki", "specieswiki", "metawiki",
                                                                         "wikidatawiki", "mediawikiwiki", "sourceswiki"):
                    articles.append((q, site[:-4].replace("_", "-"), link["title"]))
        time.sleep(0.2)
    articles = pd.DataFrame(articles, columns=["qid", "lang", "article"])
    print(len(articles), "articles in", articles.lang.nunique(), "languages")
    # 2. the images each article shows (50 articles per call)
    used = []  # (qid, file title)
    for lang, g in tqdm(articles.groupby("lang"), desc="article images"):
        by_title = dict(zip(g.article, g.qid))
        titles = list(by_title)
        for i in range(0, len(titles), 50):
            params = {"action": "query", "format": "json", "prop": "images", "imlimit": "max", "redirects": 1,
                      "titles": "|".join(titles[i:i + 50])}
            while True:
                r = _post(s, f"https://{lang}.wikipedia.org/w/api.php", params)
                q = r.get("query", {})
                back = {n["to"]: n["from"] for k in ("normalized", "redirects") for n in q.get(k, [])}
                for p in q.get("pages", {}).values():
                    t = p.get("title", "")
                    qid = by_title.get(t) or by_title.get(back.get(t, "")) or by_title.get(back.get(back.get(t, ""), ""))
                    for im in p.get("images", []):
                        if qid:
                            used.append((qid, im["title"]))
                if "continue" not in r:
                    break
                params.update(r["continue"])
                time.sleep(0.1)
            time.sleep(0.1)
    return pd.DataFrame(used, columns=["qid", "local"])


def fetch_older(born_from=1850, born_before=MIN_BIRTH_YEAR):
    """Physicists/mathematicians born 1850-1899 (training-only extra data) -> data/older/people.csv, data/older/images/."""
    (DATA / "older" / "images").mkdir(parents=True, exist_ok=True)
    frames = {label: parse_people(sparql(people_query([q], born_from, born_before))) for label, q in OCCUPATION.items()}
    both = set(frames["physicist"].qid) & set(frames["mathematician"].qid)
    df = pd.concat([f.assign(label=label) for label, f in frames.items()])
    df = df[~df.qid.isin(both)].drop_duplicates("qid")
    df = df.merge(occupations(df.qid.tolist()), on="qid")
    df["strict"] = df.occupations.map(is_strict)
    df["file"] = df.img.map(lambda u: image_path(u, "data/older/images"))
    s = session()
    df["thumb"] = df.img.map(resolve_thumbs(s, df.img.unique()))
    df["downloaded"] = download_all(s, df)
    df.to_csv(DATA / "older" / "people.csv", index=False)
    print("older people downloaded:", df[df.downloaded].label.value_counts().to_dict(), "| strict:", int(df.strict.sum()))
