"""Collect labelled portraits of physicists and mathematicians from Wikidata / Wikimedia Commons.

People tagged with BOTH occupations are saved separately (data/both.csv) as a fun bonus test set.
"""
import os
import time
import hashlib
import urllib.parse
import requests
import pandas as pd
from tqdm import tqdm

UA = "PhysOrMath/0.1 (personal hobby project; https://github.com/Itay-Manevych/Physicist-Or-Mathematician) python-requests"
HEADERS = {"User-Agent": UA}
SPARQL = "https://query.wikidata.org/sparql"
OCC = {"physicist": "Q169470", "mathematician": "Q170790"}
MIN_BIRTH_YEAR = 1900  # limit the "old 1800s engraving vs modern photo" confound
os.makedirs("data/images", exist_ok=True)


def query(occ_qid):
    q = f"""
    SELECT DISTINCT ?p ?pLabel ?img ?birth ?genderLabel WHERE {{
      ?p wdt:P31 wd:Q5; wdt:P106 wd:{occ_qid}; wdt:P18 ?img; wdt:P569 ?birth.
      OPTIONAL {{ ?p wdt:P21 ?gender. }}
      FILTER(YEAR(?birth) >= {MIN_BIRTH_YEAR})
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}"""
    r = requests.get(SPARQL, params={"query": q, "format": "json"}, headers=HEADERS, timeout=300)
    r.raise_for_status()
    rows = []
    for b in r.json()["results"]["bindings"]:
        rows.append({
            "qid": b["p"]["value"].rsplit("/", 1)[1],
            "name": b.get("pLabel", {}).get("value", ""),
            "img": b["img"]["value"],
            "birth_year": int(b["birth"]["value"][:4].lstrip("+-") or 0),
            "gender": b.get("genderLabel", {}).get("value", ""),
        })
    return pd.DataFrame(rows).drop_duplicates("qid")


def resolve_thumbs(session, commons_urls, width=500):
    """Map Commons file URL -> direct upload.wikimedia.org thumbnail URL, 50 titles per API call."""
    names = {u: "File:" + urllib.parse.unquote(u.rsplit("/", 1)[1]) for u in commons_urls}
    by_title, out = {}, {}
    titles = list(set(names.values()))
    for i in tqdm(range(0, len(titles), 50), desc="resolving thumbs"):
        chunk = titles[i:i + 50]
        r = session.get("https://commons.wikimedia.org/w/api.php", params={
            "action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url",
            "iiurlwidth": width, "titles": "|".join(chunk)}, timeout=60)
        q = r.json().get("query", {})
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        info = {p["title"]: p.get("imageinfo", [{}])[0].get("thumburl") for p in q.get("pages", {}).values()}
        for t in chunk:
            by_title[t] = info.get(norm.get(t, t))
        time.sleep(0.2)
    for u, t in names.items():
        out[u] = by_title.get(t)
    return out


def download(session, url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    if not isinstance(url, str):
        return False
    for attempt in range(5):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 429:
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


def main():
    from concurrent.futures import ThreadPoolExecutor
    s = requests.Session()
    s.headers.update(HEADERS)
    if os.path.exists("data/people_meta.csv"):
        allp = pd.read_csv("data/people_meta.csv")
    else:
        dfs = {}
        for label, qid in OCC.items():
            print("querying", label)
            dfs[label] = query(qid)
            print(" ", len(dfs[label]), "people with images")
            time.sleep(2)
        both = set(dfs["physicist"].qid) & set(dfs["mathematician"].qid)
        print("tagged as both:", len(both))
        allp = pd.concat([df.assign(label=label) for label, df in dfs.items()])
        allp.loc[allp.qid.isin(both), "label"] = "both"
        allp = allp.drop_duplicates("qid")
        allp["file"] = allp.img.map(lambda u: "data/images/" + hashlib.md5(u.encode()).hexdigest() + ".jpg")
        allp.to_csv("data/people_meta.csv", index=False)

    thumbs = resolve_thumbs(s, allp.img.unique())
    allp["thumb"] = allp.img.map(thumbs)
    with ThreadPoolExecutor(4) as ex:
        ok = list(tqdm(ex.map(lambda a: download(s, *a), zip(allp.thumb, allp.file)), total=len(allp), desc="downloading"))
    allp["downloaded"] = ok
    allp.to_csv("data/people.csv", index=False)
    print(allp.groupby("label").downloaded.sum())


if __name__ == "__main__":
    main()
