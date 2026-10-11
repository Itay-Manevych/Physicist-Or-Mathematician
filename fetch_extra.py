"""Download extra portraits per person from their Wikimedia Commons category (up to MAX_EXTRA each).

Only for the clean academic set. Identity is verified later (embed_extra.py) with FaceNet.
"""
import os
import time
import hashlib
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from fetch_data import HEADERS, download
from labels import load_faces

MAX_EXTRA = 4
os.makedirs("data/extra", exist_ok=True)


def list_category(session, cat, exclude):
    r = session.get("https://commons.wikimedia.org/w/api.php", params={
        "action": "query", "format": "json", "generator": "categorymembers", "gcmtitle": f"Category:{cat}",
        "gcmtype": "file", "gcmlimit": 30, "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 500}, timeout=60)
    out = []
    for p in r.json().get("query", {}).get("pages", {}).values():
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") in ("image/jpeg", "image/png") and info.get("thumburl") and p["title"] not in exclude:
            out.append((p["title"], info["thumburl"]))
    return sorted(out)[:MAX_EXTRA]


def main():
    df = load_faces()
    df = df[df.academic & df.label.isin(["physicist", "mathematician"]) & (df.commons_category != "")]
    s = requests.Session()
    s.headers.update(HEADERS)
    jobs = []
    for row in tqdm(df.itertuples(), total=len(df), desc="listing categories"):
        own = "File:" + requests.utils.unquote(row.img.rsplit("/", 1)[1]).replace("_", " ")
        try:
            for title, url in list_category(s, row.commons_category, {own}):
                path = "data/extra/" + hashlib.md5(title.encode()).hexdigest() + ".jpg"
                jobs.append({"qid": row.qid, "title": title, "thumb": url, "file": path})
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.1)
    jobs = pd.DataFrame(jobs)
    with ThreadPoolExecutor(4) as ex:
        jobs["downloaded"] = list(tqdm(ex.map(lambda a: download(s, *a), zip(jobs.thumb, jobs.file)),
                                       total=len(jobs), desc="downloading"))
    jobs.to_csv("data/extra.csv", index=False)
    print(len(jobs), "extra images,", jobs.downloaded.sum(), "downloaded, for", jobs.qid.nunique(), "people")


if __name__ == "__main__":
    main()
