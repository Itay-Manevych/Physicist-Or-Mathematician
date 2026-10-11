"""Polite scraper for university department people pages.

  python uni_scrape.py probe            -> check every seed in uni_seeds.csv (robots, status, #portrait candidates)
  python uni_scrape.py crawl            -> crawl allowed seeds (+ pagination), download candidate portraits

Respects robots.txt (incl. Crawl-delay) for our User-Agent, never retries 403/429 blocks, stores no names.
Seeds file columns: university,label,url
"""
import os
import re
import sys
import time
import hashlib
import urllib.robotparser
from urllib.parse import urljoin, urlparse
import requests
import pandas as pd
from bs4 import BeautifulSoup
from fetch_data import UA

S = requests.Session()
S.headers.update({"User-Agent": UA})
SKIP = re.compile(r"logo|icon|facebook|youtube|twitter|linkedin|instagram|banner|header|footer|sprite|placeholder|"
                  r"default[-_]?(user|avatar|profile)|no[-_]?(photo|image)|arrow|search|menu|\.svg|\.gif", re.I)
_robots, _last = {}, {}


def robots(url):
    host = urlparse(url).scheme + "://" + urlparse(url).netloc
    if host not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = S.get(host + "/robots.txt", timeout=20)
            rp.parse(r.text.splitlines() if r.ok else [])
        except requests.RequestException:
            rp.parse([])
        _robots[host] = rp
    return _robots[host]


def polite_get(url):
    rp = robots(url)
    if not rp.can_fetch(UA, url):
        return None, "robots-disallow"
    host = urlparse(url).netloc
    delay = max(float(rp.crawl_delay(UA) or 0), 2.0)
    wait = _last.get(host, 0) + delay - time.time()
    if wait > 0:
        time.sleep(wait)
    _last[host] = time.time()
    try:
        r = S.get(url, timeout=30)
    except requests.RequestException as e:
        return None, type(e).__name__
    return (r, r.status_code) if r.ok else (None, r.status_code)


def portrait_candidates(html, base):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for img in soup.find_all("img"):
        src = img.get("data-src") or img.get("data-lazy-src") or img.get("src") or ""
        if not src and img.get("srcset"):
            src = img["srcset"].split(",")[-1].split()[0]
        if not src or src.startswith("data:") or SKIP.search(src) or SKIP.search(img.get("class", [""])[0] if img.get("class") else ""):
            continue
        try:
            w, h = int(img.get("width", 999)), int(img.get("height", 999))
            if min(w, h) < 60:
                continue
        except ValueError:
            pass
        out.append(urljoin(base, src))
    return list(dict.fromkeys(out))


def pagination(html, base, start):
    """Links that look like further pages of the same listing (same path, different page/letter query)."""
    soup = BeautifulSoup(html, "html.parser")
    p0 = urlparse(start)
    links = set()
    for a in soup.find_all("a", href=True):
        u = urljoin(base, a["href"]).split("#")[0]
        p = urlparse(u)
        if p.netloc == p0.netloc and p.path.rstrip("/") == p0.path.rstrip("/") and re.search(r"(page|letter|p)=", p.query):
            links.add(u)
    return sorted(links)


PEOPLE_LINK = re.compile(r"faculty|people|staff|members|researchers|scientists|סגל|אנשים|חברי", re.I)


def discover(path="uni_homepages.csv", out="uni_seeds.csv"):
    """From department homepages, follow 'faculty/people/staff' links and keep the page with the most portraits."""
    homes = pd.read_csv(path)
    seeds = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["university", "label", "url"])
    for h in homes.itertuples():
        r, status = polite_get(h.url)
        if r is None:
            print(f"{h.university}/{h.label}: homepage {status}", flush=True)
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        host = urlparse(r.url).netloc
        cands = [r.url]
        for a in soup.find_all("a", href=True):
            u = urljoin(r.url, a["href"]).split("#")[0]
            if urlparse(u).netloc == host and PEOPLE_LINK.search(a.get_text(" ") + " " + a["href"]):
                cands.append(u)
        best = (0, None)
        for u in list(dict.fromkeys(cands))[:10]:
            rr, _ = polite_get(u)
            if rr is not None:
                n = len(portrait_candidates(rr.text, rr.url))
                if n > best[0]:
                    best = (n, rr.url)
        print(f"{h.university}/{h.label}: best {best}", flush=True)
        if best[0] >= 5 and best[1] not in set(seeds.url):
            seeds.loc[len(seeds)] = [h.university, h.label, best[1]]
    seeds.to_csv(out, index=False)


def probe():
    seeds = pd.read_csv("uni_seeds.csv")
    rows = []
    for s in seeds.itertuples():
        r, status = polite_get(s.url)
        n = len(portrait_candidates(r.text, r.url)) if r is not None else 0
        pages = len(pagination(r.text, r.url, s.url)) if r is not None else 0
        rows.append({"university": s.university, "label": s.label, "url": s.url, "status": status, "imgs": n, "pages": pages})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("uni_probe.csv", index=False)


def crawl():
    seeds = pd.read_csv("uni_seeds.csv")
    if os.path.exists("uni_probe.csv"):
        ok = pd.read_csv("uni_probe.csv")
        ok = ok[(ok.status.astype(str) == "200") & (ok.imgs >= 5)]
        seeds = seeds.merge(ok[["url"]], on="url")
    meta_path = "data/uni.csv"
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else pd.DataFrame(columns=["university", "label", "file", "img_url"])
    seen = set(meta.img_url)
    for s in seeds.itertuples():
        folder = f"data/uni/{re.sub(r'[^a-z0-9]+', '_', s.university.lower())}"
        os.makedirs(folder, exist_ok=True)
        todo, done, imgs = [s.url], set(), []
        while todo and len(done) < 40:
            u = todo.pop(0)
            if u in done:
                continue
            done.add(u)
            r, _ = polite_get(u)
            if r is None:
                continue
            imgs += portrait_candidates(r.text, r.url)
            todo += [p for p in pagination(r.text, r.url, s.url) if p not in done]
        new = 0
        for img_url in dict.fromkeys(imgs):
            if img_url in seen:
                continue
            r, _ = polite_get(img_url)
            if r is None or not r.headers.get("content-type", "").startswith("image"):
                continue
            f = f"{folder}/{hashlib.md5(img_url.encode()).hexdigest()}.jpg"
            with open(f, "wb") as fh:
                fh.write(r.content)
            seen.add(img_url)
            meta.loc[len(meta)] = [s.university, s.label, f, img_url]
            new += 1
        meta.to_csv(meta_path, index=False)
        print(f"{s.university} / {s.label}: {len(done)} pages, {new} images", flush=True)


if __name__ == "__main__":
    {"probe": probe, "crawl": crawl, "discover": discover}[sys.argv[1]]()
