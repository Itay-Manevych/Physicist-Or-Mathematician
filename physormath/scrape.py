"""Polite scraper for university department pages and Simons Investigators.

Respects robots.txt (including Crawl-delay) for our User-Agent, never retries 403/429 blocks, stores no names.
These photos are only used for local training (see docs/DATA.md).
"""
import hashlib
import os
import re
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import DATA, SOURCES, USER_AGENT

S = requests.Session()
S.headers.update({"User-Agent": USER_AGENT})
SKIP_IMG = re.compile(r"logo|icon|facebook|youtube|twitter|linkedin|instagram|banner|header|footer|sprite|placeholder|"
                      r"default[-_]?(user|avatar|profile)|no[-_]?(photo|image)|arrow|search|menu|\.svg|\.gif", re.I)
PEOPLE_LINK = re.compile(r"faculty|people|staff|members|researchers|scientists|סגל|אנשים|חברי", re.I)
NOT_DIRECTORY = re.compile(r"news|event|memoriam|in-memory|obituar|blog|seminar|colloquium|award|gallery", re.I)
SEEDS, HOMEPAGES, PROBE = SOURCES / "uni_seeds.csv", SOURCES / "uni_homepages.csv", SOURCES / "uni_probe.csv"
META = DATA / "uni.csv"
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
    """GET honoring robots.txt and its crawl delay (at least 2 s per host). Returns (response | None, status)."""
    rp = robots(url)
    if not rp.can_fetch(USER_AGENT, url):
        return None, "robots-disallow"
    host = urlparse(url).netloc
    wait = _last.get(host, 0) + max(float(rp.crawl_delay(USER_AGENT) or 0), 2.0) - time.time()
    if wait > 0:
        time.sleep(wait)
    _last[host] = time.time()
    try:
        r = S.get(url, timeout=30)
    except requests.RequestException as e:
        return None, type(e).__name__
    return (r, r.status_code) if r.ok else (None, r.status_code)


def portrait_candidates(html, base):
    out = []
    for img in BeautifulSoup(html, "html.parser").find_all("img"):
        src = img.get("data-src") or img.get("data-lazy-src") or img.get("src") or ""
        if not src and img.get("srcset"):
            src = img["srcset"].split(",")[-1].split()[0]
        cls = " ".join(img.get("class") or [])
        if not src or src.startswith("data:") or SKIP_IMG.search(src) or SKIP_IMG.search(cls):
            continue
        try:
            if min(int(img.get("width", 999)), int(img.get("height", 999))) < 60:
                continue
        except ValueError:
            pass
        out.append(urljoin(base, src))
    return list(dict.fromkeys(out))


def pagination(html, base, start):
    """Further pages of the same listing: same path, different page/letter query."""
    p0, links = urlparse(start), set()
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        u = urljoin(base, a["href"]).split("#")[0]
        p = urlparse(u)
        if p.netloc == p0.netloc and p.path.rstrip("/") == p0.path.rstrip("/") and re.search(r"(page|letter|p)=", p.query):
            links.add(u)
    return sorted(links)


def load_meta():
    return pd.read_csv(META) if META.exists() else pd.DataFrame(columns=["university", "label", "file", "img_url"])


def save_image(meta, seen, university, label, img_url, folder):
    if img_url in seen:
        return False
    r, _ = polite_get(img_url)
    if r is None or not r.headers.get("content-type", "").startswith("image"):
        return False
    f = f"{folder}/{hashlib.md5(img_url.encode()).hexdigest()}.jpg"
    with open(f, "wb") as fh:
        fh.write(r.content)
    seen.add(img_url)
    meta.loc[len(meta)] = [university, label, f, img_url]
    return True


def discover():
    """Department homepage -> follow 'faculty / people / staff' links -> keep the page with most portraits."""
    homes = pd.read_csv(HOMEPAGES)
    seeds = pd.read_csv(SEEDS) if SEEDS.exists() else pd.DataFrame(columns=["university", "label", "url"])
    for h in homes.itertuples():
        r, status = polite_get(h.url)
        if r is None:
            print(f"{h.university}/{h.label}: homepage {status}", flush=True)
            continue
        host = urlparse(r.url).netloc
        cands = [r.url] + [urljoin(r.url, a["href"]).split("#")[0]
                           for a in BeautifulSoup(r.text, "html.parser").find_all("a", href=True)
                           if PEOPLE_LINK.search(a.get_text(" ") + " " + a["href"])]
        cands = [u for u in dict.fromkeys(cands) if urlparse(u).netloc == host][:10]
        best = (0, None)
        for u in cands:
            rr, _ = polite_get(u)
            if rr is not None and (n := len(portrait_candidates(rr.text, rr.url))) > best[0]:
                best = (n, rr.url)
        print(f"{h.university}/{h.label}: best {best}", flush=True)
        if best[0] >= 5 and best[1] not in set(seeds.url):
            seeds.loc[len(seeds)] = [h.university, h.label, best[1]]
    seeds.to_csv(SEEDS, index=False)


def probe():
    """Check every seed: robots permission, HTTP status, number of portrait candidates."""
    rows = []
    for s in pd.read_csv(SEEDS).itertuples():
        r, status = polite_get(s.url)
        n = len(portrait_candidates(r.text, r.url)) if r is not None else 0
        rows.append({"university": s.university, "label": s.label, "url": s.url, "status": status, "imgs": n})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(PROBE, index=False)


def crawl(max_pages=40):
    """Download portraits from every usable seed (+ its pagination) into data/uni/<university>/."""
    seeds = pd.read_csv(SEEDS)
    if PROBE.exists():
        ok = pd.read_csv(PROBE)
        seeds = seeds.merge(ok[(ok.status.astype(str) == "200") & (ok.imgs >= 5)][["url"]], on="url")
    seeds = seeds[~seeds.url.str.contains(NOT_DIRECTORY)]
    seeds = seeds[seeds.url.map(lambda u: urlparse(u).path.strip("/") not in ("", "en", "index.html"))]
    meta = load_meta()
    seen = set(meta.img_url)
    for s in seeds.itertuples():
        folder = DATA / "uni" / re.sub(r"[^a-z0-9]+", "_", s.university.lower())
        folder.mkdir(parents=True, exist_ok=True)
        todo, done, imgs = [s.url], set(), []
        while todo and len(done) < max_pages:
            u = todo.pop(0)
            if u in done:
                continue
            done.add(u)
            r, _ = polite_get(u)
            if r is not None:
                imgs += portrait_candidates(r.text, r.url)
                todo += [p for p in pagination(r.text, r.url, s.url) if p not in done]
        new = sum(save_image(meta, seen, s.university, s.label, u, f"data/uni/{folder.name}") for u in dict.fromkeys(imgs))
        meta.to_csv(META, index=False)
        print(f"{s.university} / {s.label}: {len(done)} pages, {new} images", flush=True)


SIMONS = "https://www.simonsfoundation.org/mathematics-physical-sciences/simons-investigators/"


def simons():
    """Simons Investigators: Mathematics -> mathematician, Physics/Astrophysics -> physicist (CS and life sciences dropped)."""
    (DATA / "uni" / "simons").mkdir(parents=True, exist_ok=True)
    meta, todo, done, cards = load_meta(), [SIMONS], set(), []
    seen = set(meta.img_url)
    while todo and len(done) < 30:
        u = todo.pop(0)
        if u in done:
            continue
        done.add(u)
        r, _ = polite_get(u)
        if r is None:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("data-src") or img.get("src") or ""
            card = img.find_parent(["article", "li"]) or img.find_parent("div")
            text = " ".join(card.get_text(" ").split())[:200] if card else ""
            if "uploads" not in src or re.search(r"Computer Science|Life Sciences", text):
                continue
            if re.search(r"\bMathematics\b", text):
                cards.append((urljoin(r.url, src), "mathematician"))
            elif re.search(r"\b(Physics|Astrophysics)\b", text):
                cards.append((urljoin(r.url, src), "physicist"))
        todo += [h for a in soup.find_all("a", href=True) if (h := urljoin(r.url, a["href"])).startswith(SIMONS)
                 and re.search(r"/page/\d+/?$|[?&](paged|pg|page)=\d+", h) and h not in done]
    new = sum(save_image(meta, seen, "Simons", lab, u, "data/uni/simons") for u, lab in dict.fromkeys(cards))
    meta.to_csv(META, index=False)
    print(f"Simons: {len(cards)} cards, {new} new images")
