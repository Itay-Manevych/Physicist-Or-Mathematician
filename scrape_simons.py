"""Simons Investigators (mathematics + physics, uniform headshots) -> data/uni.csv rows with university='Simons'.

Each card lists the investigator's field; we keep Mathematics -> mathematician, Physics/Astrophysics -> physicist,
and drop Theoretical Computer Science and interdisciplinary (Life Sciences) awards.
"""
import os
import re
import hashlib
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from uni_scrape import polite_get

BASE = "https://www.simonsfoundation.org/mathematics-physical-sciences/simons-investigators/"
os.makedirs("data/uni/simons", exist_ok=True)


def field(card_text):
    if re.search(r"Computer Science|Life Sciences", card_text):
        return None
    if re.search(r"\bMathematics\b", card_text):
        return "mathematician"
    if re.search(r"\b(Physics|Astrophysics)\b", card_text):
        return "physicist"
    return None


def main():
    meta_path = "data/uni.csv"
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else pd.DataFrame(columns=["university", "label", "file", "img_url"])
    seen = set(meta.img_url)
    todo, done, cards = [BASE], set(), []
    while todo and len(done) < 30:
        u = todo.pop(0)
        if u in done:
            continue
        done.add(u)
        r, _ = polite_get(u)
        if r is None:
            continue
        s = BeautifulSoup(r.text, "html.parser")
        for img in s.find_all("img"):
            src = img.get("data-src") or img.get("src") or ""
            if "uploads" not in src:
                continue
            card = img.find_parent(["article", "li"]) or img.find_parent("div")
            head = " ".join(card.get_text(" ").split())[:200] if card else ""
            lab = field(head)
            if lab:
                cards.append((urljoin(r.url, src), lab))
        for a in s.find_all("a", href=True):
            h = urljoin(r.url, a["href"])
            if h.startswith(BASE) and re.search(r"/page/\d+/?$|[?&](paged|pg|page)=\d+", h) and h not in done:
                todo.append(h)
    new = 0
    for img_url, lab in dict.fromkeys(cards):
        if img_url in seen:
            continue
        r, _ = polite_get(img_url)
        if r is None or not r.headers.get("content-type", "").startswith("image"):
            continue
        f = f"data/uni/simons/{hashlib.md5(img_url.encode()).hexdigest()}.jpg"
        with open(f, "wb") as fh:
            fh.write(r.content)
        seen.add(img_url)
        meta.loc[len(meta)] = ["Simons", lab, f, img_url]
        new += 1
    meta.to_csv(meta_path, index=False)
    print(f"Simons: {len(done)} pages, {len(cards)} cards, {new} new images; labels:",
          meta[meta.university == "Simons"].label.value_counts().to_dict())


if __name__ == "__main__":
    main()
