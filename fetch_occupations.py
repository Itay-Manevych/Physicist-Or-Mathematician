"""Fetch all occupations + academic signals for every person in data/faces.csv (for label cleanup)."""
import time
import requests
import pandas as pd
from tqdm import tqdm
from fetch_data import HEADERS, SPARQL


def main():
    qids = pd.read_csv("data/faces.csv").qid.unique().tolist()
    rows = []
    for i in tqdm(range(0, len(qids), 300)):
        values = " ".join(f"wd:{q}" for q in qids[i:i + 300])
        q = f"""
        SELECT ?p (GROUP_CONCAT(DISTINCT ?occLabel; separator="|") AS ?occs)
               (COUNT(DISTINCT ?adv) AS ?n_adv) (COUNT(DISTINCT ?emp) AS ?n_uni) (SAMPLE(?cat) AS ?commonscat) WHERE {{
          VALUES ?p {{ {values} }}
          OPTIONAL {{ ?p wdt:P106 ?occ. ?occ rdfs:label ?occLabel. FILTER(LANG(?occLabel) = "en") }}
          OPTIONAL {{ ?p wdt:P184 ?adv. }}
          OPTIONAL {{ ?p wdt:P108 ?emp. ?emp wdt:P31/wdt:P279* wd:Q3918. }}
          OPTIONAL {{ ?p wdt:P373 ?cat. }}
        }} GROUP BY ?p"""
        for attempt in range(4):
            r = requests.get(SPARQL, params={"query": q, "format": "json"}, headers=HEADERS, timeout=300)
            if r.ok:
                break
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        for b in r.json()["results"]["bindings"]:
            rows.append({
                "qid": b["p"]["value"].rsplit("/", 1)[1],
                "occupations": b.get("occs", {}).get("value", ""),
                "has_advisor": int(b["n_adv"]["value"]) > 0,
                "uni_employer": int(b["n_uni"]["value"]) > 0,
                "commons_category": b.get("commonscat", {}).get("value", ""),
            })
        time.sleep(1)
    pd.DataFrame(rows).to_csv("data/occupations.csv", index=False)
    print(len(rows), "people")


if __name__ == "__main__":
    main()
