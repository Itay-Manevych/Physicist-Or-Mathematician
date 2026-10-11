"""python -m physormath <step> ...   (run from the repository root)

Data
  fetch                       Wikidata physicists/mathematicians -> data/people.csv, data/images/
  fetch-subfields             people listed only under a sub-field -> data/new/
  fetch-extra                 more Commons photos per clean person -> data/extra/
  scrape {discover,probe,crawl,simons}   university / Simons headshots -> data/uni/
Processing
  process <images.csv> [tag]  face crops + all embeddings -> data/faces<tag>.csv, data/feats_*<tag>.npy
  clean [--apply]             strict, photos-only Wikidata set -> data/clean.csv
Model
  train                       fixed train/test split, CV on train, report -> outputs/
  predict <photo> [...]       judge one person from one or more photos
  game                        build the game's data from held-out test people -> game/
Guess the department (separate mode)
  dept {fetch,train,game}     7 departments: data/dept/, outputs/dept/, game/dept/
"""
import argparse
import os

from .config import ROOT


def main():
    os.chdir(ROOT)  # data paths in the CSVs are relative to the repo root
    ap = argparse.ArgumentParser(prog="python -m physormath", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["fetch", "fetch-subfields", "fetch-extra", "train", "game"]:
        sub.add_parser(name)
    sub.add_parser("scrape").add_argument("stage", choices=["discover", "probe", "crawl", "simons"])
    p = sub.add_parser("process")
    p.add_argument("csv")
    p.add_argument("tag", nargs="?", default="")
    sub.add_parser("clean").add_argument("--apply", action="store_true")
    sub.add_parser("predict").add_argument("photos", nargs="+")
    sub.add_parser("dept").add_argument("stage", choices=["fetch", "train", "game"])
    a = ap.parse_args()

    if a.cmd == "fetch":
        from .wikidata import fetch_people
        fetch_people()
    elif a.cmd == "fetch-subfields":
        from .wikidata import fetch_subfields
        fetch_subfields()
    elif a.cmd == "fetch-extra":
        from .wikidata import fetch_extra
        fetch_extra()
    elif a.cmd == "scrape":
        from . import scrape
        getattr(scrape, a.stage)()
    elif a.cmd == "process":
        from .embed import embed_context, embed_faces
        from .faces import crop_faces
        faces_csv = f"data/faces{a.tag}.csv"
        crop_faces(a.csv, faces_csv)
        embed_faces(faces_csv, a.tag)
        embed_context(faces_csv, a.tag)
    elif a.cmd == "clean":
        from .clean import clean
        clean(a.apply)
    elif a.cmd == "train":
        from .train import main as train
        train()
    elif a.cmd == "game":
        from .game import build
        build()
    elif a.cmd == "dept":
        from . import departments
        getattr(departments, a.stage)()
    elif a.cmd == "predict":
        from .predict import judge
        judge(a.photos)


if __name__ == "__main__":
    main()
