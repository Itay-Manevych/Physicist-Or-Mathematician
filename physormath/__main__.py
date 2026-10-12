"""python -m physormath <step> ...   (run from the repository root)

Data
  fetch                       Wikidata physicists/mathematicians -> data/people.csv, data/images/
  fetch-subfields             people listed only under a sub-field -> data/new/
  fetch-extra                 more Commons photos per clean person -> data/extra/
  fetch-depicts               photos tagged "depicts <person>" on Commons -> data/depicts/
  fetch-wiki                  photos in each person's articles on every language Wikipedia -> data/articles/
  scrape {discover,probe,crawl,simons}   university / Simons headshots -> data/uni/
Processing
  process <images.csv> [tag]  face crops + all embeddings -> data/faces<tag>.csv, data/feats_*<tag>.npy
  clean [--apply]             strict, photos-only Wikidata set -> data/clean.csv
  verify                      strict same-person / not-a-copy checks on extra photos -> data/verified.csv
Model
  train                       fixed train/test split, CV on train, report -> outputs/
  predict <photo> [...]       judge one person from one or more photos
  game                        build the game's data from held-out test people -> game/
Guess the department (separate mode)
  dept {fetch,train,game}     7 departments: data/dept/, outputs/dept/, game/dept/
Research
  experiment {classifiers,features,backbones,extract,ensembles}   model experiments -> outputs/experiments.csv
  finetune <backbone> <face|full>                 LP-FT a vision transformer + blend with the probes (local GPU)
  kaggle {pack,import,blend}                      embed / fine-tune giant models on a free Kaggle GPU (see kaggle/)
"""
import argparse
import os

from .config import ROOT


def main():
    os.chdir(ROOT)  # data paths in the CSVs are relative to the repo root
    ap = argparse.ArgumentParser(prog="python -m physormath", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["fetch", "fetch-subfields", "fetch-extra", "fetch-depicts", "fetch-wiki", "fetch-more", "fetch-older", "verify", "train", "game"]:
        sub.add_parser(name)
    sub.add_parser("scrape").add_argument("stage", choices=["discover", "probe", "crawl", "simons"])
    p = sub.add_parser("process")
    p.add_argument("csv")
    p.add_argument("tag", nargs="?", default="")
    sub.add_parser("clean").add_argument("--apply", action="store_true")
    sub.add_parser("predict").add_argument("photos", nargs="+")
    sub.add_parser("dept").add_argument("stage", choices=["fetch", "train", "game"])
    sub.add_parser("kaggle").add_argument("stage", choices=["pack", "import", "blend"])
    ft = sub.add_parser("finetune")
    ft.add_argument("backbone")
    ft.add_argument("view", choices=["face", "full"])
    sub.add_parser("experiment").add_argument("suite", choices=["classifiers", "features", "backbones", "extract", "ensembles"])
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
    elif a.cmd == "fetch-older":
        from .wikidata import fetch_older
        fetch_older()
    elif a.cmd == "fetch-more":
        from .wikidata import fetch_more_photos
        fetch_more_photos()
    elif a.cmd == "fetch-wiki":
        from .wikidata import fetch_wikipedia_photos
        fetch_wikipedia_photos()
    elif a.cmd == "fetch-depicts":
        from .wikidata import fetch_depicts
        fetch_depicts()
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
    elif a.cmd == "verify":
        from .identity import verify
        verify()
    elif a.cmd == "train":
        from .train import main as train
        train()
    elif a.cmd == "game":
        from .game import build
        build()
    elif a.cmd == "kaggle":
        from . import kaggle
        {"pack": kaggle.pack, "import": kaggle.import_embeddings, "blend": kaggle.blend}[a.stage]()
    elif a.cmd == "dept":
        from . import departments
        getattr(departments, a.stage)()
    elif a.cmd == "finetune":
        from .finetune import finetune
        finetune(a.backbone, a.view)
    elif a.cmd == "experiment":
        from . import experiments
        getattr(experiments, a.suite)()
    elif a.cmd == "predict":
        from .predict import judge
        judge(a.photos)


if __name__ == "__main__":
    main()
