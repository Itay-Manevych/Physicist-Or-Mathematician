# Physicist or Mathematician?

A long-running joke, settled by a model: given a photo, is this person a physicist or a mathematician?

## Results

Held-out **test set** of 1,092 people the model never saw during training:

| | AUC | Accuracy |
|---|---|---|
| All test people | 0.79 | 70% |
| Its most confident 25% | — | 94% |
| Several photos of one person, averaged | 0.89 | 77% |

Train accuracy is 74%, so overfitting is mild, and cross-validation on the training set (AUC 0.79) agrees with the test score.
What the model picks up is mostly presentation — clothes, hair, expression, photo setting — not face shape.
Details and every experiment: [docs/RESULTS.md](docs/RESULTS.md).

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip install facenet-pytorch --no-deps

python -m physormath predict friend1.jpg friend2.jpg friend3.jpg   # several photos of ONE person
```

`predict` needs a trained model in `outputs/` — run the pipeline below first.

## Pipeline

```bash
python -m physormath fetch                         # Wikidata physicists & mathematicians + Commons photos
python -m physormath process data/people.csv       # face crops + embeddings
python -m physormath clean --apply                 # strict labels, real photos only
python -m physormath train                         # fixed train/test split, report in outputs/
python -m physormath game                          # game data from held-out test people -> game/
```

Optional extra data: `fetch-subfields`, `fetch-extra`, `scrape {discover,probe,crawl,simons}` (each followed by `process <csv> <tag>`).
Sources and data policy: [docs/DATA.md](docs/DATA.md).

## Layout

```
physormath/   the code: one module per step, run as `python -m physormath <step>`
sources/      seed lists for the university scraper
game/         retro "human vs. machine" fighting game (static site)
docs/         results and data notes
data/         photos, crops, embeddings   (not in git)
outputs/      models, predictions, reports (not in git)
```

It's a joke model. Don't use it to judge real people for anything that matters.
