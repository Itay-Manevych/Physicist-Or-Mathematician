# Physicist or Mathematician?

A long-running joke, settled by a model: given a photo, is this person a physicist or a mathematician?

## Results

Held-out **test set** of 1,092 people the model never saw during training:

| | AUC | Accuracy |
|---|---|---|
| All test people (one photo each) | 0.833 | 74.3% |
| Its most confident 50% | — | 89% |
| Its most confident 25% | — | 98% |
| Its most confident 10% | — | 100% |
| Using every verified photo of the person | 0.839 | 74.6% |

Train accuracy is 75.4% and cross-validated AUC on the training set is 0.818, so there is no overfitting.
What the model picks up is mostly presentation (clothes, hair, expression, photo setting), not face shape:
a face-recognition embedding alone scores only 0.61. Details and every experiment: [docs/RESULTS.md](docs/RESULTS.md).

## How it works

**Data.** People on Wikidata with occupation physicist or mathematician, born 1900 or later, with a portrait on
Wikimedia Commons. Labels are strict: anyone with an occupation outside physics, maths or generic academia
(astronaut, engineer, politician, ...) or listed as both is left out, and paintings, drawings, statues and stamps are
removed with a CLIP zero-shot check. That leaves 5,208 people. Training also uses photos the test never sees:
non-strict physicists/mathematicians, people born 1850–1899, and extra photos of training people that passed a strict
same-person and no-duplicate check (`verify`, see [docs/DATA.md](docs/DATA.md)).

**Embeddings.** No network is trained from scratch. Each photo goes through two frozen pretrained vision
transformers, each seeing the face crop (MTCNN, 1.6× the face box) and the whole photo padded to a square:

| Model | Params | Input | Size |
|---|---|---|---|
| PE-Core bigG/14 (Meta Perception Encoder) | ~2B | 448 px | 1280 |
| MetaCLIP-2 Worldwide ViT-bigG/14 | ~2B | 378 px | 1280 |

They are too big for this PC's 6 GB GPU, so all photos were embedded once on a free Kaggle GPU (`kaggle/`). Chosen by
grouped CV over CLIP-L/H, SigLIP2 (incl. the 1B giant), EVA-02, DINOv2 and MetaCLIP-2 H: the two bigG models together
lifted test AUC from 0.813 to 0.833 (paired bootstrap 95% CI +0.007 to +0.032).

**Classifier.** Logistic regression on the 5,120 concatenated features. Regularisation strength and the decision
threshold come from 5-fold cross-validation on the training set, grouped by person so no one is in two folds.
A person with several photos is judged by the mean of their per-photo scores.

**Evaluation.** A fixed 20% of people (`outputs/split.csv`), split by person and stratified by label, is the test set.
None of a test person's photos are used for training or model selection, and the test set is scored once at the end.
The game only shows test people. On university department headshots the model is at chance (AUC 0.50): each
department's photographer sets the style, and style is what the model reads.

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
python -m physormath verify                        # extra photos: same person for sure, no duplicates
python -m physormath train                         # fixed train/test split, report in outputs/
python -m physormath game                          # game data from held-out test people -> game/
```

Optional extra data: `fetch-subfields`, `fetch-older`, `fetch-extra`, `fetch-depicts`, `fetch-more`, `fetch-wiki`
(each followed by `process <csv> <tag>`). Cloud fine-tuning of bigger models: [kaggle/README.md](kaggle/README.md).
Sources and data policy: [docs/DATA.md](docs/DATA.md).

## The game

`game/` is a retro fighting game on held-out test people only (the model never trained on them):

- **Vs the machine**: 10 rounds against the model; online leaderboard. Fighters with extra verified photos show up to 3.
- **Daily challenge**: the same 10 fighters for everyone each day (`game/daily.js`, shared with the API), one try, daily leaderboard, streaks.
- **Live duel**: create a room, send the link, fight a friend in real time (15 s per round, answers hidden until both lock in, rematch).
- **Party mode** (Jackbox-style): the host screen goes on a TV with a room code + QR; everyone answers on their phone;
  each reveal shows who picked what, points are 100 + a speed bonus, the machine plays too, and the game ends on a podium.
- After each guess the reveal shows what the machine noticed (CLIP concept probe), plus shareable scorecards and an OG link preview.

Photo credits: [game/CREDITS.md](game/CREDITS.md).

## Layout

```
physormath/   the code: one module per step, run as `python -m physormath <step>`
sources/      seed lists for the university scraper
game/         retro "human vs. machine" fighting game (static site)
api/          serverless functions: leaderboards, live duel rooms, party rooms
dev/          local server for the game + API
docs/         results and data notes
kaggle/       notebooks for fine-tuning on a free cloud GPU
data/         photos, crops, embeddings   (not in git)
outputs/      models, predictions, reports (not in git)
```

It's a joke model. Don't use it to judge real people for anything that matters.
