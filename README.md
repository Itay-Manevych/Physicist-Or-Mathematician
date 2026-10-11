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

## The game

`game/` is a retro fighting game on held-out test people only (the model never trained on them):

- **Vs the machine**: 10 rounds against the model; online leaderboard.
- **Daily challenge**: the same 10 fighters for everyone each day (`game/daily.js`, shared with the API), one try, daily leaderboard, streaks.
- **Live duel**: create a room, send the link, fight a friend in real time (15 s per round, answers hidden until both lock in, rematch).
- After each guess the reveal shows what the machine noticed (CLIP concept probe), plus shareable scorecards and an OG link preview.

Running it:

- **Deploy:** Vercel serves `game/` and the functions in `api/` (`vercel.json`, no build step).
- **Storage:** leaderboards and duel rooms use Upstash Redis — add the free **Upstash** integration in the Vercel project
  (Storage → Upstash → Redis → connect) and redeploy. Without it, vs-machine scores stay on each device and duels are off.
- **Local:** `node dev/server.js` → http://localhost:8137 (in-memory store).
- Photo credits: [game/CREDITS.md](game/CREDITS.md).

## Layout

```
physormath/   the code: one module per step, run as `python -m physormath <step>`
sources/      seed lists for the university scraper
game/         retro "human vs. machine" fighting game (static site)
api/          serverless functions: leaderboards, live duel rooms
dev/          local server for the game + API
docs/         results and data notes
data/         photos, crops, embeddings   (not in git)
outputs/      models, predictions, reports (not in git)
```

It's a joke model. Don't use it to judge real people for anything that matters.
