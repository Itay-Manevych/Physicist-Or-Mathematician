# Physicist or Mathematician?

A long-running joke turned into a model: given a photo, is this person a physicist or a mathematician?
Includes a browser game where you guess and compare yourself against the AI.

> Work in progress — data collection and the game are still evolving.

## Current results (clean dataset, 5,247 people)

| What the model sees | CV AUC | Accuracy |
|---|---|---|
| Face shape only (FaceNet identity embedding) | 0.61 | 58% |
| Face (CLIP ViT-L/14 + ViT-H/14 + SigLIP2) | 0.77 | 69–70% |
| Whole photo (CLIP ViT-H/14) | 0.78 | 70% |
| **Face + whole photo** | **0.78** | **70%** |

When the model only answers its most confident cases: top 50% → 83%, top 25% → 93%, top 10% → 98% accuracy.

Findings so far:
- **Clean labels mattered most.** Restricting to people whose only occupations are physics/maths/academic raised AUC from 0.72 to 0.77 (a same-size random subset stays at 0.70).
- **More people doesn't help much** — the learning curve flattens after ~4k people.
- **Bigger backbones / fine-tuning don't help** — CLIP-L, CLIP-H, SigLIP2 and a partially fine-tuned CLIP all land at ~0.72–0.73 on the original data.
- **It's mostly not the face itself** — a face-recognition embedding barely beats chance; CLIP picks up clothes, hair, expression and photo context. Photo style + birth year + gender alone reach AUC 0.69.
- CLIP concept probe: physicists lean *confident, suit and tie, serious*; mathematicians lean *hippie, t-shirt, sweater, messy hair, blackboard*.

## Pipeline

| Step | Script | |
|---|---|---|
| 1 | `fetch_data.py` | Wikidata: humans with occupation physicist / mathematician, born ≥ 1900, with a Commons portrait |
| 2 | `fetch_occupations.py`, `labels.py` | All occupations per person → **strict** allowlist (no astronauts, politicians, engineers, …) |
| 3 | `embed.py`, `embed_multi.py`, `embed_context.py` | MTCNN face crop → CLIP / SigLIP / FaceNet embeddings (with mirror + grayscale test-time augmentation), whole-photo and upper-body embeddings |
| 4 | `clean_dataset.py` | Removes non-photos (paintings, drawings, statues, stamps, plaques) with CLIP zero-shot on the whole image and the face crop; moves excluded files to `data/excluded/<reason>/` |
| 5 | `train_clean.py` | Logistic-regression probes, CV + held-out test, confound baselines |
| 6 | `fetch_more.py`, `fetch_extra.py` | More data: Wikidata sub-field occupations; extra Commons photos per person (identity-verified with FaceNet) |
| 7 | `uni_scrape.py`, `scrape_simons.py` | University department pages + Simons Investigators (see data policy) |
| 8 | `train_all.py` | Combines all sources; evaluates by person and leave-one-university-out |

`predict.py photo.jpg` scores faces in a photo. `game/` is a static site (guess, then compare with the AI).

## Data policy

- No images are committed. Wikidata/Commons photos are freely licensed but need per-file attribution; the game links each photo to its Commons page.
- The university scraper respects `robots.txt` (including `Crawl-delay`), never retries 403/429 blocks, stores no names, and those photos are used only for local training — never redistributed or shown in the game.
- Sources must contain **both** physicists and mathematicians photographed the same way, otherwise a model learns the source instead of the person. University data is balanced per university and evaluated on held-out universities.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install open_clip_torch scikit-learn pandas requests pillow tqdm beautifulsoup4 joblib
.venv\Scripts\pip install facenet-pytorch --no-deps
```

## Caveat

It's a joke model. ~70% accuracy is a party trick, not a judgment about anyone — don't use it for decisions about real people.
