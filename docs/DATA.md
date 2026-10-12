# Data

No photos are stored in this repository.

## Sources

| Source | Command | What | License |
|---|---|---|---|
| Wikidata + Commons | `fetch` | humans with occupation physicist (Q169470) or mathematician (Q170790), born ≥ 1900, with a portrait | free licenses; attribution per file on Commons |
| Wikidata sub-fields | `fetch-subfields` | people listed only as e.g. theoretical physicist, topologist, logician | same |
| Commons categories | `fetch-extra`, `fetch-more` | more photos per person from their Commons category | same |
| Commons "depicts" | `fetch-depicts` | files whose structured data says they show the person | same |
| Wikipedia, all languages | `fetch-wiki` | photos shown in each person's articles (30,777 articles in 193 languages; Commons files only) | same |
| University departments | `scrape discover/probe/crawl` | staff photos from maths and physics department pages | used for local training only |
| Simons Investigators | `scrape simons` | maths and physics awardees, uniform headshots | used for local training only |

## Cleaning (`clean`)

Starting from 10,346 Wikidata people:

| Removed | Count | Why |
|---|---|---|
| not strict | ~3,860 | any occupation outside physics, maths or generic academia (astronaut, politician, engineer, computer scientist, economist, biophysicist, ...) |
| both fields | 345 | tagged physicist **and** mathematician |
| no usable face | ~800 | |
| not a photograph | ~110 | stamps, drawings, statues, paintings, plaques — CLIP zero-shot on the whole image and the face crop |
| shared photo | ~20 | one photo used for two people |

Old black-and-white photographs are kept. Excluded files are moved to `data/excluded/<reason>/`, never deleted;
add paths to `data/manual_exclude.txt` to remove more by hand. Contact sheets for review go to `data/review/`.

Result: **5,208 people** (2,688 mathematicians, 2,520 physicists).

## Extra photos of a person (`verify`)

An extra photo is used only if every check passes (`physormath/identity.py`):

| Check | Rejected (of 4,082) |
|---|---|
| copy of a photo we already have (re-upload, crop, recolouring: FaceNet ≥ 0.97, CLIP image ≥ 0.95 or perceptual hash), or another frame of the same photo shoot (FaceNet ≥ 0.90 and CLIP image ≥ 0.87) | 1,161 |
| someone else of similar size in the picture, so it is unclear who it shows | 755 |
| FaceNet similarity to the main portrait below 0.62 | 659 |
| 0.62–0.70 without a second confirmation from another strongly matching photo of them | 135 |
| not a photograph | 92 |
| another person's portrait is clearly more similar (lookalike) | 63 |
| looked at by eye: not certain (big age gap, tiny crop, stylised image, a scanned document) or looks the same | 63 |
| man/woman contradicts the main portrait | 6 |

Kept: **1,148 photos of 702 people**. Different people pass the 0.70 FaceNet bar in 0.03% of pairs; every accepted
match below 0.78, and every photo the game shows, was also checked by eye (contact sheets in `outputs/identity/`; rejects in `data/identity_reject.txt`).

## Scraping policy

- `robots.txt` is respected, including `Crawl-delay` (minimum 2 s between requests to a host).
- Sites that block automated access (403/429, bot protection) are skipped, never worked around.
- No names are stored for scraped photos.
- Scraped photos are used only to train locally; they are never committed or shown in the game.
- Only universities where **both** departments are available are used, balanced per university —
  otherwise the model learns a website's photo style instead of the field.
