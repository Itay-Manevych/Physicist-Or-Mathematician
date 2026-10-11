# Data

No photos are stored in this repository.

## Sources

| Source | Command | What | License |
|---|---|---|---|
| Wikidata + Commons | `fetch` | humans with occupation physicist (Q169470) or mathematician (Q170790), born ≥ 1900, with a portrait | free licenses; attribution per file on Commons |
| Wikidata sub-fields | `fetch-subfields` | people listed only as e.g. theoretical physicist, topologist, logician | same |
| Commons categories | `fetch-extra` | up to 4 more photos per person; kept only if FaceNet confirms it is the same person | same |
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

Result: **5,209 people** (2,688 mathematicians, 2,521 physicists).

## Scraping policy

- `robots.txt` is respected, including `Crawl-delay` (minimum 2 s between requests to a host).
- Sites that block automated access (403/429, bot protection) are skipped, never worked around.
- No names are stored for scraped photos.
- Scraped photos are used only to train locally; they are never committed or shown in the game.
- Only universities where **both** departments are available are used, balanced per university —
  otherwise the model learns a website's photo style instead of the field.
