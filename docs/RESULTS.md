# Results

All numbers are for distinguishing physicists from mathematicians from a single photo unless stated.
AUC: 0.5 = coin flip, 1.0 = perfect.

## Current model

Logistic regression on embeddings of the face and the whole photo from two ~2B-parameter vision transformers,
PE-Core bigG/14 (448 px) and MetaCLIP-2 Worldwide ViT-bigG/14 (378 px), computed once on a free Kaggle GPU. Training also uses photos the test never sees: non-strict physicists/mathematicians
("loose", 3,859 photos), people born 1850–1899 (1,804) and strictly verified extra photos of training people.

**Evaluation protocol.** 20% of people are held out as a fixed test set (`outputs/split.csv`), split by person:
none of a test person's photos are used in training. Regularisation and the decision threshold are chosen by 5-fold
grouped cross-validation on the training set only, preferring the simplest model within 0.005 CV AUC of the best.
The test set is scored once.

| | AUC | Accuracy | n |
|---|---|---|---|
| Train | 0.849 | 75.4% | 4,365 |
| CV on train | 0.818 | | |
| **Test** | **0.833** | **74.3%** | 1,092 |
| Test, judging from every verified photo of the person | 0.839 | 74.6% | 1,092 |

Accuracy at one cutoff moves by ±1.5 points between retrains (72–75%) and is a blunt measure: the bigG models raised
AUC by 0.020 (paired bootstrap 95% CI 0.007–0.032) and cut log loss from 0.522 to 0.495, but accuracy at the cutoff
stayed at ~74.5%. The gain is in ranking and confidence.

**Overfitting.** None: train and test accuracy are 1.1 points apart, and CV agrees with the test score.

**Confidence.** Accuracy on the test people the model is most sure about: top 50% → 89%, top 25% → 98%, top 10% → 100%.
`predict` says "can't tell" below the median confidence.

**Several photos of one person.** For the 82 test people with verified extra photos, averaging over all their photos
lifts AUC from 0.83 to 0.90 and accuracy from 73.2% to 76.8%.

**More data.** Test AUC with 10% / 25% / 50% / 100% of the training data: 0.787 / 0.810 / 0.819 / 0.833.

## What the model looks at

| Input | AUC |
|---|---|
| Face shape only (FaceNet face-recognition embedding) | 0.61 |
| Photo style + birth year + gender (no image content) | 0.69 |
| Face (3 CLIP-type models) | 0.77 |
| Upper body | 0.76 |
| Whole photo | 0.78 |
| Face + whole photo | 0.78 |

Face shape barely beats chance; most of the signal is presentation and context. A CLIP concept probe agrees:

- leans **physicist**: confident, suit and tie, serious, older, candid snapshot
- leans **mathematician**: looks like a hippie, t-shirt, sweater, messy hair, long hair, in front of a blackboard, nerdy

## What helped, what didn't

| Change | Effect (AUC) |
|---|---|
| Strict labels: drop people with any non-physics/maths career (astronauts, politicians, engineers, ...) | **+0.05** (0.72 → 0.77; a same-size random subset stays at 0.70) |
| Whole photo instead of face only | +0.01 |
| Averaging several photos of the person | +0.03 |
| Removing paintings, drawings, statues, stamps | small, but they are not photos |
| Bigger backbones (CLIP-L → CLIP-H → SigLIP2 → all three) | +0.01 |
| MetaCLIP-2 Worldwide ViT-H/14 instead of the CLIP/SigLIP2 trio | +0.015 (0.788 → 0.803) |
| Training on non-strict ("loose") physicists/mathematicians too, scored only on strict ones | +0.003 |
| 378 px MetaCLIP-2, people born 1850–1899, CV-tuned decision threshold | +0.007 (→ 0.813, 73.1%) |
| Dropping unverifiable / duplicate extra photos (`verify`) | none (0.813 before and after) |
| PE-Core bigG + MetaCLIP-2 bigG (cloud GPU) instead of MetaCLIP-2 H + CLIP-H | **+0.020** (0.813 → 0.833) |
| SigLIP2 giant (1B) | none (worse than MetaCLIP-2 H on val) |
| DINOv2, EVA-02, LAION ViT-bigG next to MetaCLIP-2; SVM, MLP, stacking, standardisation | none |
| Fine-tuning CLIP ViT-L/14 or MetaCLIP-2 locally (LP-FT) | none to tiny (val 0.775 → 0.781 after 5.5 h) |

## University staff photos: the model doesn't transfer

3,116 photos from university maths/physics department pages and the Simons Investigators (only universities where both
departments were available, balanced per university). Each university held out in turn:

| University | photos | AUC |
|---|---|---|
| Stanford | 100 | 0.87 |
| Cornell | 98 | 0.73 |
| Yale | 30 | 0.73 |
| Bar-Ilan | 62 | 0.68 |
| Duke | 76 | 0.61 |
| Simons Investigators | 78 | 0.52 |
| Ohio State | 126 | 0.43 |
| Penn State | 350 | 0.43 |
| Weizmann | 576 | 0.40 |
| **weighted mean** | | **0.50** |

On department headshots the model is at chance: right at some universities, reversed at others. Within one university the
department's photographer, not the person, decides suit vs. casual, studio vs. candid, so the style cues the model learned
from Wikidata point the wrong way where the maths page happens to look "formal". Training on the other universities does
not help (leave-one-university-out mean 0.45). Adding the university photos to training moved the Wikidata test only from
AUC 0.788 / 70.1% to 0.790 / 71.2% with the earlier model (within noise), so they are not used.

This is the strongest evidence that the model reads photo presentation, not faces.

## Shortcuts found along the way

- **Astronauts.** Before label cleaning, the most confident "physicists" were astronauts in official NASA portraits.
- **Photo sources.** ~3% of mathematician photos come from the Oberwolfach collection; they were 10% of the
  model's confident "mathematician" calls.
- **Black and white.** Old photos skew mathematician; all embeddings average a grayscale view to blunt this.
- **Department photographers** (university data): each department has one photo style, so university photos are
  balanced per university and evaluated on held-out universities.

## Fun

People tagged as both physicist and mathematician (not used in training): John von Neumann scores 87% physicist;
David Ruelle, Jürg Fröhlich and Demetrios Christodoulou about 90% mathematician.
