# Results

All numbers are for distinguishing physicists from mathematicians from a single photo unless stated.
AUC: 0.5 = coin flip, 1.0 = perfect.

## Current model

Logistic regression on face embeddings (CLIP ViT-L/14, CLIP ViT-H/14, SigLIP2) + a whole-photo embedding (CLIP ViT-H/14).

**Evaluation protocol.** 20% of people are held out as a fixed test set (`outputs/split.csv`), split by person:
none of a test person's photos are used in training. Regularisation is chosen by 5-fold grouped cross-validation
on the training set only, preferring the simplest model within 0.005 CV AUC of the best. The test set is scored once.

| | AUC | Accuracy | n |
|---|---|---|---|
| Train | 0.826 | 73.8% | 4,366 |
| CV on train | 0.794 | | |
| **Test** | **0.788** | **70.1%** | 1,092 |

**Overfitting.** Mild: a 3.7-point train-test accuracy gap, and CV agrees with the test score. With weaker
regularisation the training accuracy rises to 77% but test accuracy stays at 70% — extra capacity only memorises.

**Confidence.** Accuracy on the test people the model is most sure about: top 50% → 84%, top 25% → 94%, top 10% → 99%.
`predict` says "can't tell" below the median confidence.

**Several photos of one person.** For 192 test people with extra Commons photos, averaging the prediction over all
their photos lifts AUC from 0.86 to 0.89 (these people score higher than average to begin with).

**More data.** Test AUC with 10% / 25% / 50% / 100% of the training data: 0.749 / 0.773 / 0.777 / 0.788.
Mostly flat — more people help little.

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
| Fine-tuning the last 6 blocks of CLIP ViT-L/14 | none (0.735 vs. 0.732 for a linear probe; overfits after 2 epochs) |
| More people / more photos per person in training | none |

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
AUC 0.788 / 70.1% to 0.790 / 71.2% (within noise; overfitting gap 3.7 → 1.7 points).

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
