# Fine-tuning on a free cloud GPU (Kaggle or Google Colab)

This PC's 6 GB GPU is too small to fine-tune the big image models well, so this runs on Kaggle (free, ~30 GPU h/week).
The notebook benchmarks four large vision transformers on our data, then fine-tunes the best one.

| Model | Notes |
|---|---|
| MetaCLIP-2 Worldwide ViT-H-378 | current production backbone (reference) |
| SigLIP2 giant (gopt) 384 | Google, ~1B params |
| MetaCLIP-2 Worldwide ViT-bigG-378 | ~1.8B params |
| PE-Core-bigG-448 | Meta Perception Encoder, ~2B params |

## Steps

1. **Package the data** (already done once): `python -m physormath kaggle pack` → `data/kaggle/pom-data.zip`
   (photos resized to 448 px + face crops + `meta.csv` with the fixed train / val / test split).
2. On kaggle.com: **Datasets → New Dataset**, upload `pom-data.zip`, name it `pom-data` (keep it private).
3. **Code → New Notebook → File → Import Notebook** → `kaggle/finetune_metaclip2.ipynb`.
4. In the notebook: **Add Input** → your `pom-data` dataset; **Settings → Accelerator: GPU T4 x2**;
   **Settings → Internet: On** (needs a phone-verified Kaggle account, to download the model weights).
5. **Save Version → Save & Run All (Commit)**. It runs in the background (~4–6 h); you can close the browser.
6. When it finishes, download the outputs (needs `~/.kaggle/kaggle.json` from kaggle.com → Settings → API):
   `kaggle kernels output <user>/<notebook> -p data/kaggle/out`
7. `python -m physormath kaggle import` → turns `emb_<model>_<view>.npy` into `data/feats_*.npy` for every source, so
   `train` can use them (`MODEL_FEATURES` in `physormath/config.py`). If the fine-tune stage also produced
   `predictions.csv`, `python -m physormath kaggle blend` scores it against the production model.

## Results (October 2026, val people, frozen embeddings + logistic regression)

| Model | Face | Whole photo | Face + photo | Minutes on a T4 |
|---|---|---|---|---|
| MetaCLIP-2 Worldwide H-378 (previous production) | 0.769 | 0.809 | 0.803 | 36 |
| SigLIP2 giant 384 | 0.760 | 0.790 | 0.790 | 45 |
| MetaCLIP-2 Worldwide bigG-378 | 0.782 | 0.824 | 0.818 | 97 |
| PE-Core bigG-448 | 0.795 | 0.831 | 0.825 | 141 |

PE-Core bigG + MetaCLIP-2 bigG together are now the production features (test AUC 0.813 → 0.833). The fine-tune stage
did not finish on a T4: a 2B-parameter model is too big to fine-tune there.

## Google Colab instead (no phone verification needed)

1. Upload `data/kaggle/pom-data.zip` to the top level of your Google Drive (My Drive).
2. Open https://colab.research.google.com → **File → Upload notebook** → `kaggle/finetune_colab.ipynb`.
3. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**; allow Drive access; keep the tab open (~4–5 h).
4. Results land in **My Drive / pom-results** as they are produced. If Colab disconnects, **Run all** again
   (finished models are skipped). Download `predictions.csv` and `benchmark.json` into `data/kaggle/`.
5. `python -m physormath kaggle blend`.

The Colab version skips the reference model (MetaCLIP-2 H-378): the blend step scores the production model on the
same val people locally.

## Rules

- Models, views, epochs and the blend weight are chosen on the **val** people (15% of the training people).
- The test people's labels are in `meta.csv` so the local blend can score them, but the notebook never uses them.
