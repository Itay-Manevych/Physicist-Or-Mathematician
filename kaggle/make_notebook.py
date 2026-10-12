"""Generates kaggle/finetune_metaclip2.ipynb (kept as a script so the notebook is reviewable in git)."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("""# Physicist or Mathematician? — big-model benchmark + fine-tune

**Before running:** Settings → Accelerator **GPU T4 x2**, **Internet on**, and add the `pom-data` dataset as input.
Then *Save Version → Save & Run All* (runs in the background, ~4–6 h). Outputs (in /kaggle/working):
`benchmark.json`, `emb_<model>_<view>.npy`, `finetune.json` and `predictions.csv` (fine-tuned model, val + test rows).

Models and views are compared on the **val** people (15% of the training people); test labels are never used for choices.""")

code("""!pip install -q open_clip_torch==3.3.0 timm
import os, glob, json, time, shutil, numpy as np, pandas as pd, torch, torch.nn as nn, open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
os.environ["HF_HUB_CACHE"] = "/tmp/hf"
DATA = os.path.dirname(glob.glob("/kaggle/input/**/meta.csv", recursive=True)[0])
OUT, DEV = "/kaggle/working", "cuda"
meta = pd.read_csv(f"{DATA}/meta.csv")
y = meta.y.values
TR, VA = (meta.role == "train").values, ((meta.role == "val") & meta.main).values
print(meta.role.value_counts().to_dict(), torch.cuda.get_device_name(0))""")

code("""STAGE = "both"        # "benchmark", "finetune", or "both"
CANDIDATES = {        # name -> (open_clip architecture, pretrained tag)
    "mc2H378": ("ViT-H-14-worldwide-378", "metaclip2_worldwide"),       # current best (reference)
    "siglip2_gopt": ("ViT-gopt-16-SigLIP2-384", "webli"),
    "mc2bigG378": ("ViT-bigG-14-worldwide-378", "metaclip2_worldwide"),
    "pe_bigG": ("PE-Core-bigG-14-448", "meta"),
}
FINETUNE = "auto"     # "auto" = best model on val, or one of the names above
EPOCHS, BATCH, ACCUM = 2, 16, 2""")

code('''def pad_square(img):
    s = max(img.size)
    out = Image.new("RGB", (s, s), (128, 128, 128))
    out.paste(img, ((s - img.size[0]) // 2, (s - img.size[1]) // 2))
    return out


class Imgs(Dataset):
    def __init__(self, paths, tf, square, labels=None):
        self.paths, self.tf, self.square, self.labels = list(paths), tf, square, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(f"{DATA}/{self.paths[i]}").convert("RGB")
        x = self.tf(pad_square(img) if self.square else img)
        return (x, float(self.labels[i])) if self.labels is not None else x


def load_visual(arch, pretrained, dtype="fp16"):
    model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device="cpu", precision=dtype)
    vis = model.visual
    del model
    return vis, pre


@torch.no_grad()
def embed(vis, pre, view):
    dl = DataLoader(Imgs(meta[view], pre, square=(view == "image")), batch_size=32, num_workers=4)
    out = []
    for x in tqdm(dl, desc=view):
        with torch.autocast(DEV, dtype=torch.float16):
            f = vis(x.to(DEV).half()).float()
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def val_auc(X):
    """Best val AUC of a logistic regression trained on the train people (and its C)."""
    X = X.astype(np.float32)
    best = (0.0, None)
    for C in [0.3, 1, 3]:
        m = LogisticRegression(max_iter=4000, C=C).fit(X[TR], y[TR])
        auc = roc_auc_score(y[VA], m.decision_function(X[VA]))
        if auc > best[0]:
            best = (auc, C)
    return best''')

code('''# ---------------- stage 1: benchmark (frozen embeddings + logistic regression, scored on val people) ----------------
bench = json.load(open(f"{OUT}/benchmark.json")) if os.path.exists(f"{OUT}/benchmark.json") else {}
if STAGE in ("benchmark", "both"):
    for name, (arch, pre_tag) in CANDIDATES.items():
        if name in bench:
            continue
        t0 = time.time()
        try:
            vis, pre = load_visual(arch, pre_tag)
            vis = vis.to(DEV).eval()
            E = {v: embed(vis, pre, v) for v in ("face", "image")}
            del vis
        except Exception as e:
            print(f"{name}: failed ({e})")
            continue
        finally:
            torch.cuda.empty_cache()
            shutil.rmtree("/tmp/hf", ignore_errors=True)
        for v, X in E.items():
            np.save(f"{OUT}/emb_{name}_{v}.npy", X)
        bench[name] = {"face": val_auc(E["face"]), "image": val_auc(E["image"]),
                       "face+image": val_auc(np.hstack([E["face"], E["image"]])),
                       "minutes": round((time.time() - t0) / 60, 1)}
        print(name, bench[name])
        json.dump(bench, open(f"{OUT}/benchmark.json", "w"), indent=2)
    print(pd.DataFrame({k: {v: r[v][0] for v in ("face", "image", "face+image")} for k, r in bench.items()}).T.round(4))''')

code('''# ---------------- stage 2: fine-tune the best model's best view (LP-FT) ----------------
def find_blocks(vis):
    """The longest ModuleList of identical transformer blocks inside the image tower."""
    best = []
    for _, m in vis.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) > len(best) and len({type(c) for c in m}) == 1:
            best = list(m)
    return best


class Net(nn.Module):
    def __init__(self, vis, head):
        super().__init__()
        self.vis, self.head = vis, head

    def forward(self, x):
        f = self.vis(x)
        return self.head(f / f.norm(dim=-1, keepdim=True)).squeeze(-1)


@torch.no_grad()
def predict(model, paths, tf, square):
    model.eval()
    out = []
    for x in DataLoader(Imgs(paths, tf, square), batch_size=32, num_workers=4):
        x = x.to(DEV)
        with torch.autocast(DEV, dtype=torch.float16):
            z = (model(x) + model(torch.flip(x, dims=[3]))) / 2  # mirror test-time augmentation
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


if STAGE in ("finetune", "both"):
    name = FINETUNE if FINETUNE != "auto" else max(bench, key=lambda k: max(bench[k]["face"][0], bench[k]["image"][0]))
    view = "image" if bench[name]["image"][0] >= bench[name]["face"][0] else "face"
    square = view == "image"
    arch, pre_tag = CANDIDATES[name]
    big = "bigG" in arch or "gopt" in arch
    train_blocks, lr_top = (4, 1e-5) if big else (8, 2e-5)
    print(f"fine-tuning {name} on '{view}' (last {train_blocks} blocks)")

    X = np.load(f"{OUT}/emb_{name}_{view}.npy").astype(np.float32)
    lp = LogisticRegression(max_iter=4000, C=1).fit(X[TR], y[TR])  # LP init for the head
    vis, pre = load_visual(arch, pre_tag, dtype="fp32")
    head = nn.Linear(X.shape[1], 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(lp.coef_, dtype=torch.float32))
        head.bias.copy_(torch.tensor(lp.intercept_, dtype=torch.float32))
    model = Net(vis, head).to(DEV)
    try:
        model.vis.set_grad_checkpointing(True)
    except Exception:
        pass

    for p in model.vis.parameters():
        p.requires_grad = False
    blocks = find_blocks(model.vis)
    in_blocks = {id(p) for b in blocks for p in b.parameters()}
    groups = [{"params": list(model.head.parameters()), "lr": 1e-4}]
    for d, blk in enumerate(reversed(blocks[-train_blocks:])):  # layer-wise learning-rate decay
        for p in blk.parameters():
            p.requires_grad = True
        groups.append({"params": list(blk.parameters()), "lr": lr_top * 0.8 ** d})
    tail = [p for n, p in model.vis.named_parameters() if id(p) not in in_blocks
            and any(k in n for k in ("norm", "ln_post", "proj", "attn_pool", "head"))]
    for p in tail:
        p.requires_grad = True
    if tail:
        groups.append({"params": tail, "lr": lr_top})

    norm = [t for t in pre.transforms if isinstance(t, T.Normalize)][0]
    size = [t for t in pre.transforms if isinstance(t, T.Resize)][0].size
    size = size if isinstance(size, int) else size[0]
    train_tf = T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.15)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.25, 0.25, 0.2, 0.03), T.RandomGrayscale(0.2), T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size), T.ToTensor(), norm])
    dl = DataLoader(Imgs(meta[view][TR], train_tf, square, y[TR]), batch_size=BATCH, shuffle=True,
                    num_workers=4, drop_last=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=EPOCHS * len(dl) // ACCUM + 1, pct_start=0.1)
    scaler, lossf = torch.amp.GradScaler(), nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    ema = [p.detach().clone() for p in trainable]

    def swap():  # exchange the live weights with the EMA weights
        for p, e in zip(trainable, ema):
            tmp = p.data.clone()
            p.data.copy_(e)
            e.copy_(tmp)

    vpaths = meta[view][VA]
    best = (roc_auc_score(y[VA], predict(model, vpaths, eval_tf, square)), [p.detach().clone() for p in trainable], -1)
    print(f"LP init val AUC {best[0]:.4f}")
    for ep in range(EPOCHS):
        model.train()
        for step, (x, t) in enumerate(tqdm(dl, desc=f"epoch {ep}")):
            with torch.autocast(DEV, dtype=torch.float16):
                loss = lossf(model(x.to(DEV)).float(), t.to(DEV).float() * 0.9 + 0.05) / ACCUM  # label smoothing
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                with torch.no_grad():
                    for p, e in zip(trainable, ema):
                        e.mul_(0.998).add_(p.detach(), alpha=0.002)
        swap()
        auc = roc_auc_score(y[VA], predict(model, vpaths, eval_tf, square))
        print(f"epoch {ep}: EMA val AUC {auc:.4f}")
        if auc > best[0]:
            best = (auc, [p.detach().clone() for p in trainable], ep)
        swap()
    for p, w in zip(trainable, best[1]):
        p.data.copy_(w)

    rows = meta.role.isin(["val", "test"]).values
    z = predict(model, meta[view][rows], eval_tf, square)
    pd.DataFrame({"row_id": meta.row_id[rows].values, "logit": z}).to_csv(f"{OUT}/predictions.csv", index=False)
    json.dump({"model": name, "view": view, "best_epoch": best[2], "val_auc": best[0]}, open(f"{OUT}/finetune.json", "w"))
    print("saved predictions.csv |", name, view, "best epoch", best[2], "val AUC", round(best[0], 4))''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("finetune_metaclip2.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("notebook written:", len(cells), "cells")
