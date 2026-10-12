"""Fine-tune a vision transformer (LP-FT recipe) and blend it with the linear-probe ensemble.

  python -m physormath finetune <backbone> <face|full>

Recipe: the head starts from a logistic-regression probe on the frozen features (LP-FT); only the last few
transformer blocks train, with layer-wise learning-rate decay, strong augmentation, label smoothing and an EMA of the
weights. People are split as in `train`; 15% of the TRAIN people are held out to pick the epoch and the blend weight
with the probe ensemble. The test set is scored once at the end. Results append to outputs/experiments.csv.
"""
import copy
import time

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import transforms as T

from .config import BACKBONES, DEVICE, MODEL_FEATURES, OUTPUTS
from .embed import resolve
from .experiments import load, log
from .faces import open_image, pad_square

TRAIN_BLOCKS, EPOCHS, BATCH, LR_TOP, LR_DECAY, LR_HEAD = 4, 3, 24, 2e-5, 0.7, 1e-4


class Images(torch.utils.data.Dataset):
    def __init__(self, paths, y, view, tf):
        self.paths, self.y, self.view, self.tf = list(paths), y, view, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = resolve(self.paths[i])
        img = Image.open(p).convert("RGB") if self.view == "face" else pad_square(open_image(p))
        return self.tf(img), float(self.y[i])


class Net(nn.Module):
    def __init__(self, visual, head):
        super().__init__()
        self.visual, self.head = visual, head

    def forward(self, x):
        f = self.visual(x)
        return self.head(f / f.norm(dim=-1, keepdim=True)).squeeze(-1)


def blocks_of(visual):
    """Transformer blocks of an open_clip image tower (OpenAI/OpenCLIP ViT or timm-based)."""
    if hasattr(visual, "transformer"):
        return list(visual.transformer.resblocks)
    trunk = getattr(visual, "trunk", visual)
    return list(trunk.blocks)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(DEVICE, dtype=torch.float16):
            out.append(torch.sigmoid(model(x.to(DEVICE)).float()).cpu().numpy())
    return np.concatenate(out)


def finetune(backbone="clipH", view="full", smoke=False):
    t0 = time.time()
    torch.manual_seed(0)
    feat_name = backbone if view == "face" else f"{backbone}_full"
    df, X_probe = load(MODEL_FEATURES)             # production probe features (for the blend)
    _, X_own = load([feat_name])                  # this backbone's frozen features (for the LP init)
    y = df.y.values
    test = df.test.values
    groups = df.group.values
    rng = np.random.default_rng(0)
    train_people = np.unique(groups[~test])
    hold = set(rng.choice(train_people, int(0.15 * len(train_people)), replace=False))
    val = ~test & np.isin(groups, list(hold))
    fit = ~test & ~val
    main = df.source.isin(["wiki", "new"]).values
    paths = df.face.values if view == "face" else df.file.values
    if smoke:  # tiny run to check the code path end to end
        keep = lambda m, n: (m & (y == 1) & (np.cumsum(m & (y == 1)) <= n // 2)) | (m & (y == 0) & (np.cumsum(m & (y == 0)) <= n // 2))
        fit_small, val, test = keep(fit, 48), keep(val, 32), keep(test, 32)
    else:
        fit_small = fit

    # probe ensemble (frozen features) trained on `fit`, scored everywhere
    probe = LogisticRegression(max_iter=5000, C=1).fit(X_probe[fit], y[fit])
    p_probe = probe.predict_proba(X_probe)[:, 1]
    # LP init: logistic regression on this backbone's normalised frozen features
    lp = LogisticRegression(max_iter=5000, C=1).fit(X_own[fit], y[fit])

    arch, pretrained = BACKBONES[backbone]
    clip_model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device="cpu")
    visual = clip_model.visual
    del clip_model
    head = nn.Linear(X_own.shape[1], 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(lp.coef_, dtype=torch.float32))
        head.bias.copy_(torch.tensor(lp.intercept_, dtype=torch.float32))
    model = Net(visual, head).to(DEVICE)

    for p in model.visual.parameters():
        p.requires_grad = False
    blocks = blocks_of(model.visual)
    groups_lr = [{"params": model.head.parameters(), "lr": LR_HEAD}]
    for depth, blk in enumerate(reversed(blocks[-TRAIN_BLOCKS:])):
        for p in blk.parameters():
            p.requires_grad = True
        groups_lr.append({"params": list(blk.parameters()), "lr": LR_TOP * LR_DECAY ** depth})
    tail = [m for m in [getattr(model.visual, "ln_post", None), getattr(getattr(model.visual, "trunk", None), "norm", None),
                        getattr(model.visual, "head", None)] if isinstance(m, nn.Module)]
    for m in tail:
        for p in m.parameters():
            p.requires_grad = True
        groups_lr.append({"params": list(m.parameters()), "lr": LR_TOP})
    if getattr(model.visual, "proj", None) is not None and isinstance(model.visual.proj, nn.Parameter):
        model.visual.proj.requires_grad = True
        groups_lr.append({"params": [model.visual.proj], "lr": LR_TOP})

    norm = [t for t in preprocess.transforms if isinstance(t, T.Normalize)][0]
    size = [t for t in preprocess.transforms if isinstance(t, T.Resize)][0].size
    size = size if isinstance(size, int) else size[0]
    train_tf = T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.15)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.25, 0.25, 0.2, 0.03), T.RandomGrayscale(0.2), T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size), T.ToTensor(), norm])
    mk = lambda mask, tf, shuffle: torch.utils.data.DataLoader(
        Images(paths[mask], y[mask], view, tf), batch_size=BATCH, shuffle=shuffle, num_workers=4,
        persistent_workers=False, drop_last=shuffle)
    train_dl, val_dl, test_dl = mk(fit_small, train_tf, True), mk(val, eval_tf, False), mk(test, eval_tf, False)

    opt = torch.optim.AdamW(groups_lr, weight_decay=0.05)
    epochs = 1 if smoke else EPOCHS
    steps = epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups_lr], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    lossf = nn.BCEWithLogitsLoss()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad = False

    def val_auc(m):
        return roc_auc_score(y[val], predict(m, val_dl))

    best = (val_auc(ema), copy.deepcopy(ema.state_dict()), -1)
    print(f"{backbone}/{view}: LP init val AUC {best[0]:.3f}", flush=True)
    for ep in range(epochs):
        model.train()
        for x, t in train_dl:
            x, t = x.to(DEVICE), t.to(DEVICE).float() * 0.9 + 0.05
            with torch.autocast(DEVICE, dtype=torch.float16):
                loss = lossf(model(x).float(), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    if pm.requires_grad:
                        pe.mul_(0.998).add_(pm.detach(), alpha=0.002)
        auc = val_auc(ema)
        print(f"  epoch {ep}: val AUC {auc:.3f}", flush=True)
        if auc > best[0]:
            best = (auc, copy.deepcopy(ema.state_dict()), ep)
    ema.load_state_dict(best[1])

    p_ft_val, p_ft_test = predict(ema, val_dl), predict(ema, test_dl)
    # blend weight chosen on the held-out TRAIN people only
    logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
    ws = np.linspace(0, 1, 11)
    w = max(ws, key=lambda w: roc_auc_score(y[val], w * logit(p_ft_val) + (1 - w) * logit(p_probe[val])))
    blend_test = w * logit(p_ft_test) + (1 - w) * logit(p_probe[test])
    tm = main[test]
    res_ft = {"cv_auc": best[0], "test_auc": roc_auc_score(y[test][tm], p_ft_test[tm]),
              "test_acc": accuracy_score(y[test][tm], p_ft_test[tm] > 0.5)}
    res_blend = {"cv_auc": roc_auc_score(y[val], w * logit(p_ft_val) + (1 - w) * logit(p_probe[val])),
                 "test_auc": roc_auc_score(y[test][tm], blend_test[tm]), "test_acc": accuracy_score(y[test][tm], blend_test[tm] > 0)}
    res_probe = {"cv_auc": roc_auc_score(y[val], p_probe[val]), "test_auc": roc_auc_score(y[test][tm], p_probe[test][tm]),
                 "test_acc": accuracy_score(y[test][tm], p_probe[test][tm] > 0.5)}
    secs = time.time() - t0
    if smoke:
        print("smoke test ok:", res_ft, res_blend)
        return
    log(f"[val=15% train] probe ensemble (fit on 85%)", res_probe, secs)
    log(f"finetune {backbone}/{view} (best epoch {best[2]})", res_ft, secs)
    log(f"blend probe + finetune {backbone}/{view} (w={w:.1f})", res_blend, secs)
    np.save(OUTPUTS / f"ft_{backbone}_{view}_test.npy", p_ft_test)
    torch.save(best[1], OUTPUTS / f"ft_{backbone}_{view}.pt")
