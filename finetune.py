"""Partially fine-tune CLIP ViT-L/14 on the face crops.

Last N transformer blocks + head are trained; heavy colour/grayscale augmentation discourages
the "photo style" shortcut. Same person-disjoint test split as train.py.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import open_clip
from PIL import Image
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

DEV = "cuda"
N_TRAIN_BLOCKS = 6
EPOCHS = 8
BS = 32
torch.manual_seed(0)

MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3, 0.05),
    T.RandomGrayscale(0.5),
    T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(MEAN, STD)])


class Faces(torch.utils.data.Dataset):
    def __init__(self, files, labels, tf):
        self.files, self.labels, self.tf = files, labels, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB")), self.labels[i]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        clip, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.visual = clip.visual
        self.head = nn.Linear(768, 1)

    def forward(self, x):
        f = self.visual(x)
        return self.head(f / f.norm(dim=-1, keepdim=True) * 20).squeeze(-1)


def tta_predict(model, files):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16):
        for i in range(0, len(files), 64):
            x = torch.stack([eval_tf(Image.open(f).convert("RGB")) for f in files[i:i + 64]]).to(DEV)
            mean = torch.tensor(MEAN, device=DEV).view(1, 3, 1, 1)
            std = torch.tensor(STD, device=DEV).view(1, 3, 1, 1)
            raw = x * std + mean
            gray = (0.299 * raw[:, 0:1] + 0.587 * raw[:, 1:2] + 0.114 * raw[:, 2:3]).expand_as(raw)
            g = (gray - mean) / std
            logits = sum(model(v) for v in (x, x.flip(3), g, g.flip(3))) / 4
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def main():
    df = pd.read_csv("data/faces.csv")
    df = df[df.label.isin(["physicist", "mathematician"])].reset_index(drop=True)
    y = (df.label == "physicist").astype(np.float32).values
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=0.1, stratify=y[idx_tr], random_state=0)
    files = df.face.values

    model = Model().to(DEV)
    for p in model.visual.parameters():
        p.requires_grad = False
    trainable = list(model.visual.transformer.resblocks[-N_TRAIN_BLOCKS:]) + [model.visual.ln_post]
    for m in trainable:
        for p in m.parameters():
            p.requires_grad = True
    model.visual.proj.requires_grad = True
    backbone_params = [p for p in model.visual.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},
        {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    dl = torch.utils.data.DataLoader(Faces(files[idx_tr], y[idx_tr], train_tf), batch_size=BS, shuffle=True,
                                     num_workers=4, persistent_workers=True, drop_last=True)
    total = EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-5, 1e-3], total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    lossf = nn.BCEWithLogitsLoss()

    best_auc, history = 0, []
    for ep in range(EPOCHS):
        model.train()
        tot = 0
        for x, t in tqdm(dl, desc=f"epoch {ep}"):
            x, t = x.to(DEV), t.to(DEV)
            t = t * 0.9 + 0.05  # label smoothing
            with torch.autocast(DEV, dtype=torch.float16):
                loss = lossf(model(x).float(), t)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        va_auc = roc_auc_score(y[idx_va], tta_predict(model, files[idx_va]))
        history.append({"epoch": ep, "train_loss": tot / len(dl), "val_auc": va_auc})
        print(history[-1], flush=True)
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save({k: v for k, v in model.state_dict().items()}, "finetuned.pt")

    model.load_state_dict(torch.load("finetuned.pt"))
    p = tta_predict(model, files[idx_te])
    res = {"test_auc": roc_auc_score(y[idx_te], p), "test_acc": accuracy_score(y[idx_te], p > 0.5),
           "best_val_auc": best_auc, "history": history}
    json.dump(res, open("results_finetune.json", "w"), indent=2)
    df.iloc[idx_te].assign(p=p).sort_values("p").to_csv("test_predictions_finetune.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
