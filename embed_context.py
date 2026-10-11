"""Embed MORE than the face: the whole photo (padded to square) and an upper-body crop.

Uses ViT-H/14 with original+grayscale TTA. Writes data/feats_clipH_full.npy and data/feats_clipH_upper.npy,
row-aligned with data/faces.csv. Upper-body crops are also saved to data/upper/ for the game.
"""
import os
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
from tqdm import tqdm

DEV = "cuda"
os.makedirs("data/upper", exist_ok=True)


def pad_square(img):
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), (128, 128, 128))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def upper_body(img, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, s = (x1 + x2) / 2, max(w, h) * 3.2
    top = y1 - 0.6 * h
    crop = img.crop((int(cx - s / 2), int(top), int(cx + s / 2), int(top + s)))  # out-of-bounds -> black fill
    return crop


def main(csv="data/faces.csv", tag=""):
    df = pd.read_csv(csv)
    mtcnn = MTCNN(keep_all=True, device=DEV)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14-quickgelu", pretrained="dfn5b", device=DEV)
    model = model.half().eval()
    feats = {"full": [], "upper": []}

    def embed(imgs):
        views = [v for im in imgs for v in (im, ImageOps.grayscale(im).convert("RGB"))]
        x = torch.stack([preprocess(v) for v in views]).to(DEV).half()
        with torch.no_grad():
            e = model.encode_image(x).float()
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.view(len(imgs), 2, -1).mean(1)
        return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()

    B = 16
    for i in tqdm(range(0, len(df), B)):
        fulls, uppers = [], []
        for row in df.iloc[i:i + B].itertuples():
            img = ImageOps.exif_transpose(Image.open(row.file)).convert("RGB")
            boxes, probs = mtcnn.detect(img)
            box = boxes[int(np.argmax(probs))] if boxes is not None else (0, 0, img.size[0], img.size[1] / 3)
            up = upper_body(img, box)
            up.resize((256, 256)).save("data/upper/" + os.path.basename(row.file))
            fulls.append(pad_square(img))
            uppers.append(up)
        feats["full"].append(embed(fulls))
        feats["upper"].append(embed(uppers))
    for k, v in feats.items():
        np.save(f"data/feats_clipH_{k}{tag}.npy", np.concatenate(v))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:3])
