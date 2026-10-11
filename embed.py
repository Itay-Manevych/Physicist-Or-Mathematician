"""Detect + crop faces, then compute CLIP embeddings for every downloaded portrait."""
import os
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
from tqdm import tqdm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PRETRAINED = "ViT-L-14", "openai"
os.makedirs("data/faces", exist_ok=True)


def crop_face(mtcnn, img):
    """Return a square crop around the single most confident face (with margin), or None."""
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return None
    i = int(np.argmax(probs))
    if probs[i] < 0.95:
        return None
    x1, y1, x2, y2 = boxes[i]
    w, h = x2 - x1, y2 - y1
    if w < 40:  # tiny face in a group photo / far away
        return None
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(w, h) * 1.6
    return img.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))


def main():
    df = pd.read_csv("data/people.csv")
    df = df[df.downloaded].reset_index(drop=True)
    mtcnn = MTCNN(keep_all=True, device=DEV)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=DEV)
    model.eval()

    keep, feats, n_faces = [], [], []
    batch, batch_idx = [], []

    def flush():
        with torch.no_grad(), torch.autocast(DEV):
            f = model.encode_image(torch.stack(batch).to(DEV))
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu().numpy())
        keep.extend(batch_idx)
        batch.clear()
        batch_idx.clear()

    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img = ImageOps.exif_transpose(Image.open(row.file)).convert("RGB")
        except Exception:
            continue
        face = crop_face(mtcnn, img)
        if face is None:
            continue
        face_path = row.file.replace("images", "faces")
        face.resize((224, 224)).save(face_path)
        batch.append(preprocess(face))
        batch_idx.append(i)
        if len(batch) == 64:
            flush()
    if batch:
        flush()

    out = df.loc[keep].copy()
    out["face"] = out.file.str.replace("images", "faces")
    out.to_csv("data/faces.csv", index=False)
    np.save("data/clip_feats.npy", np.concatenate(feats))
    print(out.label.value_counts())


if __name__ == "__main__":
    main()
