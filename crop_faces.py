"""Detect + crop the main face for every image listed in a CSV (column `file`).

  python crop_faces.py data/uni.csv data/faces_uni.csv

Rows without a confident face are dropped; crops go to data/faces/ (same naming as embed.py).
"""
import os
import sys
import pandas as pd
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
from tqdm import tqdm
from embed import crop_face

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main(src, dst):
    df = pd.read_csv(src)
    if "downloaded" in df:
        df = df[df.downloaded]
    mtcnn = MTCNN(keep_all=True, device=DEV)
    faces = []
    for row in tqdm(df.itertuples(), total=len(df)):
        face_path = "data/faces/" + os.path.basename(row.file)
        try:
            img = ImageOps.exif_transpose(Image.open(row.file)).convert("RGB")
            face = crop_face(mtcnn, img)
        except Exception:
            face = None
        if face is not None:
            face.resize((224, 224)).save(face_path)
        faces.append(face_path if face is not None else None)
    df["face"] = faces
    df = df.dropna(subset=["face"])
    df.to_csv(dst, index=False)
    print(len(df), "faces")


if __name__ == "__main__":
    main(*sys.argv[1:3])
