"""Face detection and the three crops the models see: face, upper body, whole photo."""
import os

import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
from tqdm import tqdm

from .config import DATA, DEVICE

FACE_MARGIN = 1.6  # saved face crops are this many times the detected box


def detector():
    return MTCNN(keep_all=True, device=DEVICE)


def open_image(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def main_face(mtcnn, img, min_prob=0.95, min_size=40):
    """Box of the most confident face, or None if there is no clear, reasonably large face."""
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return None
    i = int(np.argmax(probs))
    if probs[i] < min_prob or boxes[i][2] - boxes[i][0] < min_size:
        return None
    return boxes[i]


def face_crop(img, box):
    x1, y1, x2, y2 = box
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * FACE_MARGIN
    return img.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((224, 224))


def tight(face):
    """The detected face box itself, from a saved face crop (for FaceNet)."""
    w = face.size[0]
    m = int(w * (1 - 1 / FACE_MARGIN) / 2)
    return face.crop((m, m, w - m, w - m)).resize((160, 160))


def upper_body(img, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, s, top = (x1 + x2) / 2, max(w, h) * 3.2, y1 - 0.6 * h
    return img.crop((int(cx - s / 2), int(top), int(cx + s / 2), int(top + s)))


def pad_square(img):
    s = max(img.size)
    out = Image.new("RGB", (s, s), (128, 128, 128))
    out.paste(img, ((s - img.size[0]) // 2, (s - img.size[1]) // 2))
    return out


def crop_faces(src, dst):
    """For every image in `src` (column `file`) save the main face to data/faces/; rows without a face are dropped."""
    (DATA / "faces").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    if "downloaded" in df:
        df = df[df.downloaded.astype(bool)]
    mtcnn, faces = detector(), []
    for row in tqdm(df.itertuples(), total=len(df), desc="cropping faces"):
        path = f"data/faces/{os.path.basename(row.file)}"
        try:
            img = open_image(row.file)
            box = main_face(mtcnn, img)
        except Exception:
            box = None
        if box is not None:
            face_crop(img, box).save(path)
        faces.append(path if box is not None else None)
    df["face"] = faces
    df = df.dropna(subset=["face"])
    df.to_csv(dst, index=False)
    print(f"{len(df)} faces -> {dst}")
