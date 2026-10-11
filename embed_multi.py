"""Embed the saved face crops with several backbones, using test-time augmentation
(original / mirrored / grayscale / grayscale+mirrored, averaged) so colour-vs-B&W stops being a cue.

Writes data/feats_<name><tag>.npy, row-aligned with the given CSV (default data/faces.csv).

  python embed_multi.py [--csv data/faces_uni.csv --tag _uni] [backbone ...]
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image, ImageOps
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1

DEV = "cuda"
CLIP_BACKBONES = {
    "clipL": ("ViT-L-14", "openai"),
    "clipH": ("ViT-H-14-quickgelu", "dfn5b"),
    "siglip2": ("ViT-SO400M-14-SigLIP2", "webli"),
}


def views(img):
    g = ImageOps.grayscale(img).convert("RGB")
    return [img, ImageOps.mirror(img), g, ImageOps.mirror(g)]


def tight(img):
    # saved crops are 1.6x the face box; take the central face box itself
    w = img.size[0]
    m = int(w * (1 - 1 / 1.6) / 2)
    return img.crop((m, m, w - m, w - m)).resize((160, 160))


def run(name, files, encode, transform, bs=32, tag=""):
    out_path = f"data/feats_{name}{tag}.npy"
    if os.path.exists(out_path):
        print("skip", name)
        return
    feats = []
    for i in tqdm(range(0, len(files), bs), desc=name):
        imgs = [Image.open(f).convert("RGB") for f in files[i:i + bs]]
        x = torch.stack([transform(v) for im in imgs for v in views(im)]).to(DEV)
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16):
            e = encode(x).float()
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.view(len(imgs), 4, -1).mean(1)
        feats.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    np.save(out_path, np.concatenate(feats))


def main(which, csv="data/faces.csv", tag=""):
    files = pd.read_csv(csv).face.tolist()
    for name in which:
        if name == "facenet":
            net = InceptionResnetV1(pretrained="vggface2").eval().to(DEV)

            def tf(im):
                a = torch.from_numpy(np.asarray(tight(im), dtype=np.float32)).permute(2, 0, 1)
                return (a - 127.5) / 128.0
            run(name, files, net, tf, bs=64, tag=tag)
            del net
        else:
            arch, pre = CLIP_BACKBONES[name]
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pre, device=DEV)
            model = model.half().eval()
            run(name, files, lambda x: model.encode_image(x.half()), preprocess, bs=16, tag=tag)
            del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/faces.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("backbones", nargs="*", default=["facenet", "clipL", "clipH", "siglip2"])
    a = ap.parse_args()
    main(a.backbones, a.csv, a.tag)
