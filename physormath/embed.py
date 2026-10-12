"""Image embeddings with test-time augmentation.

Face crops: CLIP-L, CLIP-H, SigLIP2 and FaceNet, each averaged over original / mirrored / grayscale /
grayscale+mirrored, so colour vs. black-and-white stops being a cue.
Whole photo and upper body: CLIP-H averaged over original / grayscale.

Writes data/feats_<name><tag>.npy, row-aligned with the input CSV.
"""
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageOps
from tqdm import tqdm

from .config import BACKBONES, DATA, DEVICE, TIMM_BACKBONES
from .faces import detector, open_image, pad_square, tight, upper_body


def face_views(img):
    g = ImageOps.grayscale(img).convert("RGB")
    return [img, ImageOps.mirror(img), g, ImageOps.mirror(g)]


def color_views(img):
    return [img, ImageOps.grayscale(img).convert("RGB")]


def load_backbone(name):
    """-> (encode(batch) -> features, transform(PIL) -> tensor)"""
    if name == "facenet":
        net = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

        def transform(im):
            return (torch.from_numpy(np.asarray(tight(im), dtype=np.float32)).permute(2, 0, 1) - 127.5) / 128.0
        return net, transform
    if name in TIMM_BACKBONES:
        import timm
        net = timm.create_model(TIMM_BACKBONES[name], pretrained=True, num_classes=0, img_size=224).eval().half().to(DEVICE)
        cfg = {**timm.data.resolve_data_config({}, model=net), "input_size": (3, 224, 224), "crop_pct": 1.0}
        return (lambda x: net(x.half())), timm.data.create_transform(**cfg)
    arch, pretrained = BACKBONES[name]
    # load on the CPU and move only the image tower to the GPU (the text tower isn't needed; saves memory)
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device="cpu", precision="fp16")
    visual = model.visual.eval().to(DEVICE)
    del model
    return (lambda x: visual(x.half())), preprocess


def embed_images(encode, transform, images, views):
    """Mean of the L2-normalised embeddings of each image's augmented views."""
    vs = [views(im) for im in images]
    x = torch.stack([transform(v) for group in vs for v in group]).to(DEVICE)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16):
        e = encode(x).float()
    e = e / e.norm(dim=-1, keepdim=True)
    e = e.view(len(images), len(vs[0]), -1).mean(1)
    return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()


_moved = None


def resolve(path):
    """Path of an image, also when `clean --apply` moved it to data/excluded/<reason>/ (embeddings stay row-aligned)."""
    global _moved
    if Path(path).exists():
        return path
    if _moved is None:
        _moved = {f"{q.parent.name}/{q.name}": str(q) for q in (DATA / "excluded").glob("*/*/*")}
    p = Path(path)
    return _moved.get(f"{p.parent.name}/{p.name}", path)


def _done(path, n):
    return path.exists() and np.load(path, mmap_mode="r").shape[0] == n


BATCH = {"bigG": 4, "mc2bigG": 4, "mc2H378": 8}  # images per batch (x4 augmented views); the largest model needs smaller batches on 6 GB


def embed_full(csv, tag="", backbones=("clipH",), bs=16):
    """Whole photo (padded to square), original + grayscale views -> data/feats_<name>_full<tag>.npy."""
    files = pd.read_csv(csv).file.tolist()
    for name in backbones:
        out = DATA / f"feats_{name}_full{tag}.npy"
        if _done(out, len(files)):
            print("already embedded:", out.name)
            continue
        encode, transform = load_backbone(name)
        b = BATCH.get(name, bs)
        feats = [embed_images(encode, transform, [pad_square(open_image(resolve(f))) for f in files[i:i + b]], color_views)
                 for i in tqdm(range(0, len(files), b), desc=f"{name} whole photo")]
        np.save(out, np.concatenate(feats))
        del encode
        torch.cuda.empty_cache()


def embed_faces(csv, tag="", backbones=("facenet", "clipL", "clipH", "siglip2"), bs=16):
    files = pd.read_csv(csv).face.tolist()
    for name in backbones:
        out = DATA / f"feats_{name}{tag}.npy"
        if _done(out, len(files)):
            print("already embedded:", out.name)
            continue
        encode, transform = load_backbone(name)
        b = BATCH.get(name, bs)
        feats = [embed_images(encode, transform, [Image.open(resolve(f)).convert("RGB") for f in files[i:i + b]], face_views)
                 for i in tqdm(range(0, len(files), b), desc=name)]
        np.save(out, np.concatenate(feats))
        del encode
        torch.cuda.empty_cache()


def embed_context(csv, tag="", bs=16):
    """Whole photo (padded to square) and upper-body crop, with CLIP-H. Upper crops are saved to data/upper/."""
    df = pd.read_csv(csv)
    outs = {k: DATA / f"feats_clipH_{k}{tag}.npy" for k in ("full", "upper")}
    if all(_done(p, len(df)) for p in outs.values()):
        print("already embedded: whole photo + upper body")
        return
    (DATA / "upper").mkdir(exist_ok=True)
    mtcnn = detector()
    encode, transform = load_backbone("clipH")
    feats = {"full": [], "upper": []}
    for i in tqdm(range(0, len(df), bs), desc="whole photo + upper body"):
        fulls, uppers = [], []
        for row in df.iloc[i:i + bs].itertuples():
            img = open_image(row.file)
            boxes, probs = mtcnn.detect(img)
            box = boxes[int(np.argmax(probs))] if boxes is not None else (0, 0, img.size[0], img.size[1] / 3)
            up = upper_body(img, box)
            up.resize((256, 256)).save(DATA / "upper" / row.file.split("/")[-1])
            fulls.append(pad_square(img))
            uppers.append(up)
        feats["full"].append(embed_images(encode, transform, fulls, color_views))
        feats["upper"].append(embed_images(encode, transform, uppers, color_views))
    for k, p in outs.items():
        np.save(p, np.concatenate(feats[k]))
