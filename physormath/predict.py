"""Judge one person from one or more photos (predictions are averaged across photos)."""
import joblib
import numpy as np
import torch

from .config import CLOUD_BACKBONES, OUTPUTS
from .embed import color_views, embed_images, face_views, load_backbone
from .faces import detector, face_crop, main_face, open_image, pad_square


def verdict(p, margins):
    m = abs(p - 0.5)
    side = "PHYSICIST" if p > 0.5 else "MATHEMATICIAN"
    if m >= margins["confident"]:
        return f"{side} (confident)"
    if m >= margins["leaning"]:
        return f"leans {side.lower()}"
    return "can't tell - could be either"


def judge(paths):
    bundle = joblib.load(OUTPUTS / "model.joblib")
    mtcnn = detector()
    faces, wholes, used = [], [], []
    for path in paths:
        img = open_image(path)
        box = main_face(mtcnn, img, min_prob=0.9)
        if box is None:
            print(f"{path}: no clear face, skipped")
            continue
        faces.append(face_crop(img, box))
        wholes.append(pad_square(img))
        used.append(path)
    if not faces:
        return None
    feats = {}
    for name in sorted({f.replace("_full", "") for f in bundle["features"]}):
        encode, transform = load_backbone(name)
        single = (lambda im: [im]) if name in CLOUD_BACKBONES else None  # match how the training photos were embedded
        if name in bundle["features"]:
            feats[name] = embed_images(encode, transform, faces, single or face_views)
        if f"{name}_full" in bundle["features"]:
            feats[f"{name}_full"] = embed_images(encode, transform, wholes, single or color_views)
        del encode
        torch.cuda.empty_cache()
    X = np.hstack([feats[f] for f in bundle["features"]])
    ps = bundle["clf"].predict_proba(X)[:, 1]
    for path, p in zip(used, ps):
        print(f"{path}: P(physicist) = {p:.2f}")
    p = float(ps.mean())
    print(f"\n=> {verdict(p, bundle['margins'])}   (P(physicist) = {p:.2f} from {len(ps)} photo(s))")
    return p
