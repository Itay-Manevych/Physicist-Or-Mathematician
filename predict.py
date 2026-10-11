"""Usage: python predict.py path/to/photo.jpg [more.jpg ...]

Judges every face it finds in each photo: physicist or mathematician?
"""
import sys
import joblib
import torch
import open_clip
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN

DEV = "cuda" if torch.cuda.is_available() else "cpu"
clf = joblib.load("model.joblib")
mtcnn = MTCNN(keep_all=True, device=DEV)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=DEV)
model.eval()


def verdict(p):
    if p > 0.75:
        return "PHYSICIST. Definitely approximates pi as 3. Possibly as 1."
    if p > 0.55:
        return "physicist-ish. Would call a Fourier series 'basically a sum'."
    if p >= 0.45:
        return "mathematical physicist. Both camps disown them."
    if p >= 0.25:
        return "mathematician-ish. Has opinions about the axiom of choice."
    return "MATHEMATICIAN. Proved their coffee exists but never drank it."


for path in sys.argv[1:]:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        print(f"{path}: no face found")
        continue
    for k, ((x1, y1, x2, y2), pr) in enumerate(zip(boxes, probs)):
        if pr < 0.9:
            continue
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.6
        face = img.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))
        with torch.no_grad():
            f = model.encode_image(preprocess(face).unsqueeze(0).to(DEV))
            f = (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        p = clf.predict_proba(f)[0, 1]
        print(f"{path} face #{k}: P(physicist) = {p:.2f} -> {verdict(p)}")
