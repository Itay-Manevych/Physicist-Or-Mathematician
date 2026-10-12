"""Paths, models and constants shared by every step."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"          # downloaded photos, crops, embeddings (git-ignored)
OUTPUTS = ROOT / "outputs"    # trained models, predictions, reports (git-ignored)
SOURCES = ROOT / "sources"    # scraper seed lists

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USER_AGENT = ("PhysOrMath/0.1 (personal hobby project; "
              "https://github.com/Itay-Manevych/Physicist-Or-Mathematician) python-requests")

MIN_BIRTH_YEAR = 1900  # avoids the "1800s engraving vs. modern photo" shortcut
LABELS = ("mathematician", "physicist")

# image backbones: name -> (open_clip architecture, pretrained tag)
BACKBONES = {
    "clipL": ("ViT-L-14", "openai"),
    "clipH": ("ViT-H-14-quickgelu", "dfn5b"),
    "siglip2": ("ViT-SO400M-14-SigLIP2", "webli"),
    # candidates being evaluated (python -m physormath experiment backbones)
    "eva02L": ("EVA02-L-14", "merged2b_s4b_b131k"),
    "mc2H": ("ViT-H-14-worldwide", "metaclip2_worldwide"),
    "bigG": ("ViT-bigG-14", "laion2b_s39b_b160k"),
    "mc2H378": ("ViT-H-14-worldwide-378", "metaclip2_worldwide"),
    "mc2bigG": ("ViT-bigG-14-worldwide", "metaclip2_worldwide"),
}
TIMM_BACKBONES = {"dinov2L": "vit_large_patch14_dinov2.lvd142m"}  # non-CLIP vision transformers
# what the final model sees: MetaCLIP-2 Worldwide face + whole photo, CLIP-H whole photo (chosen by repeated CV)
MODEL_FEATURES = ["mc2H378", "mc2H378_full", "clipH_full"]
