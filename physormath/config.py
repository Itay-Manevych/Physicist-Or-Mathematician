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
}
# what the final model sees: three face embeddings + the whole photo
MODEL_FEATURES = ["clipL", "clipH", "siglip2", "clipH_full"]
