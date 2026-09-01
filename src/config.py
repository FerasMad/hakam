"""Central configuration for Hakam.

Every constant the pipeline depends on lives here so that experiments differ by
config, not by edited code scattered across notebooks.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"          # dataset lands here (gitignored)
FEATURES_CACHE = PROJECT_ROOT / "features_cache"  # backbone embeddings (gitignored)
LAWS_DIR = PROJECT_ROOT / "laws"           # Laws of the Game text for retrieval
ARTIFACTS = PROJECT_ROOT / "artifacts"     # checkpoints, metrics, figures

# Split directory names as they ship from SoccerNet. Do not rename.
SPLIT_DIRS = {
    "train": "Train",
    "valid": "Valid",
    "test": "Test",
    "challenge": "Chall",
}

# Official split sizes, for sanity-checking the download completed.
SPLIT_SIZES = {"train": 2916, "valid": 411, "test": 301, "challenge": 273}

# --------------------------------------------------------------------------
# Frame sampling
#
# Taken from the published SoccerNet-MVFoul baseline configuration:
#   --start_frame 63 --end_frame 87 --fps 17
# The foul occupies roughly 1.5 seconds centred in the clip. Sampling the whole
# clip dilutes the signal; this window is where the incident actually happens.
# --------------------------------------------------------------------------

START_FRAME = 63
END_FRAME = 87
FPS = 17
NUM_FRAMES = 16          # frames fed to the backbone after subsampling
FRAME_SIZE = 224

# --------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------

BACKBONES = {
    # Baseline: reproduces the published result.
    "mvit_v2_s": {"source": "torchvision", "name": "mvit_v2_s", "dim": 768},
    # Experiment 1: stronger self-supervised video pretraining.
    "videomae_base": {
        "source": "huggingface",
        "name": "MCG-NJU/videomae-base-finetuned-kinetics",
        "dim": 768,
    },
    # Faster variant for iteration.
    "videomae_small": {
        "source": "huggingface",
        "name": "MCG-NJU/videomae-small-finetuned-kinetics",
        "dim": 384,
    },
}

DEFAULT_BACKBONE = "mvit_v2_s"

# --------------------------------------------------------------------------
# Cascade task definition
#
# Three binary decisions rather than one multi-class head. This mirrors how a
# referee actually decides, and each stage has a 50% chance floor instead of
# 12.5%, which is what makes ~70% per stage attainable. Published multi-class
# results on this dataset sit near 50% balanced accuracy, so a single 8-way
# head would look far worse for the same underlying model quality.
# --------------------------------------------------------------------------

CASCADE_STAGES = [
    {
        "name": "offence",
        "question": "Is this an offence?",
        "classes": ["no_offence", "offence"],
        "parent": None,          # always evaluated
    },
    {
        "name": "card",
        "question": "Does it warrant a card?",
        "classes": ["no_card", "card"],
        "parent": ("offence", "offence"),   # only when stage 1 says offence
    },
    {
        "name": "card_colour",
        "question": "Yellow or red?",
        "classes": ["yellow", "red"],
        "parent": ("card", "card"),         # only when stage 2 says card
    },
]

# Auxiliary heads. Trained jointly as a regulariser on a small dataset — the
# annotations are already there, so this signal is free.
AUXILIARY_HEADS = [
    "action_class",
    "body_part",
    "contact",
    "try_to_play",
    "touch_ball",
]

# NOTE: the raw label vocabulary for each property must be read from the
# dataset's annotations.json once downloaded and mapped here. Do not hardcode
# guessed label strings — confirm them against the file.
RAW_LABEL_VOCAB: dict[str, list[str]] = {}

# --------------------------------------------------------------------------
# Confidence gating
#
# With state of the art near 50% on the multi-class task, a system that always
# answers is worse than one that declines. Below this threshold the pipeline
# abstains instead of generating an explanation.
# --------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60
ABSTAIN_MESSAGE_AR = "الثقة منخفضة — هذه الحالة تحتاج مراجعة بشرية."

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
CLASS_WEIGHTING = True   # balanced accuracy is the metric; weight the rare classes

# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

LLM_MODEL = "claude-sonnet-5"
LLM_MAX_TOKENS = 1024
RETRIEVAL_TOP_K = 3
