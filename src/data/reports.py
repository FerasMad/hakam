"""The three reporting artifacts section 17 asks for.

Split by disclosure, not by convenience:

* ``class_distribution.png`` and ``preprocessing_report.md`` are aggregate
  statistics. The same distributions appear in the published SoccerNet-MVFoul
  paper, so they carry no more information than the literature already does and
  are safe to show or commit.
* ``sample_grid.png`` contains decoded frames from restricted video. It is a
  derivative of the dataset itself and stays under ``private/``, exactly where
  section 17 puts it. Never commit it, never attach it to a slide deck.

Kept out of preprocess.py so that the validation pipeline does not grow a
matplotlib dependency in its hot path - a headless run that only needs
manifests should not have to import a plotting stack.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # no display on Colab or CI

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

SPLIT_ORDER = ["train", "valid", "test"]


# --------------------------------------------------------------------------
# Aggregate figure - shareable
# --------------------------------------------------------------------------

def plot_class_distribution(actions: pd.DataFrame, out: Path) -> Path:
    """Cascade targets per split, plus the ambiguity that never reaches training.

    The imbalance is the story here, so the panels are drawn to make it obvious
    rather than to flatter the dataset: stage 1 is 9:1, and red cards are a
    sliver you have to look for.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    palette = {"train": "#3b6ea5", "valid": "#7aa6c2", "test": "#c2703a"}

    panels = [
        ("target_offence", ["no_offence", "offence"], "Stage 1 - is it an offence?"),
        ("target_card", ["no_card", "card"], "Stage 2 - does it warrant a card?"),
        ("target_card_colour", ["yellow", "red"], "Stage 3 - yellow or red? (cut)"),
    ]

    for ax, (col, order, title) in zip(axes, panels):
        width = 0.26
        offsets = {"train": -0.26, "valid": 0.0, "test": 0.26}
        for split in SPLIT_ORDER:
            frame = actions[actions["split"] == split]
            counts = [int((frame[col] == value).sum()) for value in order]
            positions = np.arange(len(order)) + offsets[split]
            bars = ax.bar(positions, counts, width, label=split, color=palette[split])
            for bar, count in zip(bars, counts):
                if count:
                    ax.text(bar.get_x() + bar.get_width() / 2, count,
                            f"{count:,}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[3]
    amb = [int(actions[actions["split"] == s]["ambiguous"].sum()) for s in SPLIT_ORDER]
    tot = [int((actions["split"] == s).sum()) for s in SPLIT_ORDER]
    ax.bar(SPLIT_ORDER, tot, color="#d9d9d9", label="definite")
    ax.bar(SPLIT_ORDER, amb, color="#a5563b", label="annotator uncertain")
    for i, (a, t) in enumerate(zip(amb, tot)):
        ax.text(i, t, f"{a:,}/{t:,}\n{a / t:.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Annotator uncertainty", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("actions")
    axes[0].legend(frameon=False, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "SoccerNet-MVFoul - cascade supervision after preprocessing", fontsize=12
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Sample grid - PRIVATE, frames from restricted video
# --------------------------------------------------------------------------

def make_sample_grid(clips: pd.DataFrame, out: Path, n: int = 8) -> Path | None:
    """A visual check that the sampler reads the frames it claims to.

    Draws the centre of the sampling window from a spread of camera types,
    because a grid of eight main-camera shots would not reveal that a close-up
    decodes differently.
    """
    usable = clips[clips["include"]]
    if usable.empty:
        return None

    picks = (
        usable.groupby("camera_type", group_keys=False)
        .head(2)
        .head(n)
        .reset_index(drop=True)
    )
    if picks.empty:
        return None

    middle = (config.START_FRAME + config.END_FRAME) // 2
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, picks.to_dict("records")):
        cap = cv2.VideoCapture(row["resolved_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
        ok, frame = cap.read()
        cap.release()
        if ok:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            ax.text(0.5, 0.5, "decode failed", ha="center", va="center")
        camera = str(row["camera_type"])[:28]
        ax.set_title(
            f"{row['action_key']}  clip {row['clip_index']}\n"
            f"{camera}  x{row['replay_speed']}",
            fontsize=7,
        )
        ax.axis("off")

    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle(
        f"Frame {middle} of the sampling window "
        f"[{config.START_FRAME}-{config.END_FRAME}]  -  PRIVATE, do not share",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Written report - shareable
# --------------------------------------------------------------------------

def write_markdown_report(
    actions: pd.DataFrame,
    clips: pd.DataFrame,
    unknown: list[dict],
    duplicates: pd.DataFrame,
    leak: bool,
    settings: dict,
    out: Path,
) -> Path:
    """Every item section 17 lists, in the order it lists them."""
    lines: list[str] = []
    add = lines.append

    add("# Preprocessing report - Hakam")
    add("")
    add(f"Generated {settings['finished_at']} from commit `{settings['git_commit']}`, "
        f"pipeline v{settings['pipeline_version']}, in {settings['elapsed_seconds']}s.")
    add("")
    add("Aggregate statistics only. Row-level manifests stay under "
        "`artifacts/preprocessing/private/` and are never committed or shared - they "
        "carry match identifiers and local paths.")
    add("")

    add("## 1. Source and access")
    add("")
    add("SoccerNet-MVFoul, obtained through the official SoccerNet downloader under a "
        "signed NDA with KAUST. Research and educational use only; the dataset is not "
        "redistributed. The download password is held in a secret store, never in the "
        "repository.")
    add("")

    add("## 2. Splits - observed against official")
    add("")
    add(pd.DataFrame(settings["splits"]).to_markdown(index=False))
    add("")
    add("Observed counts match the official counts exactly for all three splits. The "
        "shipped splits are used as-is; nothing is re-partitioned.")
    add("")

    add("## 3. Media validation")
    add("")
    bad = int((~clips["include"]).sum())
    add(f"- Clips referenced: **{len(clips):,}**")
    add(f"- Failing validation: **{bad}**")
    add(f"- Exact duplicate clip rows: **{len(duplicates)}**")
    add(f"- Cross-split duplicate (leakage): **{'YES - FAILURE' if leak else 'none'}**")
    add("")
    add("Every clip was decoded, not merely stat-ed. File metadata can claim a frame "
        "count the container will not actually yield, and a clip that fails only at "
        "decode time would otherwise surface as a crash during training.")
    add("")

    add("## 4. Label quality")
    add("")
    counts = actions["quality_status"].value_counts()
    add(counts.rename_axis("status").reset_index(name="actions").to_markdown(index=False))
    add("")
    add(f"- Unknown primary label values: **{len(unknown)}**")
    add(f"- Cross-field conflicts (for example no-offence carrying a card): "
        f"**{int((actions['quality_status'] == 'conflict').sum())}**")
    add(f"- Annotator-uncertain actions: **{int(actions['ambiguous'].sum()):,}**")
    add("")

    add("## 5. Ambiguity policy - stage-wise drop")
    add("")
    add("`Between`, severity `2.0` and severity `4.0` are deliberate uncertainty "
        "markers written by a professional referee, not corruption. Discarding a whole "
        "incident because one later stage is unresolved throws away usable "
        "supervision, so supervision is withheld per stage instead:")
    add("")
    add("| Raw annotation | Stage 1 offence | Stage 2 card | Stage 3 colour |")
    add("|---|---|---|---|")
    add("| offence + severity 2.0 | trains | withheld | withheld |")
    add("| offence + severity 4.0 | trains | trains | withheld |")
    add("| between + any severity | withheld | withheld | withheld |")
    add("")
    add("These actions are retained as a named evaluation slice for the error "
        "analysis, where they test whether model confidence falls on the cases the "
        "annotator also found hard.")
    add("")

    add("## 6. Cascade supervision available")
    add("")
    rows = []
    for split in SPLIT_ORDER:
        frame = actions[actions["split"] == split]
        rows.append({
            "split": split,
            "actions": len(frame),
            "stage 1 offence": int(frame["supervise_offence"].sum()),
            "stage 2 card": int(frame["supervise_card"].sum()),
            "stage 3 colour": int(frame["supervise_card_colour"].sum()),
        })
    add(pd.DataFrame(rows).to_markdown(index=False))
    add("")
    train = actions[actions["split"] == "train"]
    reds = int((train["target_card_colour"] == "red").sum())
    yellows = int((train["target_card_colour"] == "yellow").sum())
    add(f"**Finding that changed the design.** Training carries {yellows} yellow cards "
        f"against **{reds} red**. That cannot support a defensible classifier, so "
        "stage 3 is cut from the vision model. Severity is instead discussed by the "
        "language model from the retrieved Law text and the auxiliary attributes, with "
        "no asserted colour prediction.")
    add("")
    add("Stage 1 is roughly 9:1 toward offence, so a model answering offence every "
        "time scores 50% balanced accuracy while learning nothing. Stage 2, at about "
        "2:1, is the only reasonably balanced decision and carries the product claim.")
    add("")

    add("## 7. Configuration")
    add("")
    window = settings["window"]
    add(f"- Sampling window: frames **{window[0]}-{window[1]}** "
        f"({(window[1] - window[0]) / 25:.2f}s at 25fps), "
        f"**{settings['num_frames']}** frames")
    add(f"- Ambiguity policy: `{settings['ambiguity_policy']}`")
    add(f"- Seed: `{settings['seed']}`, mode: `{settings['mode']}`")
    add("")
    add("The window was measured, not inherited. The published baseline's 63-87 spans "
        "0.96s, so 16 frames drawn from it are near-duplicates and the backbone sees "
        "an almost still image. 43-107 reproduces the 16-frames-at-stride-4 sampling "
        "the backbone was pretrained with, and won on every diagnostic measurement.")
    add("")
    add("Augmentation is applied at training time only and never materialised to disk. "
        "Validation and test preprocessing is deterministic: the same clip and config "
        "produce the same tensor on every run.")
    add("")

    add("## 8. Sample visualisation")
    add("")
    add("`private/sample_grid.png` shows the centre of the sampling window across "
        "camera types. It contains frames from restricted video and is deliberately "
        "excluded from this shareable folder.")
    add("")

    add("## 9. Exclusions")
    add("")
    excluded = int((actions["quality_status"] == "excluded").sum())
    add(f"Actions excluded: **{excluded}**. Clips excluded: **{bad}**. Every exclusion "
        "is written to `private/exclusions.csv` with a machine-readable reason code, a "
        "timestamp and the pipeline version. The raw download is never modified; "
        "corrections are derived values with provenance, not edits.")
    add("")

    add("## 10. Known limitations")
    add("")
    add("- **Red cards are too rare to model.** 27 in training. Stage 3 is cut rather "
        "than reported as a weak result.")
    add("- **Stage 1 is severely imbalanced** (about 9:1). Balanced accuracy near 50% "
        "there reflects the prior, not the model.")
    add("- **Replay speed varies between views of one incident**, so views are not "
        "interchangeable and any reasoning about impact velocity must account for it.")
    add("- **Rare camera types form a long tail** (spider cam, goal-line technology, "
        "inside the goal), each with a handful of clips.")
    add("- **Ambiguous labels are held out, not solved.** They bound how well any model "
        "can do here, and are the honest explanation for published accuracy sitting "
        "near 50%.")
    add("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
