# حكم (Hakam)

> Arabic explanations for football refereeing decisions — multi-view foul
> classification grounded in the Laws of the Game.

## The problem

VAR decisions are announced without their reasoning. The referee signals, the
screen shows a check, a card appears — and no explanation follows. The viewer,
the commentator, and the club are left to guess.

The Laws of the Game already contain precise criteria: a challenge is *careless*,
*reckless*, or made with *excessive force*, and each carries a defined sanction.
The answer exists in writing. Nobody connects it to the incident in time.

## What this does

Given a foul incident filmed from multiple camera angles, Hakam:

1. Classifies what physically happened — severity, foul type, contact point
2. Retrieves the matching article from the Laws of the Game
3. Generates an Arabic explanation of why the decision was what it was

## Pipeline

```
   multi-view clip
        |
        v  frame sampling (frames 63-87 @ 17fps)
        v  frozen video backbone, embeddings cached to disk
        v  multi-view pooling
        v  cascade:  offence?  ->  card?  ->  yellow / red?
        |
   =====|=====  contract boundary
        v
   HakamContract (JSON)  { labels + confidences only }
        |
        v  FAISS retrieval over the Laws of the Game
        v  Claude
        |
        v
   Arabic explanation
```

**Design note.** The language model never sees the video. It receives only the
JSON contract, so it cannot assert a visual detail the classifier did not
produce. Grounding is enforced by construction, not by prompting.

Full diagrams: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Data

The dataset is **not** in this repository and cannot be redistributed.

**SoccerNet-MVFoul** — 3,901 foul incidents from 500 matches, each filmed from
two or more angles, annotated across 10 referee-perspective properties by a
professional referee with 300+ official matches.

To obtain it, sign the data agreement at
<https://github.com/SoccerNet/sn-mvfoul> and place the files under `data/`.
See [`data/README.md`](data/README.md).

## Structure

| Path | Contents |
|---|---|
| `README.md` | this file |
| `data/` | dataset location (contents gitignored) |

Source, notebooks, app and law-text directories are added as they are built.

## Related work

This project builds on published research and does not claim to outperform it:

- **VARS** — Held et al., CVPR 2023 Workshop. Introduced SoccerNet-MVFoul and
  the multi-view approach. <https://arxiv.org/abs/2304.04617>
- **X-VARS** — a multimodal LLM producing refereeing explanations in English.
  <https://arxiv.org/abs/2404.06332>

Hakam differs in architecture and language: explanations are Arabic, retrieval
is grounded in the Laws of the Game text, and the language model is isolated
from the video by a structured contract.

## Status

Early development. Bootcamp capstone project.

## Disclaimer

Experimental prototype. Does not make refereeing decisions and does not replace
a referee. It explains decisions that have already been made, and declares low
confidence rather than guessing.
