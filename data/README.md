# Data

**The dataset is not stored in this repository and must not be committed to it.**

Access is granted under a signed NDA whose terms prohibit redistribution. Everything in this
directory except this README and `.gitignore` is ignored by git.

## SoccerNet-MVFoul

Multi-view foul incidents from professional football matches.

| | |
|---|---|
| Source | <https://github.com/SoccerNet/sn-mvfoul> |
| Total actions | 3,901 from 500 matches (6 European leagues, 2014-2017) |
| Train / Valid / Test | 2,916 / 411 / 301 |
| Challenge (unlabeled) | 273 |
| Views per action | at least two - live action plus one or more replays |
| Annotation | 10 referee-perspective properties (offence, severity, foul type, contact point, body part, try-to-play, touch-ball) labelled by a professional referee with 300+ official matches |

### 1. Request access

Fill the NDA form:
<https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform>

A password is emailed after approval. This is the longest lead time in the project - do it first.

### 2. Download

```bash
pip install SoccerNet
```

```python
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
mySNdl.downloadDataTask(
    task="mvfouls",
    split=["train", "valid", "test", "challenge"],
    password="YOUR_PASSWORD",
)
```

Standard definition is the default. Pass `version="720p"` for higher resolution - start with SD,
it is smaller and sufficient for a first pass.

### 3. Extract

Preserve the folder naming convention: `Train`, `Valid`, `Test`, `Chall`. The loader depends on it.

If `train_720p.zip` fails to extract with the default extractor, use The Unarchiver.

### Notes

- Use the official splits as they ship. Do not re-split.
- The test set is only 301 actions, so metrics carry real variance - worth stating in error analysis.

## Baseline weights

The published baseline checkpoint `14_model.pth.tar` is available from the Google Drive link in the
[sn-mvfoul repository](https://github.com/SoccerNet/sn-mvfoul). Its configuration:

```
--pre_model "mvit_v2_s" --pooling_type "attention" --start_frame 63 --end_frame 87 --fps 17
```

Backbone MViT-v2-S, attention pooling across views, and the foul occurring in frames 63-87
(roughly 1.5 seconds). Reproducing this number is the first milestone.

## Laws of the Game

The Arabic edition of the IFAB Laws of the Game is the retrieval corpus. Store the extracted text
under `laws/`.

**Verify before relying on it:** confirm the PDF has a real text layer rather than scanned images.
Open it and try selecting a paragraph. If it is scanned, the retrieval component needs rethinking.
