# Task: build the language half of Hakam (حكم)

**Owner:** Anas · **Deadline:** working end-to-end on **14 Sep**, integrated **15 Sep**, presented **16 Sep 2026**

You are building the part that turns a vision model's decision into a grounded
Arabic explanation that cites the Laws of the Game. This file is self-contained:
hand it to your coding agent as-is.

---

## 1. The project in one paragraph

Hakam is an Arabic VAR explainer (Tuwaiq AI bootcamp capstone). A video model
watches a football foul and decides: *was it an offence?* and *does it deserve a
card?* That decision is passed as a small JSON object — the **contract** — to a
language model, which retrieves the relevant article from the IFAB Laws of the
Game and writes an Arabic explanation.

**The one rule that matters: the language model never sees the video.** It only
receives the contract. Every sentence it writes must be traceable to a field in
the contract or to the retrieved Law text. This is what the grading rubric
(§10, "real integration") rewards, and it is our differentiator from prior work
(X-VARS, which feeds video straight into an LLM).

You do **not** need the dataset, the GPU, or the trained model. Everything below
works against `mock_contract()`, which already exists.

---

## 2. Setup

```bash
git clone https://github.com/FerasMad/hakam.git
cd hakam
pip install -r requirements.txt
pip install anthropic sentence-transformers pypdf
```

API key — environment variable only, **never commit it, never paste it in code**:

```bash
export ANTHROPIC_API_KEY=...      # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
```

Model: `claude-sonnet-5` (already set as `LLM_MODEL` in `src/config.py`).

Existing files you will use (do not rewrite):

| File | What it gives you |
|---|---|
| `src/contract.py` | `HakamContract`, `Prediction`, `mock_contract()` |
| `src/llm/lexicon.py` | Arabic surface forms for every label + Arabic text normaliser |
| `src/config.py` | `LLM_MODEL`, `RETRIEVAL_TOP_K = 3`, `CONFIDENCE_THRESHOLD = 0.60`, `ABSTAIN_MESSAGE_AR` |

---

## 3. The contract you receive

```python
from src.contract import mock_contract
c = mock_contract()
print(c.to_json())
```

```json
{
  "action_id": "mock-0001",
  "offence":     {"label": "offence", "confidence": 0.88},
  "card":        {"label": "card",    "confidence": 0.88},
  "card_colour": {"label": "yellow",  "confidence": 0.88},
  "attributes": {
    "action_class": {"label": "tackling",     "confidence": 0.81},
    "body_part":    {"label": "under_body",   "confidence": 0.90},
    "contact":      {"label": "with_contact", "confidence": 0.95},
    "try_to_play":  {"label": "no",           "confidence": 0.72}
  },
  "scene": null,
  "model_version": "mock",
  "num_views": 2,
  "low_confidence_fields": [],
  "abstain": false
}
```

Label vocabulary you must handle:

| Field | Possible labels |
|---|---|
| `offence` | `offence`, `no_offence` |
| `card` | `card`, `no_card`, or `null` (when no offence) |
| `card_colour` | `yellow`, `red`, or **`null` — the real model will almost always send null.** We cut this stage (only 27 red cards in training). Never invent a colour. |
| `action_class` | `tackling`, `standing tackling`, `high leg`, `holding`, `pushing`, `elbowing`, `challenge`, `dive`, `dont know` |
| `body_part` | `upper_body`, `under_body` |
| `contact` | `with_contact`, `without_contact` |
| `try_to_play`, `touch_ball` | `yes`, `no` |

Useful methods:
- `c.should_abstain()` → `True` when offence confidence < 0.60. **Then do not explain** — return `ABSTAIN_MESSAGE_AR`.
- `c.low_confidence_fields()` → fields below 0.60. **Hedge these** ("يبدو أن…", "على الأرجح") instead of stating them as fact.
- `c.citable_values()` → the set of label values the explanation may assert.

Test variants: `mock_contract(offence="no_offence", card=None, colour=None)`,
`mock_contract(confidence=0.45)` (abstain), `mock_contract(colour=None)` (realistic).

---

## 4. What to build

### 4.1 Laws corpus — `laws/`

1. Download the **IFAB Laws of the Game** PDF from theifab.com — **Arabic and English** editions. Put the PDFs in `laws/raw/`.
2. **Check the PDF has a real text layer** (`pypdf` extracts real words, not empty strings). If the Arabic PDF is scanned images, use the English PDF for retrieval and generate Arabic from it — say so in the report.
3. **Curate, do not chunk the whole book.** Only these are relevant:
   - **Law 12 — Fouls and Misconduct** (direct/indirect free kicks, careless / reckless / excessive force, cautionable offences, sending-off offences)
   - **Law 5 — The Referee** (powers, advantage)
   - Glossary definitions: careless, reckless, using excessive force, serious foul play, violent conduct, holding, tackle, challenge
4. Produce `laws/corpus.json` — **30–50 chunks**, each one meaningful rule rather than a fixed character window:

```json
[
  {
    "id": "law12-1-reckless",
    "law": "Law 12",
    "section": "1. Direct free kick",
    "title_ar": "التهور",
    "text_ar": "...",
    "text_en": "...",
    "tags": ["reckless", "yellow", "tackling", "with_contact"]
  }
]
```

`tags` use the contract's label vocabulary — that is what lets retrieval match a contract to an article.

### 4.2 Retrieval — `src/llm/retrieve.py`

```python
def retrieve(contract: HakamContract, k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Top-k corpus chunks for this incident."""
```

- Build the query **from contract fields only** (e.g. `"offence card tackling with_contact under_body"` plus Arabic forms from `lexicon.arabic_forms`).
- Recommended: **hybrid** = tag-overlap score + embedding similarity with `intfloat/multilingual-e5-base` (English-only sentence-transformers retrieve Arabic badly). Tag overlap alone is an acceptable v1.
- Cache corpus embeddings to `laws/embeddings.npy` so the app starts fast.
- For `no_offence`, retrieve the article explaining why a challenge is **not** a foul (fair challenge / playing the ball).

### 4.3 Prompts — `src/llm/prompts.py`

Two versions, because the brief requires evidence of prompt iteration:

- **`PROMPT_V1`** — simple: contract + articles → explain in Arabic.
- **`PROMPT_V2`** — grounded. The system prompt enforces:
  1. Use **only** facts from the contract JSON and the provided articles.
  2. Never describe anything visual that is not a contract field (no "he jumped", "the ball was far", no speed, minute, or player names).
  3. Hedge every field listed in `low_confidence_fields`.
  4. If `card_colour` is null, discuss the possible sanction **conditionally** from the Law text ("قد يستوجب إنذاراً إذا اعتُبر التدخل متهوراً") — never assert a colour.
  5. Cite the article (law + section) explicitly.
  6. Use the fixed Arabic structure below.

Required output structure:

```
القرار: ...
المادة: القانون 12 — ...
التفسير: ...
مستوى الثقة: ...
```

### 4.4 Generation — `src/llm/generate.py`

```python
@dataclass
class Explanation:
    text_ar: str
    articles: list[dict]      # the retrieved chunks actually used
    prompt_version: str       # "v1" | "v2"
    abstained: bool
    model: str

def explain(contract: HakamContract, prompt_version: str = "v2") -> Explanation:
```

- If `contract.should_abstain()`: return `Explanation(text_ar=ABSTAIN_MESSAGE_AR, abstained=True, ...)` **without calling the API**.
- Otherwise: retrieve → build prompt → call Claude (`anthropic` SDK, low temperature) → return.
- Pass the contract as JSON in the user message. The model must receive nothing else about the incident.

### 4.5 Faithfulness scorer — `src/llm/faithfulness.py`

The metric that proves the integration is real. Use the existing lexicon.

```python
def score(explanation_ar: str, contract: HakamContract, articles: list[dict]) -> dict:
    """
    returns {
      "claimed_labels":  [...],   # labels found in the text via lexicon.found_labels
      "supported":       [...],   # claimed AND in contract.citable_values()
      "unsupported":     [...],   # claimed but NOT in the contract -> hallucination
      "faithfulness":    0.0-1.0, # supported / claimed
      "cites_article":   bool,    # mentions a retrieved law/section
      "hedged_low_conf": bool,    # every low-confidence field is hedged or absent
    }
    """
```

- `lexicon.found_labels(text)` and `lexicon.normalise(...)` already handle Arabic spelling variants (إنذار/انذار, ة/ه) and underscores (`with_contact`).
- **Watch for:** `tackling` and `challenge` share the form `التحام` — don't count that as a contradiction. Colour words (`إنذار`, `طرد`) are unsupported only when asserted as the decision, not when quoted conditionally from the Law. A simple rule is fine; document it.

### 4.6 Evaluation — `scripts/eval_llm.py`

Run both prompt versions over **at least 20 mock contracts** (vary offence/no_offence, card/no_card, colour null, low confidence, abstain, each action class). Write `artifacts/llm/eval.csv` and print:

| Metric | v1 | v2 |
|---|---|---|
| mean faithfulness | | |
| % with unsupported claims | | |
| % citing an article | | |
| % correctly abstained | | |
| % hedging low-confidence fields | | |

Also hand-audit **10 outputs** (read each, mark every claim OK / wrong) into `artifacts/llm/manual_audit.csv`. The brief rewards human evaluation alongside the automatic metric.

### 4.7 Demo — `scripts/demo_llm.py`

```bash
python scripts/demo_llm.py    # prints contract, retrieved articles, Arabic explanation, faithfulness
```

Must run on `mock_contract()`. Feras swaps in the real contract on 15 Sep.

---

## 5. Tests — `tests/llm/`

Minimum:
- `lexicon.missing_labels(contract.citable_values())` is empty for every mock variant.
- `retrieve()` returns k chunks; for a `with_contact` + `tackling` contract, a Law 12 chunk is in the top 3.
- `explain()` on an abstaining contract makes **no API call** and returns `ABSTAIN_MESSAGE_AR` (mock the client).
- `score()` flags an invented label as unsupported (text says "طرد" when the contract has no colour).
- `score()` gives 1.0 for text that uses only contract facts.

```bash
python -m pytest tests -q
```

Tests must never call the real API — mock `anthropic`.

---

## 6. Rules

- **Never commit** API keys, the dataset, or anything under `data/`, `frame_cache/`, `artifacts/`. They are gitignored — keep it that way.
- The Laws of the Game PDF is public; `laws/corpus.json` can be committed.
- The model must never receive video, frames, file paths, or match names — **only the contract JSON and Law text.**
- Work on a branch `llm` and open a PR to `main`. Small commits, clear messages.
- Don't modify `src/contract.py` or `src/llm/lexicon.py` without telling Feras — the vision side depends on them. If you need a new label or Arabic form, add it and flag it in the PR.

---

## 7. Done means

- [ ] `laws/corpus.json` with 30–50 curated chunks (Law 12, Law 5, definitions)
- [ ] `retrieve()`, `explain()`, `score()` with the exact signatures above
- [ ] `PROMPT_V1` and `PROMPT_V2`
- [ ] `scripts/demo_llm.py` runs on `mock_contract()`
- [ ] `artifacts/llm/eval.csv` + v1 vs v2 table + 10-item manual audit
- [ ] Tests pass with no real API calls
- [ ] PR open against `main`

## 8. Notes for the report (write as you go)

Short paragraphs Bader can paste into the report:
- Why retrieval over curated Law 12 rather than the whole book
- Which embedding model and why (Arabic support)
- v1 vs v2 table, with one example of a hallucination v2 prevented
- How abstention and hedging work
- Limitations: faithfulness is lexicon-based (catches label claims, not every paraphrase); colour is never predicted, only discussed conditionally

Questions → Feras.
