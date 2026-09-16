# Grounded Arabic explanation layer

## Scope and evidence boundary

The language layer turns `HakamContract` into an Arabic explanation of a decision. It never
receives video, frames, paths, match names, or player identities. Its only incident evidence is
the JSON contract; its only external evidence is the retrieved Laws of the Game text. Development
and evaluation use `mock_contract()`, so these results measure the language and retrieval layer,
not the trained vision model.

The original task named Anthropic Claude. The implementation uses OpenAI `gpt-5-nano` instead,
following the project owner's later provider choice. The key is read only from
`OPENAI_API_KEY`; no secret is stored in the repository.

## Laws corpus

The source is the official IFAB Laws of the Game 2026/27 in Arabic and English. The English PDF
has 260 pages, of which 257 yielded text (242,782 extracted characters). The Arabic PDF has 236
pages, of which 235 yielded text (208,743 characters, including 157,438 Arabic characters). Law
12 was also rendered from both editions and visually compared with its extracted text. The Arabic
edition therefore has a usable text layer and no OCR was required.

Rather than splitting the full book by character count, `laws/corpus.json` contains 47 manually
curated bilingual rules. They cover Law 12, Law 5, and the relevant glossary definitions. This
keeps each retrieval unit legally meaningful and avoids filling the prompt with unrelated rules.
Each chunk records Arabic and English summaries, contract-vocabulary tags, and source page
numbers.

## Retrieval and generation

Retrieval combines exact contract-tag overlap with cosine similarity from
`intfloat/multilingual-e5-base`. Multilingual E5 was selected because the corpus and query contain
both Arabic and English; an English-only embedding model would weaken Arabic matching. The 47
passage vectors are cached in `laws/embeddings.npy`. Tag-only retrieval remains available as a
deterministic fallback and is used in unit tests.

The ranking contains explicit safety rules. A `no_offence` contract prioritises the fair-challenge
article. A missing card colour prioritises the conditional-sanction rule but only when the
contract actually says a card is warranted. Articles about DOGSO, promising attacks, thrown
objects, or a different action class are demoted because the current contract cannot establish
those facts.

Prompt v1 asks for a concise Arabic explanation. Prompt v2 adds the evidence boundary, exact
four-line format, null-colour handling, abstention and hedging rules, canonical Arabic translations,
and an incident-safe draft built directly from the contract. The model may improve the draft's
style but may not change its facts. The output is checked for unsupported labels, missing citation,
unhedged low-confidence values, foreign script, broken structure, and a small set of high-impact
semantic errors. A failed answer is repaired up to twice. If it still fails, the system renders the
safe draft rather than returning an unsafe answer.

## Evaluation

`scripts/eval_llm.py` evaluated all three prompts on 24 contracts (72 outputs). The set covers every
action class, offence and no-offence decisions, card and no-card decisions, known and null colours,
low-confidence attributes, and two abstention cases. The figures below are from the final hybrid
retrieval run with `gpt-5-nano` on 16 September 2026.

| Metric | v1 | v2 | v3 |
|---|---:|---:|---:|
| Mean label faithfulness | 0.784 | **1.000** | **1.000** |
| Outputs with unsupported label claims | 54.2% | **0.0%** | **0.0%** |
| Outputs citing a retrieved article | 91.7% | **91.7%** | **91.7%** |
| Correct abstention | 100.0% | **100.0%** | **100.0%** |
| Hedging on non-abstaining low-confidence cases | 33.3% | **100.0%** | **100.0%** |

The citation denominator includes the two correctly abstained cases. They deliberately retrieve no
article and call no API, so the maximum observed all-case citation rate is 22/24 = 91.7%.

A representative v1 high-leg answer invented a hand signal, possible penalty-area location, and
potential harm, none of which existed in the contract. For the same case, v2 states only that the
contract reports an offence, a card with no specified colour, raised-foot action, and contact; it
then cites the dangerous-play-with-contact article. This is the practical gain from the safe draft
and response guard.

Ten v3 outputs were also audited manually. Every incident claim was marked individually as either
supported by the contract, supported by a retrieved law, or wrong. The final audit passed 10/10:
no added visual fact, sanction, colour, or unhedged low-confidence value remained. The detailed
rows are in the local `artifacts/llm/manual_audit.csv`.

## Prompt v3 — the full ruling after the clip (final rerun 16 September 2026)

v3 is now the default. It explains the whole ruling in six lines:

```
القرار: ...
العقوبة الفنية: ...          restart: free kick, indirect free kick, penalty kick
العقوبة الانضباطية: ...      disciplinary sanction: none, caution, sending-off
المادة: القانون 12 — ...: «quoted Law text»
لماذا تُعد مخالفة: ...        why it is (or is not) a violation
مستوى الثقة: ...
```

Only the «why» line is written by the model. The restart and the sanction come from
`src/llm/ruling.py`, a table of Law 12 rules keyed on the contract, each tied to the
corpus chunks it rests on, and those chunks are always among the cited articles.
When the contract lacks a fact the Law depends on — where the foul happened (penalty
area), whether a raised foot made contact, the card colour — the rule is stated
conditionally rather than guessed. The same claim check, repair loop and safe
fallback as v2 apply; the deterministic ruling passes the faithfulness check with
no unsupported claims on every contract type in `tests/llm/test_ruling.py`.
The final live rerun confirms 1.000 mean faithfulness, zero unsupported outputs,
100% correct abstention, and 100% hedging on low-confidence non-abstaining cases.

## Abstention and uncertainty

When offence confidence is below 0.60, `explain()` returns the fixed Arabic human-review message
before importing or constructing the OpenAI client. This is covered by a mocked-client test. A
lower-confidence secondary field does not stop the explanation; its incident statement is prefixed
with `على الأرجح`, or the field is omitted. A null `card_colour` is stated as unknown and never
converted into yellow or red.

## Production verification and cost

The integrated FastAPI/Next.js application was exercised with the real
`hakam-refit-v3` checkpoint and an approved SoccerNet-MVFoul clip. The warm CPU
container completed vision inference, retrieval, live generation, claim checking,
and response rendering in 3.2 seconds. The uploaded clip produced an offence/no-card
contract and a complete ruling with no unsupported claims. The same production image
also passed frontend delivery, health/readiness, model-head, threshold, and request-ID
checks.

A separately instrumented v3 explanation used one OpenAI request, 1,549 input tokens,
76 output tokens, and 1.704 seconds. At the published GPT-5 nano rates of $0.05 per
million input tokens and $0.40 per million output tokens, that sample cost approximately
**$0.000108**. Actual cost can be higher when the repair loop retries. Request IDs and
token usage are recorded in server logs without logging prompts or secrets.

| Case | Engine | Seconds | Unsupported | Restart | Sanction | Evidence |
|---|---|---:|---:|---|---|---|
| Mock offence + card | grounded LLM | 1.704 | 0 | correct | correct/conditional colour | instrumented v3 demo |
| Approved uploaded clip | grounded LLM | 3.2 warm CPU | 0 | correct | correct (no card) | container smoke test |
| Low-confidence offence | abstention, no API | n/a | 0 | n/a | n/a | evaluation + unit test |
| Provider unavailable/key absent | deterministic v3 | tested | 0 | correct | correct | backend fallback test |

The pricing calculation uses the official
[GPT-5 nano model page](https://developers.openai.com/api/docs/models/gpt-5-nano).

## Limitations

The automatic faithfulness score is lexicon-based. It reliably catches Arabic surface forms for
contract labels and conditional card-colour claims, but it cannot understand every paraphrase or
all possible contradictions. Targeted semantic guards and the ten-item manual audit cover errors
observed during development, but broader human evaluation is still required before deployment.

Retrieval quality depends on the manually curated tags and summaries. The corpus is intentionally
narrow and does not cover every IFAB situation. Finally, a faithful explanation is not proof that
the upstream CV decision is correct: when the real vision model supplies a wrong label, the
language layer may accurately explain that wrong input. CV accuracy and LLM faithfulness must be
reported as separate measurements.
