# Hakam final local checklist

**Delivery:** local presentation laptop only · online deployment intentionally out of scope

## Product integration

- [x] Real `hakam-refit-v3` checkpoint downloaded and SHA-256 recorded locally.
- [x] Validation thresholds verified from `final_v3/metrics.json`: offence 0.60, card 0.50, body part 0.49.
- [x] Checkpoint strictly loads offline with the expected four production heads.
- [x] Next.js frontend calls the real local FastAPI API.
- [x] FastAPI streams and validates one video, caps uploads at 200 MB, serializes inference, and deletes temporary files.
- [x] Readiness, liveness, request IDs, safe errors, security headers, and deterministic fallback verified.
- [x] Real approved clip passed the local smoke test with a complete grounded ruling and no unsupported claims.

## Language layer — Anas

- [x] OpenAI key detected locally without entering Git or logs.
- [x] Prompt v3 live demo returns all six Arabic ruling sections.
- [x] Final 72-call evaluation complete: v3 faithfulness 1.000, unsupported outputs 0%, abstention 100%, low-confidence hedging 100%.
- [x] Ten v3 outputs manually audited: 10/10 pass, zero wrong claims.
- [x] Arabic wording fixes for simulation, severity inference, and canonical action phrasing covered by regression tests.
- [x] One live request measured: 1,549 input tokens, 76 output tokens, 1.704 seconds, approximately $0.000108.
- [x] `docs/llm/REPORT.md`, `README.md`, and `docs/RESULTS.md` updated.
- [ ] Confirm the OpenAI project budget/alert in the owner account (account-owner action).

## Quality evidence

- [x] Python suite: 153 passed (re-run 16 Sep during the audit).
- [x] Frontend invariant verification passed.
- [x] TypeScript check passed.
- [x] Optimized frontend build passed.
- [x] Local real-checkpoint smoke test passed.
- [ ] Presentation-machine responsive/mobile visual pass (manual screen check).
- [x] Presentation deck created (team is finalising the slides).

## Final local release

- [ ] Run the final API and frontend together on the presentation laptop.
- [ ] Run `scripts/smoke_test.py` with an approved local clip.
- [ ] Verify live OpenAI generation and the deterministic offline fallback.
- [x] Integrated branch merged to `main` (PR #2).
- [ ] Keep the local checkpoint and a tested Python/Node environment on the presentation laptop.

No Hugging Face Space, public URL, or online deployment secrets are required.

## Audit — 16 September

- [x] Full test of the final model on all 301 test incidents reproduces the reported results ([`docs/AUDIT.md`](docs/AUDIT.md)).
- [x] Offline ruling on every answered test decision: six sections, 0 unsupported claims, correct Law 12 articles.
- [x] Backend warms up the retrieval model at start (no slow first analysis).
- [x] Website **Test yourself** section with the five highest-scoring test incidents (clips in `web/public/samples/`).
- [x] One-command local demo: `run_demo.bat`.
- [ ] Commit the final presentation.

