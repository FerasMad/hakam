# Laws corpus

`corpus.json` contains 47 manually curated, bilingual rule summaries from the
IFAB Laws of the Game 2026/27. Each chunk is one meaningful rule rather than a
fixed-size slice. `source_pages` records the page(s) used in the Arabic and
English single-page editions.

## Sources

- English: <https://downloads.theifab.com/downloads/laws-of-the-game-202627-single-pages?l=en>
- Arabic: <https://downloads.theifab.com/downloads/laws-of-the-game-202627-arabic-single-pages?l=en>

The source PDFs belong in `laws/raw/` and are intentionally ignored by Git.
Download them with:

```bash
mkdir -p laws/raw
curl -fL 'https://downloads.theifab.com/downloads/laws-of-the-game-202627-single-pages?l=en' \
  -o laws/raw/ifab_laws_2026_27_en.pdf
curl -fL 'https://downloads.theifab.com/downloads/laws-of-the-game-202627-arabic-single-pages?l=en' \
  -o laws/raw/ifab_laws_2026_27_ar.pdf
```

## Text-layer check

Checked with `pypdf` on 14 September 2026:

| Edition | Pages | Pages with extractable text | Extracted characters |
|---|---:|---:|---:|
| English | 260 | 257 | 242,782 |
| Arabic | 236 | 235 | 208,743 (157,438 Arabic characters) |

The Arabic edition therefore has a real text layer and does not require OCR.
Page 111 (Arabic) and page 115 (English), the opening page of Law 12, were also
rendered and visually checked against the extracted text.

The corpus uses concise faithful summaries, not blind copies of page text. This
keeps each retrieval unit focused and makes the cited rule easier to audit.

## Retrieval index

`embeddings.npy` holds one normalised `intfloat/multilingual-e5-base` vector per chunk;
`embeddings.json` records the model and a hash of the corpus text. Both are committed, so
retrieval works out of the box. After editing `corpus.json`, rebuild with:

```bash
python scripts/build_index.py
```

The index is also rebuilt automatically on the first query if the hash no longer matches.
