# Hakam web frontend

This is the Next.js interface for one-video Hakam incident review. Its typed
`AnalysisService` sends a single multipart field named `video` to the local
FastAPI backend; UI components do not call the backend directly.

## Run locally

Start the FastAPI backend first (see `../backend/README.md`), then:

```bash
cd web
cp .env.example .env.local
npm ci
npm run dev
```

On Windows Command Prompt, use `copy .env.example .env.local`. Open
`http://localhost:3000`, select one browser-playable video, and keep the
incident near the clip midpoint.

`NEXT_PUBLIC_HAKAM_API_URL` defaults to `http://localhost:8000`. It is the only
frontend environment variable required for local integration. Never expose
`OPENAI_API_KEY` through a `NEXT_PUBLIC_` variable.

## Production flow

`lib/services/analysis.ts` is the sole HTTP integration point. The former
prototype scenario service and selector have been removed; every normal analysis
request now goes to the real local backend.

## Verify

```bash
npm run typecheck
npm run verify
npm run build
```
