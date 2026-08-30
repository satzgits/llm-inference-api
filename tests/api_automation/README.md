# API Test Automation Suite

A contract / integration test suite for the LLM Inference API, structured the way a
**Test Automation** engineer would build it — covering REST API validation, error
handling, response schema, and smoke tests.

This is deliberately organised in two layers:

- **Contract tests (default, no server needed)** — exercise the FastAPI app via
  `TestClient`, covering input validation (422), upstream failure translation
  (504/502), response schema, and unknown routes (404). These run anywhere, offline,
  and are the bulk of the suite.
- **Live smoke tests (`RUN_LIVE=1`)** — hit a real running API end-to-end via `httpx`.
  Only run when you actually have the server up.

## Run

```bash
# Contract mode (no server needed) — everything except live smoke
python tests/api_automation/test_api_automation.py

# Live/end-to-end mode (needs the API running)
$env:RUN_LIVE="1"
$env:API_BASE_URL="http://localhost:8008"
python tests/api_automation/test_api_automation.py
```

Expect `13 passed` in contract mode.

## What it covers

| Endpoint | Tests |
|---|---|
| `GET /health` | status + schema |
| `GET /models` | response shape, model list & count |
| `POST /generate` | happy-path schema, missing prompt → 422, invalid temperature → 422, negative tokens → 422, upstream timeout → 504, upstream error → 502 |
| `POST /embed` | vector shape, empty input → 422 |
| any route | unknown route → 404 |
| live | `GET /models`, `POST /generate` against a running server |

This suite demonstrates API testing + test automation — directly aligned with the
"Test Automation / API Testing" skills an AI Software Engineer role looks for.
