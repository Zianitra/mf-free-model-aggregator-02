# free-model-aggregator-02

A pure-stdlib Python HTTP server that aggregates free LLM models (via OpenRouter
free upstreams) behind OpenAI- and Anthropic-compatible APIs.

Reads the model catalog and quality ratings from `brief_local.txt` (JSON:
`models.ratings[]` with per-model capability flags), then routes every request
to the highest-rated free model that satisfies the request's capability
requirements (tools / vision / json).

## Features

- `POST /v1/chat/completions` — OpenAI-compatible, streaming (SSE) and non-streaming
- `POST /v1/messages` — Anthropic-compatible, streaming and non-streaming
  (payloads are converted to the OpenAI shape for upstream, responses back to Anthropic shape)
- `GET /v1/models` — list aggregated models with ratings and context lengths
- `GET /health` — liveness check
- Quality router: `0.6 * rating_milli` + deterministic seeded jitter, plus a small
  health bonus for models with recent successes
- Per-model cooldown after failures with exponential backoff (2s base, 300s cap),
  and automatic failover to the next-best healthy model
- Only Python standard library — no third-party dependencies

## Run

```bash
export OPENROUTER_API_KEY=sk-or-...        # required
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   # optional override
export PORT=8080                            # optional, default 8080
python3 main.py
```

## Self-test

```bash
python3 main.py --selftest   # offline router/needs/conversion checks; exits 0 on success
```

## Example

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'

curl -s http://localhost:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","max_tokens":256,"messages":[{"role":"user","content":"hi"}]}'
```

The requested `model` field is advisory; routing is decided by the quality router.

## License

MIT — see [LICENSE](LICENSE).
