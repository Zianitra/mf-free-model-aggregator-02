#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Free-model aggregator: OpenAI/Anthropic-compatible proxy over OpenRouter free models."""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BRIEF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brief_local.txt")
JITTER_SEED = 0x5EED
JITTER_MILLI = 30  # max deterministic jitter in milli-rating
FAIL_BASE_COOLDOWN = 2.0
FAIL_MAX_COOLDOWN = 300.0


def load_brief(path=BRIEF_PATH):
    """Load brief_local.txt which may contain one or more concatenated JSON objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    dec = json.JSONDecoder()
    docs = []
    idx = 0
    while idx < len(raw):
        while idx < len(raw) and raw[idx] not in "{[":
            idx += 1
        if idx >= len(raw):
            break
        obj, end = dec.raw_decode(raw, idx)
        docs.append(obj)
        idx = end
    models_doc = None
    for d in docs:
        if isinstance(d, dict) and "models" in d:
            models_doc = d["models"]
            break
    if models_doc is None or "ratings" not in models_doc:
        raise ValueError("brief_local.txt: no models.ratings found")
    ratings = []
    for r in models_doc["ratings"]:
        ratings.append({
            "model": r["model"],
            "upstream": r.get("upstream") or r["model"],
            "rating_milli": int(r.get("rating_milli", 0)),
            "context_length": int(r.get("context_length", 0)),
            "tools": bool(r.get("tools", False)),
            "vision": bool(r.get("vision", False)),
            "json": bool(r.get("json", False)),
            "free": bool(r.get("free", True)),
        })
    return ratings


class Router:
    """Quality router with deterministic jitter, cooldowns and failover ordering."""

    def __init__(self, models, seed=JITTER_SEED):
        self.models = models
        self.seed = seed
        self._lock = threading.Lock()
        self._fail_counts = {}   # model -> consecutive failures
        self._success_streak = {}  # model -> consecutive successes
        self._cooldown_until = {}  # model -> monotonic time until which it is cooling down

    def jitter_for(self, model):
        """Deterministic per-model jitter in [-JITTER_MILLI, +JITTER_MILLI], seeded."""
        val = self.seed & 0xFFFFFFFF
        for ch in model:
            val = (val * 1000003 + ord(ch)) & 0xFFFFFFFF
        val ^= val >> 13
        val = (val * 2654435761) & 0xFFFFFFFF
        return (val % (2 * JITTER_MILLI + 1)) - JITTER_MILLI

    def health_bonus(self, model):
        """Small deterministic bonus for models with a live success streak."""
        streak = self._success_streak.get(model, 0)
        return min(streak, 3) * 2.0

    def score(self, m, needs, now=None):
        """Score a candidate; higher is better. Excludes cooldowns and unmet capabilities."""
        if m.get("free") is False:
            return None
        for flag in ("tools", "vision", "json"):
            if needs.get(flag) and not m[flag]:
                return None
        if now is None:
            now = time.monotonic()
        if self._cooldown_until.get(m["model"], 0.0) > now:
            return None
        return 0.6 * m["rating_milli"] + self.jitter_for(m["model"]) + self.health_bonus(m["model"])

    def rank(self, needs=None, now=None):
        """Return candidate models sorted best-first, satisfying needs and health."""
        needs = needs or {}
        now = now if now is not None else time.monotonic()
        scored = []
        for m in self.models:
            s = self.score(m, needs, now=now)
            if s is not None:
                scored.append((s, m["model"], m))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [m for _, _, m in scored]

    def record_failure(self, model):
        with self._lock:
            self._success_streak.pop(model, None)
            n = self._fail_counts.get(model, 0) + 1
            self._fail_counts[model] = n
            cd = min(FAIL_BASE_COOLDOWN * (2 ** (n - 1)), FAIL_MAX_COOLDOWN)
            self._cooldown_until[model] = time.monotonic() + cd
            return cd

    def record_success(self, model):
        with self._lock:
            self._fail_counts[model] = 0
            self._success_streak[model] = self._success_streak.get(model, 0) + 1
            self._cooldown_until.pop(model, None)


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
PORT = int(os.environ.get("PORT", "8080"))


def upstream_request(model, payload, timeout=120):
    """POST to OpenRouter chat/completions with the upstream model id. Returns dict."""
    body = dict(payload)
    body["model"] = model
    req = urllib.request.Request(
        OPENROUTER_BASE_URL + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENROUTER_API_KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upstream_stream(model, payload, timeout=120):
    """Yield raw SSE lines from OpenRouter for a streaming request."""
    body = dict(payload)
    body["model"] = model
    body["stream"] = True
    req = urllib.request.Request(
        OPENROUTER_BASE_URL + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENROUTER_API_KEY,
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    try:
        for raw in resp:
            yield raw.decode("utf-8", errors="replace")
    finally:
        resp.close()


def detect_needs(payload):
    """Infer capability requirements from an OpenAI or Anthropic payload."""
    needs = {}
    if payload.get("response_format", {}).get("type") in ("json_object", "json_schema"):
        needs["json"] = True
    for m in payload.get("messages") or []:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image", "image_url", "input_image"):
                    needs["vision"] = True
        if m.get("tool_calls") or m.get("tool_call_id"):
            needs["tools"] = True
    if payload.get("tools"):
        needs["tools"] = True
    return needs


def parse_needs_from_requires(requires):
    needs = {}
    for k in ("tools", "vision", "json"):
        if requires and requires.get(k):
            needs[k] = True
    return needs


def convert_openai_to_upstream(payload):
    """Payload is already OpenAI-shaped; return as-is (json mode mapped to response_format)."""
    return payload


def anthropic_to_openai(payload):
    """Convert Anthropic /v1/messages payload to OpenAI chat/completions shape."""
    msgs = []
    system_text = ""
    sys = payload.get("system")
    if isinstance(sys, str):
        system_text = sys
    elif isinstance(sys, list):
        system_text = "\n".join(b.get("text", "") for b in sys if isinstance(b, dict))
    for m in payload.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content")
        if role == "assistant":
            role = "assistant"
        elif role == "user":
            role = "user"
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": b.get("text", "")})
                elif t == "image":
                    src = b.get("source", {})
                    if src.get("type") == "base64":
                        parts.append({"type": "image_url", "image_url": {
                            "url": "data:%s;base64,%s" % (src.get("media_type", "image/png"), src.get("data", ""))}})
                    elif src.get("type") == "url":
                        parts.append({"type": "image_url", "image_url": {"url": src.get("url", "")}})
                elif t == "tool_use":
                    msgs.append({"role": "assistant", "content": None,
                                 "tool_calls": [{"id": b.get("id"), "type": "function",
                                                 "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input", {}))}}]})
                    continue
                elif t == "tool_result":
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                                 "content": json.dumps(b.get("content")) if not isinstance(b.get("content"), str) else b.get("content")})
                    continue
            if parts:
                msgs.append({"role": role, "content": parts})
    out = {"model": None, "messages": msgs, "max_tokens": payload.get("max_tokens", 1024)}
    if system_text:
        out["messages"] = [{"role": "system", "content": system_text}] + out["messages"]
    if payload.get("tools"):
        out["tools"] = [{
            "type": "function",
            "function": {"name": t.get("name"), "description": t.get("description", ""),
                         "parameters": t.get("input_schema", {})}
        } for t in payload["tools"]]
    return out


def openai_to_anthropic_response(chunks_or_resp):
    """Convert a non-streaming OpenAI response to Anthropic /v1/messages shape."""
    ch = chunks_or_resp
    choice = (ch.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    content = [{"type": "text", "text": text}] if text else []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        content.append({"type": "tool_use", "id": tc.get("id", "call_0"), "name": fn.get("name"), "input": args})
    stop = choice.get("finish_reason")
    stop_reason = "end_turn"
    if stop == "length":
        stop_reason = "max_tokens"
    elif stop == "tool_calls":
        stop_reason = "tool_use"
    usage = ch.get("usage", {}) or {}
    return {
        "id": ch.get("id", "msg_0"),
        "type": "message",
        "role": "assistant",
        "model": ch.get("model", ""),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "free-model-aggregator/1.0"
    router = None  # set by serve()

    def log_message(self, fmt, *args):
        pass  # keep stdout quiet

    def _json(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def do_GET(self):
        if self.path == "/v1/models":
            models = [{"id": m["model"], "object": "model", "owned_by": "openrouter",
                       "context_length": m["context_length"], "rating_milli": m["rating_milli"]}
                      for m in self.router.models]
            self._json(200, {"object": "list", "data": models})
        elif self.path == "/health":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def _chunk_write(self, text):
        data = text.encode("utf-8")
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))

    def _sse(self, obj, event=None):
        body = ""
        if event:
            body += "event: %s\n" % event
        body += "data: %s\n\n" % json.dumps(obj)
        self._chunk_write(body)

    def _end_stream(self):
        self.wfile.write(b"0\r\n\r\n")

    # ---- request routing with failover ----

    def _route_and_call(self, payload, needs, stream, sink):
        """Try models best-first; on upstream failure record cooldown and failover.
        sink(chunk_dict, model) is called per SSE chunk / with the final response."""
        candidates = self.router.rank(needs)
        if not candidates:
            raise RuntimeError("no healthy model satisfies requirements: %s" % (needs or "none"))
        last_err = None
        for m in candidates:
            try:
                if stream:
                    buf = ""
                    for raw in upstream_stream(m["upstream"], payload):
                        buf += raw
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip("\r")
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(data)
                            except ValueError:
                                continue
                            sink(chunk, m["model"])
                    self.router.record_success(m["model"])
                    return m["model"]
                else:
                    resp = upstream_request(m["upstream"], payload)
                    self.router.record_success(m["model"])
                    sink(resp, m["model"])
                    return m["model"]
            except Exception as e:  # failover
                self.router.record_failure(m["model"])
                last_err = e
        raise RuntimeError("all candidate models failed: %s" % last_err)

    def do_POST(self):
        try:
            payload = self._read_body()
        except ValueError:
            return self._json(400, {"error": {"message": "invalid JSON body"}})
        if self.path == "/v1/chat/completions":
            return self._handle_chat(payload)
        if self.path == "/v1/messages":
            return self._handle_messages(payload)
        return self._json(404, {"error": {"message": "not found"}})

    def _handle_chat(self, payload):
        needs = detect_needs(payload)
        stream = bool(payload.get("stream"))
        if not stream:
            holder = {}
            def sink(resp, model):
                resp["model"] = model
                holder["resp"] = resp
            try:
                self._route_and_call(convert_openai_to_upstream(payload), needs, False, sink)
            except RuntimeError as e:
                return self._json(502, {"error": {"message": str(e)}})
            return self._json(200, holder["resp"])
        # streaming: re-emit OpenRouter SSE chunks as OpenAI-compatible chunks
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            def sink(chunk, model):
                chunk["model"] = model
                self._sse(chunk)
            self._route_and_call(convert_openai_to_upstream(payload), needs, True, sink)
            self._end_stream()
        except (RuntimeError, BrokenPipeError) as e:
            data = json.dumps({"error": {"message": str(e)}}).encode("utf-8")
            self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(data), data))

    def _handle_messages(self, payload):
        needs = detect_needs(anthropic_to_openai(payload))
        needs.update(parse_needs_from_requires(payload.get("requires")))
        stream = bool(payload.get("stream"))
        converted = anthropic_to_openai(payload)
        if not stream:
            holder = {}
            def sink(resp, model):
                holder["resp"] = resp
                holder["model"] = model
            try:
                self._route_and_call(converted, needs, False, sink)
            except RuntimeError as e:
                return self._json(502, {"error": {"message": str(e)}})
            out = openai_to_anthropic_response(holder["resp"])
            out["model"] = holder["model"]
            return self._json(200, out)
        # Anthropic SSE: emit message_start / content_block / message_delta events
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        state = {"started": False, "model": "", "out_tokens": 0, "open_block": None}
        def emit(event, data):
            self._sse(data, event=event)
        def sink(chunk, model):
            state["model"] = model
            if not state["started"]:
                state["started"] = True
                emit("message_start", {"type": "message_start", "message": {
                    "id": "msg_0", "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta", {}) or {}
                txt = delta.get("content")
                if txt:
                    if state["open_block"] != "text":
                        state["open_block"] = "text"
                        emit("content_block_start", {"type": "content_block_start", "index": 0,
                                                     "content_block": {"type": "text", "text": ""}})
                    state["out_tokens"] += 1
                    emit("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": txt}})
        try:
            self._route_and_call(converted, needs, True, sink)
            if state["open_block"] == "text":
                emit("content_block_stop", {"type": "content_block_stop", "index": 0})
            emit("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                   "usage": {"output_tokens": state["out_tokens"]}})
            emit("message_stop", {"type": "message_stop"})
            self.wfile.write(b"0\r\n\r\n")
        except (RuntimeError, BrokenPipeError) as e:
            data = json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e)}}).encode("utf-8")
            self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(data), data))


def make_server(port=None, models=None):
    router = Router(models if models is not None else load_brief())
    Handler.router = router
    srv = ThreadingHTTPServer(("0.0.0.0", port if port is not None else PORT), Handler)
    srv.router = router
    return srv


def selftest():
    """Offline self-test: route fake requests through the router logic. Exit 0 on success."""
    import sys
    models = load_brief()
    r = Router(models)
    assert len(models) >= 5, "expected >=5 models"

    # json requirement routes to best json-capable model
    cands = r.rank({"json": True})
    assert cands, "no candidates for json"
    assert cands[0]["json"], "top json candidate lacks json flag"

    # vision requirement excludes vision-less models
    cands = r.rank({"vision": True})
    assert cands and all(m["vision"] for m in cands[:5])

    # tools requirement
    cands = r.rank({"tools": True})
    assert cands and all(m["tools"] for m in cands[:5])

    # combined requirements
    cands = r.rank({"tools": True, "vision": True, "json": True})
    assert cands and all(m["tools"] and m["vision"] and m["json"] for m in cands[:5])

    # highest rated free model ranks first with no requirements
    best = max(models, key=lambda m: m["rating_milli"])
    top = r.rank({})[0]
    assert top["model"] == best["model"], "top model should be highest rated: %s != %s" % (top["model"], best["model"])

    # deterministic scoring: same router instance, same score twice
    s1 = r.score(models[0], {})
    s2 = r.score(models[0], {})
    assert s1 == s2, "scores must be deterministic"

    # cooldown: failing model is excluded, then recovers
    victim = r.rank({})[-1]["model"]
    r.record_failure(victim)
    assert victim not in [m["model"] for m in r.rank({})], "cooled-down model still ranked"
    r.record_success(victim)
    assert victim in [m["model"] for m in r.rank({})], "model did not recover"

    # exponential backoff grows
    r2 = Router(models)
    cd1 = r2.record_failure("x")
    cd2 = r2.record_failure("x")
    assert cd2 == 2 * cd1, "backoff not exponential: %s %s" % (cd1, cd2)

    # failover: first model fails, next succeeds
    order = r.rank({})
    first, second = order[0], order[1]
    r.record_failure(first["model"])  # simulate first failing
    new_top = r.rank({})[0]
    assert new_top["model"] != first["model"], "failover did not skip failed model"

    # needs detection from payloads
    needs = detect_needs({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})
    assert needs.get("vision"), "vision need not detected"
    needs = detect_needs({"response_format": {"type": "json_object"}, "messages": []})
    assert needs.get("json"), "json need not detected"
    needs = detect_needs({"tools": [{"type": "function", "function": {"name": "f"}}], "messages": []})
    assert needs.get("tools"), "tools need not detected"

    # anthropic conversion round-trip basics
    conv = anthropic_to_openai({"system": "be brief", "max_tokens": 10,
                                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
    assert conv["messages"][0]["role"] == "system" and conv["max_tokens"] == 10
    out = openai_to_anthropic_response({"id": "a", "model": "m", "choices": [{"finish_reason": "stop",
        "message": {"role": "assistant", "content": "yo"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    assert out["content"][0]["text"] == "yo" and out["usage"]["output_tokens"] == 2

    print("selftest OK: %d models, top=%s (%d milli)" % (len(models), best["model"], best["rating_milli"]))
    return 0


def serve():
    srv = make_server()
    print("listening on :%d with %d free models" % (srv.server_address[1], len(Handler.router.models)))
    srv.serve_forever()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    serve()