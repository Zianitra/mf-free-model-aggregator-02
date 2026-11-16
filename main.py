#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Free-model aggregator: OpenAI/Anthropic-compatible proxy over OpenRouter free models."""
import json
import os
import random
import threading
import time

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
        self._lock = threading.Lock()
        self._fail_counts = {}   # model -> consecutive failures
        self._cooldown_until = {}  # model -> monotonic time until which it is cooling down
        self._rng = random.Random(seed)

    def jitter_for(self, model):
        """Deterministic per-request jitter derived from the model name."""
        h = self._rng  # keep rng deterministic per Router instance ordering
        # stable hash without PYTHONHASHSEED dependence
        val = 0
        for ch in model:
            val = (val * 1000003 + ord(ch)) & 0xFFFFFFFF
        jitter = (val % (2 * JITTER_MILLI + 1)) - JITTER_MILLI
        h.random()  # advance rng deterministically
        return jitter

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
        return 0.6 * m["rating_milli"] + self.jitter_for(m["model"])

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
            n = self._fail_counts.get(model, 0) + 1
            self._fail_counts[model] = n
            cd = min(FAIL_BASE_COOLDOWN * (2 ** (n - 1)), FAIL_MAX_COOLDOWN)
            self._cooldown_until[model] = time.monotonic() + cd
            return cd

    def record_success(self, model):
        with self._lock:
            self._fail_counts[model] = 0
            self._cooldown_until.pop(model, None)


if __name__ == "__main__":
    pass  # HTTP server + self-test appended in later commits
