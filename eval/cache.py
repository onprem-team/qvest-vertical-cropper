# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest Group GmbH
# SPDX-License-Identifier: Apache-2.0
"""On-disk VLM response cache for the eval harness.

Wraps any VisionBackend and memoizes ``complete()`` keyed by
(model, messages, generation params) so parity iteration re-runs replay for free
(mirrors the R&D vlm_cache). Never used by the shipped tool; eval-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CachingBackend:
    """VisionBackend decorator that caches completions on disk.

    Delegates model/usage to the wrapped backend. On a cache hit it returns the
    stored content without an API call (so token totals reflect only real spend);
    ``cache_hits`` counts replays.
    """

    def __init__(self, inner, cache_dir):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hits = 0

    @property
    def model(self):
        return self.inner.model

    @property
    def last_usage(self):
        return self.inner.last_usage

    @property
    def usage_totals(self):
        return self.inner.usage_totals

    def _key(self, messages, kwargs):
        h = hashlib.sha256()
        identity = getattr(self.inner, "cache_identity", {"model": self.inner.model})
        h.update(json.dumps(identity, sort_keys=True, default=str).encode())
        h.update(json.dumps(messages, sort_keys=True, default=str).encode())
        h.update(json.dumps(kwargs, sort_keys=True, default=str).encode())
        return h.hexdigest()[:32]

    def complete(self, messages, **kwargs):
        cpath = self.cache_dir / (self._key(messages, kwargs) + ".json")
        if cpath.exists():
            self.cache_hits += 1
            return json.loads(cpath.read_text())["content"]
        content = self.inner.complete(messages, **kwargs)
        tmp = cpath.with_suffix(".tmp")
        tmp.write_text(json.dumps({"content": content}))
        tmp.replace(cpath)
        return content
