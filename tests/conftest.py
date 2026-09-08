"""Shared test fixtures: synthetic clips + fake VLM backends (all offline)."""
from __future__ import annotations

import threading

import cv2
import numpy as np


def write_clip(path, n_frames=30, w=160, h=90):
    """Write a tiny solid-color mp4 with distinguishable frames."""
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class FakeBackend:
    """In-memory VisionBackend for offline tests.

    - `responses`: a single string returned for every call, or an iterable of
      strings returned in order (last value repeats once exhausted).
    - `raise_exc`: if set, `complete` raises it instead of returning.
    - records every `messages` payload it received in `calls` for assertions.
    """

    def __init__(self, responses="{}", *, model="fake-model",
                 raise_exc=None, usage=None):
        self.model = model
        self._raise = raise_exc
        self.calls: list[list[dict]] = []
        self.call_kwargs: list[dict] = []  # generation params per call (max_tokens, etc.)
        self._per_call = usage or {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        self.last_usage = dict(self._per_call)
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
        self._usage_lock = threading.Lock()
        if isinstance(responses, str):
            self._single = responses
            self._iter = None
        else:
            self._single = None
            self._iter = iter(responses)
            self._last = None

    def complete(self, messages, *, temperature=0.1, response_format=None,
                 max_tokens=None, extra_body=None):
        self.calls.append(messages)
        self.call_kwargs.append({"temperature": temperature, "response_format": response_format,
                                 "max_tokens": max_tokens, "extra_body": extra_body})
        if self._raise is not None:
            raise self._raise
        # Mimic the real backend: record per-call usage + aggregate totals (locked).
        with self._usage_lock:
            self.last_usage = dict(self._per_call)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self.usage_totals[k] += self._per_call[k]
            self.usage_totals["api_calls"] += 1
        if self._single is not None:
            return self._single
        try:
            self._last = next(self._iter)
        except StopIteration:
            pass
        return self._last


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeUsage:
    def __init__(self, p=10, c=5, t=15):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = t


class FakeResponse:
    def __init__(self, content, usage=None):
        self.choices = [FakeChoice(content)] if content is not None else []
        self.usage = usage if usage is not None else FakeUsage()


class FakeCompletions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        exc_seq = self._owner.raise_seq
        if exc_seq:
            exc = exc_seq.pop(0)
            if exc is not None:
                raise exc
        content = self._owner.content
        if not isinstance(content, str) and content is not None:
            content = next(self._owner._content_iter)
        return FakeResponse(content, usage=self._owner.usage)


class FakeOpenAIClient:
    """Drop-in for openai.OpenAI: records constructor kwargs + create() calls.

    `content`: str returned as message content (or an iterable for successive calls).
    `raise_seq`: list of exceptions/None applied per call before returning content.
    """

    instances: list[FakeOpenAIClient] = []

    def __init__(self, content="{}", *, raise_seq=None, usage=None):
        # Default construction values; per-instance state set when SDK calls us.
        self.content = content
        # str/None are single values; anything else is an iterable of contents.
        self._content_iter = None if isinstance(content, (str, type(None))) else iter(content)
        self.raise_seq = list(raise_seq) if raise_seq else []
        self.usage = usage
        self.calls: list[dict] = []
        self.init_kwargs: dict = {}
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(self)

    @classmethod
    def factory(cls, content="{}", *, raise_seq=None, usage=None):
        """Return a callable matching openai.OpenAI(**kwargs) that yields a shared instance."""
        inst = cls(content=content, raise_seq=raise_seq, usage=usage)

        def _ctor(**kwargs):
            inst.init_kwargs = kwargs
            cls.instances.append(inst)
            return inst

        _ctor.instance = inst
        return _ctor
