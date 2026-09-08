"""Offline unit tests for the eval harness (eval/run_eval.py, eval/cache.py).

Crop geometry lives in v_cropper.render (compute_crop_path) and is tested in
test_render.py; here we cover the eval-only pieces: manifest regeneration, the response
cache, the scorer-schema round-trip, and the no-ground-truth-leakage invariant.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

import pytest

from conftest import FakeBackend, write_clip

_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
sys.path.insert(0, _EVAL_DIR)

import render_debug  # noqa: E402
import run_eval  # noqa: E402
import score as scorer  # noqa: E402
from cache import CachingBackend  # noqa: E402

from v_cropper.backend import (  # noqa: E402
    BackendConfig,
    BedrockBackend,
    BedrockConfig,
    OpenAICompatBackend,
)
from v_cropper.render import compute_crop_path  # noqa: E402


class TestBuildManifest:
    def test_decode_count_matches(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        write_clip(raw / "a.mp4", n_frames=25, w=160, h=90)
        man = run_eval.build_manifest(raw, ["a"], tmp_path / "manifest.json")
        assert man["a"]["n_frames"] == 25
        assert (man["a"]["width"], man["a"]["height"]) == (160, 90)
        assert (tmp_path / "manifest.json").exists()

    def test_windows_match_manifest_count(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        clip = write_clip(raw / "a.mp4", n_frames=33, w=192, h=108)
        man = run_eval.build_manifest(raw, ["a"], tmp_path / "m.json")
        wins, _, _, _ = compute_crop_path(clip, {0: 90.0})
        assert len(wins) == man["a"]["n_frames"]


class _CountingBackend:
    model = "fake-model"

    def __init__(self, content="POINT"):
        self.content = content
        self.cache_identity = {"base_url": "http://one.example/v1", "model": self.model}
        self.n_calls = 0
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0,
                             "total_tokens": 0, "api_calls": 0}

    def complete(self, messages, **kwargs):
        self.n_calls += 1
        self.usage_totals["api_calls"] += 1
        return self.content


class TestCachingBackend:
    def test_hit_avoids_inner_call(self, tmp_path):
        inner = _CountingBackend("R1")
        cb = CachingBackend(inner, tmp_path / "cache")
        msgs = [{"role": "user", "content": "hi"}]
        assert cb.complete(msgs, temperature=0.1) == "R1"
        assert cb.complete(msgs, temperature=0.1) == "R1"  # served from disk
        assert inner.n_calls == 1
        assert cb.cache_hits == 1

    def test_distinct_kwargs_distinct_key(self, tmp_path):
        inner = _CountingBackend("R")
        cb = CachingBackend(inner, tmp_path / "cache")
        msgs = [{"role": "user", "content": "hi"}]
        cb.complete(msgs, temperature=0.1)
        cb.complete(msgs, temperature=0.9)  # different param -> miss
        assert inner.n_calls == 2
        assert cb.cache_hits == 0

    def test_delegates_metadata(self, tmp_path):
        inner = _CountingBackend()
        cb = CachingBackend(inner, tmp_path / "cache")
        assert cb.model == "fake-model"
        assert cb.usage_totals is inner.usage_totals

    def test_same_model_different_endpoint_is_not_a_cache_hit(self, tmp_path):
        inner = _CountingBackend("R")
        cb = CachingBackend(inner, tmp_path / "cache")
        msgs = [{"role": "user", "content": "hi"}]
        cb.complete(msgs, temperature=0.1)
        inner.cache_identity = {"base_url": "http://two.example/v1", "model": inner.model}
        cb.complete(msgs, temperature=0.1)
        assert inner.n_calls == 2
        assert cb.cache_hits == 0


class TestShippedBackendCacheIdentity:
    """The shipped backends must define ``cache_identity`` themselves.

    ``CachingBackend._key`` reads it through ``getattr(..., {"model": ...})``, so a backend
    that loses the attribute does not raise — it silently re-keys on the model alone and
    starts serving cached responses across different endpoints, headers, or regions. The
    tests above use a fake that hard-codes the attribute and would not catch that, so these
    pin it onto the real classes.
    """

    MESSAGES = [{"role": "user", "content": "hi"}]

    def _openai(self, **overrides):
        fields = {"base_url": "https://one.example/v1", "api_key": "secret-key", "model": "m"}
        fields.update(overrides)
        return OpenAICompatBackend(BackendConfig(**fields))

    def _key(self, backend, tmp_path, name="cache"):
        # Key directly rather than via complete() so no request is ever attempted.
        return CachingBackend(backend, tmp_path / name)._key(self.MESSAGES, {"temperature": 0.0})

    def test_openai_backend_defines_cache_identity(self):
        assert hasattr(OpenAICompatBackend, "cache_identity")
        identity = self._openai().cache_identity
        assert identity["base_url"] == "https://one.example/v1"
        assert identity["model"] == "m"
        assert set(identity) >= {
            "base_url", "model", "response_format", "extra_headers", "max_tokens", "extra_body"
        }

    def test_bedrock_backend_defines_cache_identity(self):
        assert hasattr(BedrockBackend, "cache_identity")
        identity = BedrockBackend(BedrockConfig(region="us-east-1", model="m")).cache_identity
        assert identity["provider"] == "bedrock"
        assert identity["region"] == "us-east-1"
        assert identity["model"] == "m"

    @pytest.mark.parametrize("field,value", [
        ("base_url", "https://two.example/v1"),
        ("extra_headers", {"X-Gateway-Client": "other"}),
        ("extra_body", {"reasoning_effort": "none"}),
        ("max_tokens", 64),
        ("response_format", {"type": "json_object"}),
    ])
    def test_each_identity_field_invalidates_the_cache(self, tmp_path, field, value):
        baseline = self._key(self._openai(), tmp_path)
        assert self._key(self._openai(**{field: value}), tmp_path) != baseline

    def test_same_model_across_providers_is_not_a_cache_hit(self, tmp_path):
        openai_key = self._key(self._openai(), tmp_path)
        bedrock_key = self._key(BedrockBackend(BedrockConfig(region="us-east-1", model="m")), tmp_path)
        assert openai_key != bedrock_key

    def test_same_model_across_bedrock_regions_is_not_a_cache_hit(self, tmp_path):
        east = self._key(BedrockBackend(BedrockConfig(region="us-east-1", model="m")), tmp_path)
        west = self._key(BedrockBackend(BedrockConfig(region="us-west-2", model="m")), tmp_path)
        assert east != west

    def test_cache_identity_never_carries_the_api_key(self):
        identity = self._openai().cache_identity
        assert "secret-key" not in json.dumps(identity, sort_keys=True, default=str)


class TestScorerIntegration:
    """End-to-end: our prediction schema scores correctly against the frozen scorer."""

    def _pred(self, item_id, x_center, n=4, w=192, h=108):
        dst_w = min(h * 9 // 16, w)
        return {
            "item_id": item_id,
            "output": {
                "crop_path": [{"frame_idx": i, "x_center": x_center, "y_center": h / 2,
                               "width": dst_w, "height": h} for i in range(n)],
                "frame_width": w, "frame_height": h, "fps": 30.0,
                "processing_time_sec": 1.0,
            },
        }

    def test_covered_point_scores_hit(self):
        man = {"a": {"n_frames": 4, "fps": 30.0, "width": 192, "height": 108}}
        preds = [self._pred("a", x_center=96.0)]
        truth = [{"item_id": "a", "frame_idx": 0, "x": 96.0, "y": 54.0}]
        assert scorer.evaluate(preds, truth, "val", manifest=man)["primary_metric"] == 1.0

    def test_far_point_scores_miss(self):
        man = {"a": {"n_frames": 4, "fps": 30.0, "width": 192, "height": 108}}
        preds = [self._pred("a", x_center=30.0)]  # window [0,60], label at x=190 -> miss
        truth = [{"item_id": "a", "frame_idx": 0, "x": 190.0, "y": 54.0}]
        assert scorer.evaluate(preds, truth, "val", manifest=man)["primary_metric"] == 0.0


class TestLeakageGuards:
    """Predictions never read labels; the manifest is the authoritative ruler."""

    def test_compute_crop_path_has_no_truth_param(self):
        params = set(inspect.signature(compute_crop_path).parameters)
        assert not ({"truth", "ground_truth", "labels"} & params)

    def test_manifest_dims_are_authoritative(self):
        man = {"a": {"n_frames": 4, "fps": 30.0, "width": 192, "height": 108}}
        pred = {"item_id": "a", "output": {
            "crop_path": [{"frame_idx": i, "x_center": 96.0, "y_center": 54.0,
                           "width": 60, "height": 108} for i in range(4)],
            "frame_width": 999, "frame_height": 108,  # lies about width
            "fps": 30.0, "processing_time_sec": 1.0}}
        truth = [{"item_id": "a", "frame_idx": 0, "x": 96.0, "y": 54.0}]
        with pytest.raises(ValueError, match="manifest probed"):
            scorer.evaluate([pred], truth, "val", manifest=man)

    def test_sparse_path_flagged_not_rewarded(self):
        man = {"a": {"n_frames": 4, "fps": 30.0, "width": 192, "height": 108}}
        pred = {"item_id": "a", "output": {  # only 2 of 4 frames -> padding/sparsity attempt
            "crop_path": [{"frame_idx": i, "x_center": 96.0, "y_center": 54.0,
                           "width": 60, "height": 108} for i in range(2)],
            "frame_width": 192, "frame_height": 108,
            "fps": 30.0, "processing_time_sec": 1.0}}
        truth = [{"item_id": "a", "frame_idx": 0, "x": 96.0, "y": 54.0}]
        r = scorer.evaluate([pred], truth, "val", manifest=man)
        assert r["constraint_violations"] > 0 and r["structural_ok"] is False


class TestRunEvalEndToEnd:
    def test_run_writes_predictions_metrics_and_debug_for_custom_endpoint(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        raw = data / "raw"
        raw.mkdir(parents=True)
        write_clip(raw / "a.mp4", n_frames=20, w=160, h=90)
        (data / "splits.json").write_text(json.dumps({"val": ["a"]}))
        (data / "val").mkdir()
        (data / "val" / "ground_truth.jsonl").write_text(
            json.dumps({"item_id": "a", "frame_idx": 10, "x": 80, "y": 45}) + "\n")
        out = tmp_path / "out"
        backend = FakeBackend(json.dumps({"x": 500, "y": 500}), model="fake-local")
        captured = {}

        def make_backend(**kwargs):
            captured.update(kwargs)
            return backend

        monkeypatch.setattr(run_eval, "make_backend", make_backend)
        args = argparse.Namespace(
            split="val", data=str(data), out=str(out), model="fake-local",
            base_url="http://127.0.0.1:8000/v1", sample_every=10, sample_fps=2.0, concurrency=1,
            send_width=768, sport=None,
            prompt='Point at the action. Reply {"x": 0, "y": 0}.',
            prompt_file=None, spring_k=0.05, no_cache=True,
        )

        result = run_eval.run(args)
        assert captured == {
            "model": "fake-local",
            "base_url": "http://127.0.0.1:8000/v1",
        }
        assert result["structural_ok"] is True
        assert result["perf"]["vlm_calls"] == 2
        assert (out / "manifest.json").exists()
        assert (out / "val.jsonl").exists()
        assert (out / "metrics-val.json").exists()
        assert "Point at the action" in backend.calls[0][0]["content"][1]["text"]

        render_debug.main([
            "--split", "val", "--pred", str(out), "--data", str(data),
            "--out", str(tmp_path / "debug"), "--out-height", "90",
        ])
        assert (tmp_path / "debug" / "a_debug.mp4").exists()
