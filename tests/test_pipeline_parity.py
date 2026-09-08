"""Parity between the benchmarked path and the deployed path.

``eval/run_eval.py`` scores a crop path built by its own orchestration, while the deployed
service runs ``v_cropper.pipeline.run_pipeline`` and the CLI runs a third. When those drift
the benchmark stops describing the thing that ships — silently, because every path still
produces a plausible crop. These tests pin the decisions that determine the crop path.

The three entry points are not merged into one: ``run_pipeline`` has no debug-overlay
support, which the CLI and ``eval/render_debug.py`` both require, so collapsing them would
delete a shipped feature. Instead the sampling decision has a single implementation
(``pipeline.resolve_stride``) and these tests assert the callers agree.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

from conftest import FakeBackend, write_clip

_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
sys.path.insert(0, _EVAL_DIR)

import run_eval  # noqa: E402

from v_cropper import cli, pipeline  # noqa: E402
from v_cropper.pipeline import FALLBACK_STRIDE, CropOptions, resolve_stride  # noqa: E402

POINT = json.dumps({"x": 500, "y": 500})


class TestStrideParity:
    """All three entry points must resolve the same keyframe stride."""

    @pytest.mark.parametrize("sample_every,sample_fps,source_fps,expected", [
        (None, 2.0, 30.0, 15),                  # the common case: 30 fps source at 2 fps
        (None, 2.0, 60.0, 30),
        (None, 2.0, 25.0, 12),                  # round() is banker's rounding: 12.5 -> 12
        (7, 2.0, 30.0, 7),                      # explicit stride wins
        (0, 2.0, 30.0, 1),                      # clamped to a usable stride
        (-5, 2.0, 30.0, 1),
        (None, 2.0, 0.0, FALLBACK_STRIDE),      # unknown fps
        (None, 0.0, 30.0, FALLBACK_STRIDE),     # unusable sample_fps
        (None, 2.0, 0.5, 1),                    # very low fps still samples every frame
    ])
    def test_cli_pipeline_and_eval_agree(self, sample_every, sample_fps, source_fps, expected):
        options = CropOptions(sample_fps=sample_fps, sample_every=sample_every)
        assert resolve_stride(sample_every, sample_fps, source_fps) == expected
        assert cli._sample_every(sample_every, sample_fps, source_fps) == expected
        assert pipeline._stride(options, source_fps) == expected

    def test_eval_uses_the_shared_resolver(self):
        assert run_eval.resolve_stride is resolve_stride

    def test_eval_sampling_defaults_defer_to_source_fps(self, monkeypatch):
        """A default eval run must sample at the rate the service actually uses.

        This previously defaulted to a hardcoded ``--sample-every 30``, so a 30 fps clip was
        benchmarked at half the keyframe density the service runs it at.
        """
        captured = {}

        def fake_run(args):
            captured.update(vars(args))
            return {}

        monkeypatch.setattr(run_eval, "run", fake_run)
        monkeypatch.setattr(sys, "argv", ["run_eval.py"])
        run_eval.main()
        assert captured["sample_every"] is None, "a hardcoded stride cannot track source fps"
        assert captured["sample_fps"] == CropOptions.sample_fps


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
class TestOrchestrationParity:
    """The CLI and the service pipeline must sample identical frames for one clip.

    Runs both entry points for real, so it needs ffmpeg like the other integration tests.
    """

    SAMPLING_KEYS = ("sample_every", "send_width", "concurrency", "prompt")

    @pytest.fixture
    def capture_focus(self, monkeypatch):
        """Record the sampling arguments each entry point passes to the focus pass."""
        calls = []
        real = pipeline.extract_focus_points

        def spy(video, **kwargs):
            calls.append({key: kwargs.get(key) for key in self.SAMPLING_KEYS})
            return real(video, **kwargs)

        monkeypatch.setattr(pipeline, "extract_focus_points", spy)
        monkeypatch.setattr(cli, "extract_focus_points", spy)
        return calls

    def test_cli_and_pipeline_sample_identically(self, tmp_path, monkeypatch, capture_focus):
        clip = write_clip(tmp_path / "clip.mp4", n_frames=30, w=192, h=108)
        monkeypatch.setattr(cli, "make_backend", lambda **kw: FakeBackend(POINT))
        monkeypatch.setattr(pipeline, "make_backend", lambda **kw: FakeBackend(POINT))

        cli.main([str(clip), "-o", str(tmp_path / "cli_out.mp4"), "--api-key", "k"])
        pipeline.run_pipeline(
            clip,
            tmp_path / "svc_out.mp4",
            options=CropOptions(),
            backend=FakeBackend(POINT),
            source_is_trimmed=True,
        )

        assert len(capture_focus) == 2
        cli_call, pipeline_call = capture_focus
        assert cli_call == pipeline_call

    def test_both_produce_the_same_crop_path(self, tmp_path, monkeypatch):
        """Same clip and same focus points must yield the same windows end to end."""
        clip = write_clip(tmp_path / "clip.mp4", n_frames=30, w=192, h=108)
        paths = []
        real_render = pipeline.render

        def spy(video, focus_map, out, **kwargs):
            paths.append((sorted(focus_map.items()), kwargs.get("spring_k")))
            return real_render(video, focus_map, out, **kwargs)

        monkeypatch.setattr(pipeline, "render", spy)
        monkeypatch.setattr(cli, "render", spy)
        monkeypatch.setattr(cli, "make_backend", lambda **kw: FakeBackend(POINT))

        cli.main([str(clip), "-o", str(tmp_path / "cli_out.mp4"), "--api-key", "k"])
        pipeline.run_pipeline(
            clip,
            tmp_path / "svc_out.mp4",
            options=CropOptions(),
            backend=FakeBackend(POINT),
            source_is_trimmed=True,
        )

        assert len(paths) == 2
        assert paths[0] == paths[1]
