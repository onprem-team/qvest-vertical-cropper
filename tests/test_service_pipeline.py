from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from conftest import FakeBackend
from v_cropper.pipeline import CropOptions, PipelineCancelled, run_pipeline

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
def test_pipeline_trims_crops_h264_and_preserves_audio(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x192:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source),
        ],
        check=True,
    )
    output = tmp_path / "vertical.mp4"
    phases = []
    result = run_pipeline(
        source,
        output,
        in_s=0.5,
        out_s=2.5,
        work_dir=tmp_path / "work",
        options=CropOptions(sample_every=15, concurrency=1),
        backend=FakeBackend(json.dumps({"x": 500, "y": 500})),
        progress=lambda phase, pct, detail: phases.append((phase, pct)),
    )
    assert output.exists()
    assert result.media["video_codec"] == "h264"
    assert result.media["has_audio"]
    assert result.media["height"] / result.media["width"] == pytest.approx(16 / 9, rel=0.02)
    assert result.media["duration_sec"] == pytest.approx(2, abs=0.15)
    assert result.metrics["keyframes_ok"] == 4
    assert phases[0][0] == "trimming" and phases[-1] == ("complete", 100)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
def test_pipeline_fails_when_every_keyframe_fails(tmp_path):
    """A total VLM failure must not be reported as a successful crop.

    render() falls back to a static centre crop when the focus map is empty, so the job
    produced a plausible 9:16 file and reported "succeeded" while the model had contributed
    nothing. Caught by the deployment smoke when the provider endpoint was unreachable.
    """
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x192:rate=30:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
        ],
        check=True,
    )
    with pytest.raises(RuntimeError, match="keyframes failed"):
        run_pipeline(
            source,
            tmp_path / "vertical.mp4",
            work_dir=tmp_path / "work",
            options=CropOptions(sample_every=15, concurrency=1),
            backend=FakeBackend(raise_exc=RuntimeError("provider unreachable")),
            source_is_trimmed=True,
        )


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
def test_pipeline_tolerates_partial_keyframe_failure(tmp_path):
    """One bad keyframe among several is recoverable and must still succeed."""
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x192:rate=30:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
        ],
        check=True,
    )
    backend = FakeBackend(["not-json", json.dumps({"x": 500, "y": 500})])
    result = run_pipeline(
        source,
        tmp_path / "vertical.mp4",
        work_dir=tmp_path / "work",
        options=CropOptions(sample_every=15, concurrency=1),
        backend=backend,
        source_is_trimmed=True,
    )
    assert result.metrics["keyframes_ok"] > 0
    assert result.metrics["keyframes_failed"] >= 1


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
def test_scoreboard_run_uses_both_backends_and_bills_for_both(tmp_path):
    """The scoreboard branch fans out to a second model on a separate thread.

    It is the only orchestration path in run_pipeline that runs two backends concurrently,
    and the cost it reports has to cover both or a scoreboard run looks free.
    """
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x192:rate=30:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
        ],
        check=True,
    )
    focus = FakeBackend(json.dumps({"x": 500, "y": 500}))
    scoreboard = FakeBackend(json.dumps({
        "sport": "hockey",
        "teams": [{"name": "BOS", "score": 3}, {"name": "MTL", "score": 1}],
        "period": "2nd", "clock": "14:32",
    }))
    output = tmp_path / "vertical.mp4"

    result = run_pipeline(
        source,
        output,
        work_dir=tmp_path / "work",
        options=CropOptions(sample_every=15, concurrency=1, scoreboard=True, scoreboard_sample_count=2),
        backend=focus,
        scoreboard_backend=scoreboard,
        source_is_trimmed=True,
    )

    assert output.exists()
    assert scoreboard.usage_totals["api_calls"] == 2
    assert focus.usage_totals["api_calls"] > 0
    assert result.metrics["scoreboard_usage"]["api_calls"] == 2

    focus_only = run_pipeline(
        source,
        tmp_path / "no-overlay.mp4",
        work_dir=tmp_path / "work-plain",
        options=CropOptions(sample_every=15, concurrency=1),
        backend=FakeBackend(json.dumps({"x": 500, "y": 500})),
        source_is_trimmed=True,
    )
    assert focus_only.metrics["scoreboard_usage"] is None
    assert result.metrics["est_cost_usd"] > focus_only.metrics["est_cost_usd"]


class TestUnreadableSource:
    """A corrupt or truncated upload must fail the job, not centre-crop silence."""

    @pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
    @pytest.mark.parametrize("probe_result", [(0.0, 0), (30.0, 0), (0.0, 100)])
    def test_a_container_ffprobe_accepts_but_opencv_cannot_decode(self, tmp_path, monkeypatch, probe_result):
        """ffprobe reading the header does not mean OpenCV can decode the stream.

        Patched rather than faked with a real file: a container that satisfies ffprobe and
        defeats OpenCV is codec-specific, so it would not reproduce across machines.
        """
        source = tmp_path / "source.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x192:rate=30:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ],
            check=True,
        )
        monkeypatch.setattr("v_cropper.pipeline._probe_cv", lambda path: probe_result)

        with pytest.raises(RuntimeError, match="source video is unreadable"):
            run_pipeline(
                source,
                tmp_path / "vertical.mp4",
                work_dir=tmp_path / "work",
                options=CropOptions(sample_every=15, concurrency=1),
                backend=FakeBackend("{}"),
                source_is_trimmed=True,
            )

    @pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
    def test_a_clip_that_yields_no_keyframes_at_all_is_rejected(self, tmp_path, monkeypatch):
        """Distinct from a total VLM failure: here sampling itself produced nothing."""
        source = tmp_path / "source.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x192:rate=30:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ],
            check=True,
        )
        monkeypatch.setattr(
            "v_cropper.pipeline.extract_focus_points", lambda *a, **k: ({}, [], 0))

        with pytest.raises(RuntimeError, match="source video is unreadable"):
            run_pipeline(
                source,
                tmp_path / "vertical.mp4",
                work_dir=tmp_path / "work",
                options=CropOptions(sample_every=15, concurrency=1),
                backend=FakeBackend("{}"),
                source_is_trimmed=True,
            )


def test_pipeline_honors_preflight_cancellation(tmp_path):
    with pytest.raises(PipelineCancelled):
        run_pipeline(
            "unused.mp4",
            tmp_path / "unused.mp4",
            cancelled=lambda: True,
            backend=FakeBackend("{}"),
        )
