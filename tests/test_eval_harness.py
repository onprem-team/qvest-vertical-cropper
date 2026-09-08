"""Offline unit tests for the benchmark harness pieces added for public release:

- eval/labeler/ingest_labels.py — labels.json -> per-split ground_truth.jsonl
- eval/labeler/extract_frames.py — ~1/sec frame sampling (smoke)
- eval/labeler/app.py — Flask click-labeler (routes, validation, resume)
- eval/render_debug.py — GT-vs-prediction window / HIT-MISS math + a render smoke
- eval/run_eval.py:perf_summary — perf/cost roll-up
- CLI arg wiring (main()) for ingest_labels / render_debug

Everything here is fully offline (synthetic clips, no VLM, no network).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from conftest import write_clip

_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
sys.path.insert(0, _EVAL_DIR)
sys.path.insert(0, os.path.join(_EVAL_DIR, "labeler"))

import extract_frames  # noqa: E402
import ingest_labels  # noqa: E402
import render_debug  # noqa: E402
import run_eval  # noqa: E402

from v_cropper.render import compute_crop_path  # noqa: E402


class TestIngestLabels:
    def test_routes_rounds_and_drops(self):
        labels = {
            "1__15": {"x": 100.46, "y": 50.64, "skipped": False},
            "1__45": {"x": None, "y": None, "skipped": True},   # skipped -> dropped
            "2__10": {"x": 200.0, "y": 80.0, "skipped": False},
        }
        splits = {"val": ["1"], "test": ["2"]}
        out = ingest_labels.to_ground_truth(labels, splits)
        assert out["val"] == [{"item_id": "1", "frame_idx": 15, "x": 100.5, "y": 50.6}]
        assert out["test"] == [{"item_id": "2", "frame_idx": 10, "x": 200.0, "y": 80.0}]

    def test_non_skipped_but_missing_coords_dropped(self):
        labels = {"1__3": {"x": None, "y": 5, "skipped": False}}
        out = ingest_labels.to_ground_truth(labels, {"val": ["1"]})
        assert out["val"] == []

    def test_sorted_by_item_then_frame(self):
        labels = {
            "1__30": {"x": 1, "y": 1, "skipped": False},
            "1__5": {"x": 2, "y": 2, "skipped": False},
        }
        out = ingest_labels.to_ground_truth(labels, {"val": ["1"]})
        assert [r["frame_idx"] for r in out["val"]] == [5, 30]

    def test_item_not_in_any_split_raises(self):
        with pytest.raises(ValueError, match="not in any split"):
            ingest_labels.to_ground_truth({"9__1": {"x": 1, "y": 1}}, {"val": ["1"]})

    def test_malformed_key_raises(self):
        with pytest.raises(ValueError, match="malformed label key"):
            ingest_labels.to_ground_truth({"nounderscore": {"x": 1, "y": 1}}, {"val": []})

    def test_write_ground_truth(self, tmp_path):
        by_split = {"val": [{"item_id": "1", "frame_idx": 15, "x": 100.5, "y": 50.6}]}
        ingest_labels.write_ground_truth(by_split, str(tmp_path))
        out_file = tmp_path / "val" / "ground_truth.jsonl"
        assert out_file.exists()
        rows = [json.loads(line) for line in out_file.read_text().splitlines()]
        assert rows == by_split["val"]


class TestExtractFrames:
    def test_sample_targets(self):
        # 30 fps, 30 frames -> only the 0.5s frame (idx 15) fits.
        assert extract_frames.sample_targets(30, 30) == [15]
        # 30 fps, 90 frames -> 0.5s, 1.5s, 2.5s.
        assert extract_frames.sample_targets(30, 90) == [15, 45, 75]

    def test_extract_smoke(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        write_clip(raw / "a.mp4", n_frames=30, w=160, h=90)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(
            {"a": {"n_frames": 30, "fps": 30.0, "width": 160, "height": 90}}))
        out_dir = tmp_path / "frames"
        index_path = tmp_path / "frames_index.json"
        index = extract_frames.extract(str(manifest), str(raw), str(out_dir), str(index_path))
        assert index == [{"item_id": "a", "frame_idx": 15, "file": "a__15.jpg",
                          "width": 160, "height": 90}]
        assert (out_dir / "a__15.jpg").exists()
        assert json.loads(index_path.read_text()) == index


class TestRenderDebugMath:
    def test_compute_hits_hit_miss_and_missing(self):
        win = {"frame_idx": 5, "x_center": 100.0, "y_center": 50.0, "width": 40, "height": 71}
        win_by_frame = {5: win}
        # point inside the window -> HIT
        assert render_debug.compute_hits(
            [{"frame_idx": 5, "x": 105, "y": 50}], win_by_frame, 640, 360) == {5: True}
        # point outside the window -> MISS
        assert render_debug.compute_hits(
            [{"frame_idx": 5, "x": 300, "y": 50}], win_by_frame, 640, 360) == {5: False}
        # no predicted window for the labeled frame -> omitted (scorer counts it a miss)
        assert render_debug.compute_hits(
            [{"frame_idx": 9, "x": 100, "y": 50}], win_by_frame, 640, 360) == {}

    def test_violating_window_scores_miss(self):
        # width/height break the 9:16 aspect -> window_violations > 0 -> MISS even if inside.
        bad = {"frame_idx": 0, "x_center": 100.0, "y_center": 50.0, "width": 40, "height": 40}
        assert render_debug.compute_hits(
            [{"frame_idx": 0, "x": 100, "y": 50}], {0: bad}, 640, 360) == {0: False}

    def test_gt_center_x_interpolates_and_edge_holds(self):
        labels = [{"frame_idx": 0, "x": 100}, {"frame_idx": 10, "x": 200}]
        assert render_debug.gt_center_x(labels, 5) == 150.0   # midpoint
        assert render_debug.gt_center_x(labels, -3) == 100.0  # edge-held left
        assert render_debug.gt_center_x(labels, 99) == 200.0  # edge-held right
        assert render_debug.gt_center_x([], 5) is None

    def test_render_item_smoke(self, tmp_path):
        clip = write_clip(tmp_path / "a.mp4", n_frames=20, w=160, h=90)
        windows, fw, fh, fps = compute_crop_path(clip, {0: 80.0, 19: 120.0})
        output = {"crop_path": windows, "frame_width": fw, "frame_height": fh, "fps": fps}
        labels = [{"item_id": "a", "frame_idx": 5, "x": 80, "y": 45}]
        out_path = tmp_path / "a_debug.mp4"
        n, n_hit, n_lab = render_debug._render_item(
            "a", output, labels, str(clip), str(out_path), out_height=90)
        assert n == 20 and n_lab == 1 and n_hit in (0, 1)
        assert out_path.exists() and out_path.stat().st_size > 0


class TestPerfSummary:
    def test_perf_summary_rolls_usage_and_cost(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PRICE_IN", "1.0")
        monkeypatch.setenv("VCROPPER_PRICE_OUT", "2.0")
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000,
                 "total_tokens": 1_500_000, "api_calls": 7}
        perf = run_eval.perf_summary(usage, wall_time_sec=12.345, rtf=0.5, cache_hits=3)
        assert perf["vlm_calls"] == 7
        assert perf["total_tokens"] == 1_500_000
        assert perf["est_cost_usd"] == pytest.approx(2.0)  # 1M*$1 + 0.5M*$2
        assert perf["wall_time_sec"] == 12.35
        assert perf["rtf"] == 0.5
        assert perf["cache_hits"] == 3


def _seed_frames(tmp_path, monkeypatch):
    """Extract labeling frames from a synthetic clip and point the app at them."""
    import app as labeler_app  # local import so the suite skips cleanly if Flask is absent

    raw = tmp_path / "raw"
    raw.mkdir()
    write_clip(raw / "a.mp4", n_frames=90, w=160, h=90)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"a": {"n_frames": 90, "fps": 30.0, "width": 160, "height": 90}}))
    frames_dir = tmp_path / "frames"
    index_path = tmp_path / "frames_index.json"
    extract_frames.extract(str(manifest), str(raw), str(frames_dir), str(index_path))
    labels_path = tmp_path / "labels.json"
    monkeypatch.setattr(labeler_app, "INDEX_PATH", str(index_path))
    monkeypatch.setattr(labeler_app, "LABELS_PATH", str(labels_path))
    monkeypatch.setattr(labeler_app, "FRAMES_DIR", str(frames_dir))
    return labeler_app, labels_path


class TestLabelerApp:
    def test_routes_validation_and_resume(self, tmp_path, monkeypatch):
        pytest.importorskip("flask")
        app_mod, labels_path = _seed_frames(tmp_path, monkeypatch)
        client = app_mod.app.test_client()

        assert client.get("/").status_code == 200
        state = client.get("/api/state").get_json()
        assert len(state["frames"]) == 3 and state["labels"] == {}

        ok = client.post("/api/label", json={"item_id": "a", "frame_idx": 15,
                                             "x": 80.0, "y": 45.0, "skipped": False})
        assert ok.status_code == 200 and ok.get_json()["ok"]
        # out-of-bounds + unknown-frame clicks are rejected
        assert client.post("/api/label", json={"item_id": "a", "frame_idx": 45,
                                               "x": 9999, "y": 45}).status_code == 400
        assert client.post("/api/label", json={"item_id": "a", "frame_idx": 999,
                                               "x": 1, "y": 1}).status_code == 400
        # skip persists without coordinates
        assert client.post("/api/label", json={"item_id": "a", "frame_idx": 45,
                                               "skipped": True}).status_code == 200
        # frame is served from the configured FRAMES_DIR
        assert client.get("/frames/a__15.jpg").status_code == 200

        saved = json.loads(labels_path.read_text())
        assert saved["a__15"] == {"x": 80.0, "y": 45.0, "skipped": False}
        assert saved["a__45"]["skipped"] is True
        # resume: a fresh state call reflects the persisted labels
        assert len(client.get("/api/state").get_json()["labels"]) == 2


class TestCLIWiring:
    def test_ingest_labels_main(self, tmp_path):
        labels = tmp_path / "labels.json"
        labels.write_text(json.dumps({"1__15": {"x": 10.0, "y": 20.0, "skipped": False}}))
        splits = tmp_path / "splits.json"
        splits.write_text(json.dumps({"val": ["1"]}))
        ingest_labels.main(["--data", str(tmp_path), "--labels", str(labels),
                            "--splits", str(splits)])
        rows = [json.loads(x) for x in (tmp_path / "val" / "ground_truth.jsonl")
                .read_text().splitlines()]
        assert rows == [{"item_id": "1", "frame_idx": 15, "x": 10.0, "y": 20.0}]

    def test_render_debug_main(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        clip = write_clip(raw / "a.mp4", n_frames=20, w=160, h=90)
        (tmp_path / "val").mkdir()
        (tmp_path / "val" / "ground_truth.jsonl").write_text(
            json.dumps({"item_id": "a", "frame_idx": 5, "x": 80, "y": 45}) + "\n")
        pred = tmp_path / "pred"
        pred.mkdir()
        windows, fw, fh, fps = compute_crop_path(clip, {0: 80.0, 19: 120.0})
        pred_rec = {"item_id": "a", "output": {"crop_path": windows, "frame_width": fw,
                                               "frame_height": fh, "fps": fps}}
        (pred / "val.jsonl").write_text(json.dumps(pred_rec) + "\n")
        render_debug.main(["--split", "val", "--pred", str(pred), "--data", str(tmp_path),
                           "--out", str(tmp_path / "debug"), "--out-height", "90"])
        assert (tmp_path / "debug" / "a_debug.mp4").exists()
