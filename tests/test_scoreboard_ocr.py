"""Unit tests for v_cropper.scoreboard_ocr — sampling + parsing + extraction.

All offline: a tiny synthetic clip is written to a temp dir, and a FakeBackend
is injected so no API key or network is required.
"""
from __future__ import annotations

import json

from conftest import FakeBackend, write_clip
from v_cropper.scoreboard_ocr import (
    _parse_reading,
    _select_sample_frames,
    extract_scoreboard,
    sample_frames,
)


class TestSelectSampleFrames:
    def test_even_spacing(self):
        idx = _select_sample_frames(100, 10)
        assert idx == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    def test_short_clip_returns_all(self):
        idx = _select_sample_frames(5, 10)
        assert idx == [0, 1, 2, 3, 4]

    def test_exact_count(self):
        idx = _select_sample_frames(10, 10)
        assert idx == list(range(10))


class TestSampleFrames:
    def test_count_and_shape(self, tmp_path):
        clip = write_clip(tmp_path / "clip.mp4", n_frames=30)
        frames = sample_frames(str(clip), sample_count=10)
        assert len(frames) == 10
        for f in frames:
            assert f.shape == (90, 160, 3)

    def test_short_clip(self, tmp_path):
        clip = write_clip(tmp_path / "short.mp4", n_frames=4)
        frames = sample_frames(str(clip), sample_count=10)
        assert len(frames) == 4

    def test_missing_file_returns_empty(self):
        assert sample_frames("/nonexistent/path/to/video.mp4") == []

    def test_returns_empty_when_capture_unopenable(self, monkeypatch):
        # Frame count reports frames but the capture won't open (decode inconsistency).
        monkeypatch.setattr("v_cropper.scoreboard_ocr._count_frames", lambda p: 10)
        assert sample_frames("/nonexistent/xyz.mp4", sample_count=4) == []

    def test_stops_cleanly_on_short_read(self, tmp_path, monkeypatch):
        # Count over-reports (20) but only 5 frames decode; the loop must stop at EOF.
        clip = write_clip(tmp_path / "short.mp4", n_frames=5)
        monkeypatch.setattr("v_cropper.scoreboard_ocr._count_frames", lambda p: 20)
        frames = sample_frames(str(clip), sample_count=8)
        assert len(frames) <= 5


class TestParseReading:
    def test_valid_matchup_json(self):
        text = json.dumps({
            "sport": "hockey",
            "teams": [{"name": "BOS", "score": 3}, {"name": "MTL", "score": 1}],
            "period": "2nd", "clock": "14:32",
        })
        r = _parse_reading(0, text)
        assert r.confidence == 1.0
        assert r.sport == "hockey"
        assert len(r.teams) == 2
        assert r.teams[0].name == "BOS"
        assert r.teams[0].score == 3
        assert r.period == "2nd"

    def test_fenced_json(self):
        text = "```json\n" + json.dumps({
            "sport": "soccer",
            "teams": [{"name": "A", "score": 1}, {"name": "B", "score": 0}],
        }) + "\n```"
        r = _parse_reading(1, text)
        assert r.confidence == 1.0
        assert r.sport == "soccer"
        assert len(r.teams) == 2

    def test_malformed_json_zero_confidence(self):
        r = _parse_reading(2, "not json at all {")
        assert r.confidence == 0.0
        assert r.sport is None

    def test_string_score_coerced(self):
        text = json.dumps({"sport": "x", "teams": [{"name": "A", "score": "5"}]})
        r = _parse_reading(0, text)
        assert r.teams[0].score == 5

    def test_athlete_dedup(self):
        text = json.dumps({
            "sport": "high jump",
            "event": "High Jump",
            "athlete": "Duplantis",
            "teams": [],
            "extra_fields": [
                {"label": "Athlete", "value": "Duplantis"},  # duplicate -> dropped
                {"label": "Mark", "value": "6.24m"},
            ],
        })
        r = _parse_reading(0, text)
        athlete_fields = [ef for ef in r.extra_fields if ef["label"] == "Athlete"]
        assert len(athlete_fields) == 1
        assert athlete_fields[0]["value"] == "Duplantis"
        assert r.event == "High Jump"
        labels = [ef["label"] for ef in r.extra_fields]
        assert "Mark" in labels

    def test_null_fields_dropped(self):
        text = json.dumps({
            "sport": "soccer",
            "teams": [{"name": "A", "score": 1}, {"name": "B", "score": 2}],
            "period": "null",
            "extra_fields": [{"label": "Foo", "value": "null"}],
        })
        r = _parse_reading(0, text)
        assert r.period is None
        assert r.extra_fields == []

    def test_non_numeric_string_score_becomes_none(self):
        # A score string that isn't an int must coerce to None, not raise.
        text = json.dumps({"sport": "x", "teams": [{"name": "A", "score": "TBD"}]})
        r = _parse_reading(0, text)
        assert r.teams[0].name == "A"
        assert r.teams[0].score is None

    def test_athlete_literal_null_string_ignored(self):
        # A literal "null" string athlete must be treated as absent.
        text = json.dumps({
            "sport": "golf", "athlete": "null",
            "extra_fields": [{"label": "Hole", "value": "7"}],
        })
        r = _parse_reading(0, text)
        # No "athlete"-labeled field should be injected from a null athlete.
        assert all(ef["label"].lower() != "athlete" for ef in r.extra_fields)


def _matchup_payload():
    return json.dumps({
        "sport": "hockey",
        "teams": [{"name": "BOS", "score": 3}, {"name": "MTL", "score": 1}],
        "period": "2nd", "clock": "10:00",
    })


class TestExtractScoreboard:
    def test_extract_with_fake_backend(self, tmp_path):
        clip = write_clip(tmp_path / "clip.mp4", n_frames=30)
        frames = sample_frames(str(clip), sample_count=4)
        assert len(frames) == 4

        be = FakeBackend(_matchup_payload())
        readings = extract_scoreboard(frames, backend=be)
        assert len(readings) == 4
        assert all(r.confidence == 1.0 for r in readings)
        assert all(r.sport == "hockey" for r in readings)
        # One message per frame, each carrying exactly one image + the prompt.
        assert len(be.calls) == 4
        for msgs in be.calls:
            parts = msgs[0]["content"]
            assert sum(1 for p in parts if p["type"] == "image_url") == 1
            assert any(p["type"] == "text" for p in parts)

    def test_token_totals_aggregated(self, tmp_path):
        clip = write_clip(tmp_path / "clip.mp4", n_frames=30)
        frames = sample_frames(str(clip), sample_count=4)
        be = FakeBackend(_matchup_payload(),
                         usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        extract_scoreboard(frames, backend=be)
        # 4 frames -> 4 calls -> totals are per-call * 4, tracked thread-safely.
        assert be.usage_totals["api_calls"] == 4
        assert be.usage_totals["total_tokens"] == 60

    def test_extract_empty_frames_never_calls_backend(self):
        be = FakeBackend(_matchup_payload())
        assert extract_scoreboard([], backend=be) == []
        assert be.calls == []

    def test_extract_handles_backend_failure(self, tmp_path):
        clip = write_clip(tmp_path / "clip.mp4", n_frames=10)
        frames = sample_frames(str(clip), sample_count=3)
        be = FakeBackend(_matchup_payload(), raise_exc=RuntimeError("boom"))
        readings = extract_scoreboard(frames, backend=be)
        assert len(readings) == 3
        assert all(r.confidence == 0.0 for r in readings)

    def test_malformed_reply_zero_confidence(self, tmp_path):
        clip = write_clip(tmp_path / "clip.mp4", n_frames=10)
        frames = sample_frames(str(clip), sample_count=3)
        be = FakeBackend("not json at all {")
        readings = extract_scoreboard(frames, backend=be)
        assert all(r.confidence == 0.0 for r in readings)

    def test_on_progress_callback_invoked_per_frame(self, tmp_path):
        clip = write_clip(tmp_path / "clip.mp4", n_frames=20)
        frames = sample_frames(str(clip), sample_count=4)
        be = FakeBackend(_matchup_payload())
        seen: list[tuple[int, int]] = []
        extract_scoreboard(frames, backend=be, on_progress=lambda done, total: seen.append((done, total)))
        assert len(seen) == 4  # one callback per completed frame
        assert {total for _done, total in seen} == {4}  # total is always the frame count
        assert sorted(done for done, _total in seen) == [1, 2, 3, 4]  # monotonic completion


class TestScoreboardConsensusSeam:
    """Integration: OCR readings -> consensus GameState (offline, fake backend)."""

    def test_readings_merge_to_game_state(self, tmp_path):
        from v_cropper.scoreboard_state import merge_readings_with_debug

        clip = write_clip(tmp_path / "clip.mp4", n_frames=30)
        frames = sample_frames(str(clip), sample_count=5)
        be = FakeBackend(_matchup_payload())
        readings = extract_scoreboard(frames, backend=be)

        game_state, debug = merge_readings_with_debug(readings)
        assert game_state is not None
        assert game_state.sport == "hockey"
        # Home/away resolved from the 2-team matchup.
        assert {game_state.home_team, game_state.away_team} == {"BOS", "MTL"}

    def test_all_failed_readings_yield_no_game_state(self, tmp_path):
        from v_cropper.scoreboard_state import merge_readings_with_debug

        clip = write_clip(tmp_path / "clip.mp4", n_frames=30)
        frames = sample_frames(str(clip), sample_count=4)
        be = FakeBackend(_matchup_payload(), raise_exc=RuntimeError("boom"))
        readings = extract_scoreboard(frames, backend=be)
        game_state, _debug = merge_readings_with_debug(readings)
        assert game_state is None
