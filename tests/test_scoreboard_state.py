"""Unit tests for v_cropper.scoreboard_state — consensus / majority-vote merge.

All offline; no network or Gemini access required.
"""
from __future__ import annotations

from v_cropper.scoreboard_state import (
    GameState,
    ScoreboardReading,
    TeamInfo,
    is_matchup,
    merge_readings,
    merge_readings_with_debug,
)


def _matchup_reading(frame_idx, home="BOS", away="MTL", hs=3, as_=1,
                     period="2nd", clock="14:32", sport="hockey", extra=None):
    return ScoreboardReading(
        frame_idx=frame_idx,
        sport=sport,
        teams=[TeamInfo(name=home, score=hs), TeamInfo(name=away, score=as_)],
        period=period,
        clock=clock,
        extra_fields=extra or [],
    )


class TestIsMatchup:
    def test_true_with_teams_and_score(self):
        gs = GameState(sport="hockey", home_team="BOS", away_team="MTL",
                       home_score=3, away_score=1)
        assert is_matchup(gs) is True

    def test_false_without_teams(self):
        gs = GameState(sport="track", home_team="", away_team="",
                       extra_fields=[{"label": "Athlete", "value": "X"}])
        assert is_matchup(gs) is False

    def test_false_without_score(self):
        gs = GameState(sport="hockey", home_team="BOS", away_team="MTL")
        assert is_matchup(gs) is False


class TestMergeReadings:
    def test_none_when_no_valid(self):
        readings = [ScoreboardReading(frame_idx=i, confidence=0.0) for i in range(3)]
        assert merge_readings(readings) is None

    def test_none_when_empty(self):
        assert merge_readings([]) is None

    def test_none_when_no_teams_and_no_extras(self):
        # Valid readings but nothing to overlay.
        readings = [ScoreboardReading(frame_idx=i, sport="hockey") for i in range(3)]
        assert merge_readings(readings) is None

    def test_basic_matchup(self):
        readings = [_matchup_reading(i) for i in range(3)]
        gs = merge_readings(readings)
        assert gs is not None
        assert gs.sport == "hockey"
        assert gs.home_team == "BOS"
        assert gs.away_team == "MTL"
        assert gs.home_score == 3
        assert gs.away_score == 1
        assert is_matchup(gs)

    def test_team_name_normalized_uppercase(self):
        readings = [_matchup_reading(i, home="Bruins", away="Habs") for i in range(3)]
        gs = merge_readings(readings)
        assert gs.home_team == "BRUINS"
        assert gs.away_team == "HABS"

    def test_score_uses_mode(self):
        # Home scores 3,3,2 -> mode 3; away 1,1,1 -> 1.
        readings = [
            _matchup_reading(0, hs=3, as_=1),
            _matchup_reading(1, hs=3, as_=1),
            _matchup_reading(2, hs=2, as_=1),
        ]
        gs = merge_readings(readings)
        assert gs.home_score == 3
        assert gs.away_score == 1

    def test_sport_majority_vote(self):
        readings = [
            _matchup_reading(0, sport="hockey"),
            _matchup_reading(1, sport="hockey"),
            _matchup_reading(2, sport="soccer"),
        ]
        gs = merge_readings(readings)
        assert gs.sport == "hockey"

    def test_period_clock_from_last_frame(self):
        readings = [
            _matchup_reading(0, period="1st", clock="20:00"),
            _matchup_reading(10, period="3rd", clock="05:00"),
            _matchup_reading(5, period="2nd", clock="12:00"),
        ]
        gs = merge_readings(readings)
        # Highest frame_idx (10) wins.
        assert gs.period == "3rd"
        assert gs.clock == "05:00"

    def test_extra_fields_threshold_drops_rare(self):
        # 4 valid readings -> threshold 2. "Common" in 3 kept, "Rare" in 1 dropped.
        readings = []
        for i in range(4):
            extra = [{"label": "Common", "value": "X"}] if i < 3 else []
            if i == 0:
                extra = extra + [{"label": "Rare", "value": "Y"}]
            readings.append(_matchup_reading(i, extra=extra))
        gs = merge_readings(readings)
        labels = {ef["label"] for ef in gs.extra_fields}
        assert "Common" in labels
        assert "Rare" not in labels

    def test_single_reading_accepts_all_extras(self):
        reading = ScoreboardReading(
            frame_idx=0, sport="track", event="High Jump",
            extra_fields=[
                {"label": "Athlete", "value": "Duplantis"},
                {"label": "Mark", "value": "6.24m"},
            ],
        )
        gs = merge_readings([reading])
        assert gs is not None
        labels = {ef["label"] for ef in gs.extra_fields}
        assert labels == {"Athlete", "Mark"}
        assert not is_matchup(gs)  # info-board mode

    def test_blank_label_or_value_extras_skipped(self):
        # Extra fields with an empty label or value must be ignored, not counted.
        reading = _matchup_reading(0, extra=[
            {"label": "", "value": "5"},        # blank label -> skipped
            {"label": "Fouls", "value": ""},    # blank value -> skipped
            {"label": "Shots", "value": "30"},  # kept
        ])
        gs = merge_readings([reading])
        labels = {ef["label"] for ef in gs.extra_fields}
        assert labels == {"Shots"}


class TestMergeReadingsWithDebug:
    def test_debug_tracks_included_excluded(self):
        readings = []
        for i in range(4):
            extra = [{"label": "Common", "value": "X"}] if i < 3 else []
            if i == 0:
                extra = extra + [{"label": "Rare", "value": "Y"}]
            readings.append(_matchup_reading(i, extra=extra))
        gs, debug = merge_readings_with_debug(readings)
        assert gs is not None
        assert debug.total_readings == 4
        assert debug.valid_readings == 4
        assert debug.threshold == 2
        assert "Common" in debug.included_fields
        assert "Rare" in debug.excluded_fields
        assert debug.game_state_json  # populated

    def test_debug_none_when_no_valid(self):
        readings = [ScoreboardReading(frame_idx=i, confidence=0.0) for i in range(2)]
        gs, debug = merge_readings_with_debug(readings)
        assert gs is None
        assert debug.valid_readings == 0

    def test_debug_to_lines_smoke(self):
        readings = [_matchup_reading(i, extra=[{"label": "Fouls", "value": "4"}]) for i in range(3)]
        _gs, debug = merge_readings_with_debug(readings)
        lines = debug.to_lines()
        assert any("SCOREBOARD OCR" in ln for ln in lines)

    def test_debug_single_reading_threshold_one(self):
        # A single valid reading keeps threshold at 1 (the len(valid) > 1 branch is skipped)
        # and still includes its extras + skips blank-label fields.
        reading = _matchup_reading(0, extra=[
            {"label": "", "value": "x"},        # blank -> skipped in the debug loop too
            {"label": "Shots", "value": "22"},
        ])
        gs, debug = merge_readings_with_debug([reading])
        assert gs is not None
        assert debug.valid_readings == 1
        assert debug.threshold == 1
        assert "Shots" in debug.included_fields
