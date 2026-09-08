"""Unit tests for v_cropper.scoreboard_render — overlay compositing.

All offline; renders onto in-memory frames.
"""
from __future__ import annotations

import numpy as np

import v_cropper.scoreboard_render as sr
from v_cropper.scoreboard_render import OverlayStyle, compose_scoreboard
from v_cropper.scoreboard_state import GameState


def _make_frame(w: int = 540, h: int = 960) -> np.ndarray:
    """Create a blank 9:16 frame filled with a solid color."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _bottom_changed(result: np.ndarray, original: np.ndarray) -> bool:
    """True if the bottom band of result differs from the original frame's bottom."""
    return not np.array_equal(result[-50:], original[-50:])


class TestComposeScoreboard:
    def test_output_shape_unchanged(self) -> None:
        frame = _make_frame()
        state = GameState(
            sport="soccer", home_team="FCB", away_team="RMA",
            home_score=2, away_score=1, period="2H", clock="67:00",
        )
        result = compose_scoreboard(frame, state)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_bottom_band_modified(self) -> None:
        frame = _make_frame()
        original_bottom = frame[-50:].copy()
        state = GameState(
            sport="hockey", home_team="BOS", away_team="MTL",
            home_score=3, away_score=1, period="2nd", clock="14:32",
        )
        result = compose_scoreboard(frame, state)
        assert not np.array_equal(result[-50:], original_bottom)

    def test_top_position(self) -> None:
        frame = _make_frame()
        original_top = frame[:50].copy()
        state = GameState(
            sport="soccer", home_team="A", away_team="B",
            home_score=1, away_score=0,
        )
        style = OverlayStyle(position="top")
        result = compose_scoreboard(frame, state, style=style)
        assert not np.array_equal(result[:50], original_top)

    def test_top_position_leaves_bottom_untouched(self) -> None:
        frame = _make_frame()
        original_bottom = frame[-50:].copy()
        state = GameState(
            sport="soccer", home_team="A", away_team="B",
            home_score=1, away_score=0,
        )
        style = OverlayStyle(position="top")
        result = compose_scoreboard(frame, state, style=style)
        assert np.array_equal(result[-50:], original_bottom)

    def test_info_board_mode(self) -> None:
        frame = _make_frame()
        original_bottom = frame[-50:].copy()
        state = GameState(
            sport="track", home_team="", away_team="",
            extra_fields=[
                {"label": "Athlete", "value": "Duplantis"},
                {"label": "Mark", "value": "6.24m"},
                {"label": "Attempt", "value": "2/3"},
            ],
        )
        result = compose_scoreboard(frame, state)
        assert result.shape == (960, 540, 3)
        assert not np.array_equal(result[-50:], original_bottom)

    def test_info_board_attempt_marks(self) -> None:
        frame = _make_frame()
        state = GameState(
            sport="high jump", event="High Jump", home_team="", away_team="",
            extra_fields=[
                {"label": "Athlete", "value": "Barshim"},
                {"label": "2.29", "value": "O"},
                {"label": "2.31", "value": "XO"},
                {"label": "2.33", "value": "XXX"},
            ],
        )
        result = compose_scoreboard(frame, state)
        assert result.shape == frame.shape

    def test_custom_height_ratio(self) -> None:
        frame = _make_frame(h=960)
        state = GameState(
            sport="soccer", home_team="A", away_team="B",
            home_score=0, away_score=0,
        )
        style = OverlayStyle(height_ratio=0.15)
        result = compose_scoreboard(frame, state, style=style)
        assert result.shape == frame.shape

    def test_custom_opacity(self) -> None:
        frame = _make_frame()
        state = GameState(
            sport="football", home_team="KC", away_team="SF",
            home_score=21, away_score=17, period="Q3", clock="4:20",
        )
        style = OverlayStyle(bg_opacity=0.5)
        result = compose_scoreboard(frame, state, style=style)
        assert result.shape == frame.shape

    def test_empty_extra_fields(self) -> None:
        frame = _make_frame()
        state = GameState(
            sport="soccer", home_team="A", away_team="B",
            home_score=1, away_score=1,
            extra_fields=[],
        )
        result = compose_scoreboard(frame, state)
        assert result.shape == frame.shape

    def test_many_extra_fields_wrap(self) -> None:
        frame = _make_frame()
        state = GameState(
            sport="basketball", home_team="LAL", away_team="BOS",
            home_score=98, away_score=102, period="Q4", clock="2:15",
            extra_fields=[
                {"label": "Fouls", "value": "4"},
                {"label": "TO", "value": "2"},
                {"label": "Poss", "value": "BOS"},
                {"label": "Shot Clock", "value": "14"},
                {"label": "Bonus", "value": "Yes"},
            ],
        )
        result = compose_scoreboard(frame, state)
        assert result.shape == frame.shape


class TestGetFontFallback:
    def test_falls_back_to_default_when_truetype_unavailable(self, monkeypatch) -> None:
        # Isolate from any real font cached by earlier tests, then fail only the
        # file-path truetype loads so the ImageFont.load_default() fallback is taken.
        # (load_default itself calls truetype with a BytesIO, which must still work.)
        monkeypatch.setattr(sr, "_FONT_CACHE", {})
        real_truetype = sr.ImageFont.truetype

        def fake_truetype(font=None, *a, **k):
            if isinstance(font, str):
                raise OSError("no font file here")
            return real_truetype(font, *a, **k)

        monkeypatch.setattr(sr.ImageFont, "truetype", fake_truetype)
        font = sr._get_font(37, bold=True)
        assert font is not None
        # Second call must hit the cache branch and return the same object.
        assert sr._get_font(37, bold=True) is font


class TestCv2Fallback:
    """When Pillow is unavailable, compose_scoreboard uses the OpenCV text fallback."""

    def test_matchup_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(sr, "_HAS_PILLOW", False)
        frame = _make_frame()
        original = frame.copy()
        state = GameState(
            sport="hockey", home_team="BOS", away_team="MTL",
            home_score=3, away_score=1, period="2nd", clock="14:32",
        )
        result = compose_scoreboard(frame, state)
        assert result.shape == original.shape
        assert _bottom_changed(result, original)

    def test_matchup_fallback_without_clock(self, monkeypatch) -> None:
        # No period/clock -> the center-text draw branch is skipped.
        monkeypatch.setattr(sr, "_HAS_PILLOW", False)
        frame = _make_frame()
        original = frame.copy()
        state = GameState(
            sport="hockey", home_team="BOS", away_team="MTL",
            home_score=3, away_score=1,
        )
        result = compose_scoreboard(frame, state)
        assert result.shape == original.shape
        assert _bottom_changed(result, original)

    def test_info_board_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(sr, "_HAS_PILLOW", False)
        frame = _make_frame()
        original = frame.copy()
        state = GameState(
            sport="track", home_team="", away_team="",
            extra_fields=[
                {"label": "Athlete", "value": "Duplantis"},
                {"label": "Mark", "value": "6.24m"},
            ],
        )
        result = compose_scoreboard(frame, state)
        assert result.shape == original.shape
        assert _bottom_changed(result, original)


class TestInfoBoardBranches:
    def test_empty_info_board_returns_early(self) -> None:
        # No teams, no event, no fields: the info-board renderer returns without
        # drawing text (but the band background is still composited).
        frame = _make_frame()
        state = GameState(sport="", event="", home_team="", away_team="", extra_fields=[])
        result = compose_scoreboard(frame, state)
        assert result.shape == frame.shape

    def test_info_board_without_athlete(self) -> None:
        frame = _make_frame()
        original = frame.copy()
        state = GameState(
            sport="racing", event="Lap 3", home_team="", away_team="",
            extra_fields=[{"label": "Pos", "value": "1"}, {"label": "Gap", "value": "+2.3s"}],
        )
        result = compose_scoreboard(frame, state)
        assert _bottom_changed(result, original)

    def test_info_board_athlete_no_detail_fields(self) -> None:
        # Athlete present but nothing else: the detail-field loop is skipped.
        frame = _make_frame()
        original = frame.copy()
        state = GameState(
            sport="golf", home_team="", away_team="",
            extra_fields=[{"label": "Athlete", "value": "Woods"}],
        )
        result = compose_scoreboard(frame, state)
        assert _bottom_changed(result, original)

    def test_info_board_no_event_no_sport(self) -> None:
        # event_label is empty -> the event-label row is skipped.
        frame = _make_frame()
        original = frame.copy()
        state = GameState(
            sport="", event="", home_team="", away_team="",
            extra_fields=[
                {"label": "Athlete", "value": "X"},
                {"label": "Mark", "value": "5m"},
            ],
        )
        result = compose_scoreboard(frame, state)
        assert _bottom_changed(result, original)

    def test_info_board_long_fields_wrap(self) -> None:
        # Narrow frame + many long fields forces the pre-measured wrap-before-draw path.
        frame = _make_frame(w=360)
        original = frame.copy()
        long_fields = [{"label": f"Metric{i}", "value": f"Value{i}"} for i in range(10)]
        state = GameState(
            sport="decathlon", event="Decathlon", home_team="", away_team="",
            extra_fields=[{"label": "Athlete", "value": "Mayer"}, *long_fields],
        )
        result = compose_scoreboard(frame, state)
        assert _bottom_changed(result, original)

    def test_info_board_attempt_marks_with_dash(self) -> None:
        # A dash in an attempt value exercises the non-O/X colour branch.
        frame = _make_frame()
        original = frame.copy()
        state = GameState(
            sport="high jump", event="High Jump", home_team="", away_team="",
            extra_fields=[
                {"label": "Athlete", "value": "Barshim"},
                {"label": "2.30", "value": "XO-"},
            ],
        )
        result = compose_scoreboard(frame, state)
        assert _bottom_changed(result, original)


class TestMatchupExtraFields:
    def test_matchup_single_line_extras(self) -> None:
        # Short extras that fit on one line take the non-wrapping matchup branch.
        frame = _make_frame()
        original = frame.copy()
        state = GameState(
            sport="basketball", home_team="LAL", away_team="BOS",
            home_score=98, away_score=102, period="Q4", clock="2:15",
            extra_fields=[{"label": "Fouls", "value": "3"}],
        )
        result = compose_scoreboard(frame, state)
        assert _bottom_changed(result, original)
