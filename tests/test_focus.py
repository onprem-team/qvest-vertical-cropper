"""Unit + adversarial tests for point-based focus extraction (all offline)."""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from conftest import FakeBackend, write_clip
from v_cropper import focus
from v_cropper.focus import (
    extract_focus_points,
    interpolate_focus,
    median3,
    parse_point_response,
)
from v_cropper.prompts import PRESETS


def _decode_data_uri(part):
    """Turn an image_url content part back into a BGR frame (for downscale checks)."""
    payload = part["image_url"]["url"].split(",", 1)[1]
    raw = base64.b64decode(payload)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- #
# parse_point_response
# --------------------------------------------------------------------------- #
class TestParsePoint:
    def test_plain_json(self):
        assert parse_point_response('{"x": 300, "y": 700}') == (300.0, 700.0)

    def test_fenced_json(self):
        assert parse_point_response('```json\n{"x": 10, "y": 20}\n```') == (10.0, 20.0)

    def test_extra_prose(self):
        assert parse_point_response('Sure! {"x": 5, "y": 6} hope that helps') == (5.0, 6.0)

    def test_floats(self):
        assert parse_point_response('{"x": 12.5, "y": 33.0}') == (12.5, 33.0)

    def test_bounds_zero_and_thousand_ok(self):
        assert parse_point_response('{"x": 0, "y": 1000}') == (0.0, 1000.0)

    def test_list_packed_x_takes_first_element(self):
        # qwen3-vl packs the point into the x array; y may echo the example.
        assert parse_point_response('{"x": [570, 324], "y": [500, 500]}') == (570.0, 500.0)

    def test_list_x_scalar_y(self):
        assert parse_point_response('{"x": [575, 310], "y": 310}') == (575.0, 310.0)

    def test_malformed_json_missing_quote_recovers(self):
        # qwen3-vl occasionally drops a key quote; regex fallback recovers the point.
        assert parse_point_response('{"x": 624, y": 437}') == (624.0, 437.0)

    @pytest.mark.parametrize("text", [
        "", None, "no json here", "{}", '{"x": 1}', '{"y": 2}',
        '{"x": "left", "y": 2}', '{"x": 1200, "y": 5}', '{"x": -1, "y": 5}',
        '{"x": 5, "y": 1001}', "not json {broken",
        '{"x": [], "y": 5}',
    ])
    def test_invalid_returns_none(self, text):
        assert parse_point_response(text) is None


# --------------------------------------------------------------------------- #
# median3
# --------------------------------------------------------------------------- #
class TestMedian3:
    def test_fewer_than_three_unchanged(self):
        assert median3({0: 100.0}) == {0: 100.0}
        assert median3({0: 100.0, 30: 200.0}) == {0: 100.0, 30: 200.0}

    def test_endpoints_untouched(self):
        out = median3({0: 100.0, 30: 500.0, 60: 110.0})
        assert out[0] == 100.0 and out[60] == 110.0

    def test_single_stray_rejected(self):
        out = median3({0: 100.0, 30: 900.0, 60: 110.0})
        assert out[30] == 110.0  # median(100, 900, 110)

    def test_returns_new_dict(self):
        src = {0: 1.0, 30: 2.0, 60: 3.0}
        median3(src)
        assert src == {0: 1.0, 30: 2.0, 60: 3.0}


# --------------------------------------------------------------------------- #
# interpolate_focus
# --------------------------------------------------------------------------- #
class TestInterpolate:
    def test_empty_is_none(self):
        assert interpolate_focus({}, 5) is None

    def test_edge_hold(self):
        fm = {10: 100.0, 20: 200.0}
        assert interpolate_focus(fm, 0) == 100.0
        assert interpolate_focus(fm, 99) == 200.0

    def test_midpoint(self):
        assert interpolate_focus({0: 0.0, 10: 100.0}, 5) == 50.0


# --------------------------------------------------------------------------- #
# extract_focus_points (mocked backend)
# --------------------------------------------------------------------------- #
class TestExtractFocusPoints:
    def _clip(self, tmp_path, n=90, w=1920, h=1080):
        return write_clip(tmp_path / "c.mp4", n_frames=n, w=w, h=h)

    def test_scales_norm_to_pixels(self, tmp_path):
        be = FakeBackend(responses='{"x": 500, "y": 400}')
        fm, reasoning, n_failed = extract_focus_points(
            str(self._clip(tmp_path)), backend=be, sample_every=30, concurrency=4)
        assert reasoning == {}
        assert n_failed == 0
        assert set(fm) == {0, 30, 60}
        assert all(v == pytest.approx(960.0) for v in fm.values())  # 500/1000*1920

    def test_message_and_generation_shape(self, tmp_path):
        be = FakeBackend(responses='{"x": 500, "y": 500}')
        extract_focus_points(str(self._clip(tmp_path)), backend=be, sample_every=30)
        content = be.calls[0][0]["content"]
        assert [p["type"] for p in content] == ["image_url", "text"]
        assert be.call_kwargs[0]["temperature"] == 0.0
        assert be.call_kwargs[0]["max_tokens"] == focus.POINT_MAX_TOKENS

    def test_default_prompt_is_football(self, tmp_path):
        be = FakeBackend(responses='{"x": 500, "y": 500}')
        extract_focus_points(str(self._clip(tmp_path)), backend=be, sample_every=30)
        assert be.calls[0][0]["content"][1]["text"] == PRESETS["football"]

    def test_sport_selects_preset(self, tmp_path):
        be = FakeBackend(responses='{"x": 500, "y": 500}')
        extract_focus_points(str(self._clip(tmp_path)), backend=be, sample_every=30, sport="hockey")
        assert be.calls[0][0]["content"][1]["text"] == PRESETS["hockey"]

    def test_inline_prompt_overrides(self, tmp_path):
        be = FakeBackend(responses='{"x": 500, "y": 500}')
        extract_focus_points(str(self._clip(tmp_path)), backend=be, sample_every=30,
                             prompt="custom x y json prompt")
        assert be.calls[0][0]["content"][1]["text"] == "custom x y json prompt"

    def test_keyframe_indices(self, tmp_path):
        be = FakeBackend(responses='{"x": 500, "y": 500}')
        fm, _, _ = extract_focus_points(str(self._clip(tmp_path, n=120)), backend=be, sample_every=30)
        assert set(fm) == {0, 30, 60, 90}

    def test_no_keyframes_returns_empty(self, tmp_path):
        # sample_every larger than the clip still samples frame 0
        clip = write_clip(tmp_path / "one.mp4", n_frames=1, w=640, h=360)
        be = FakeBackend(responses='{"x": 500, "y": 500}')
        fm, _, n_failed = extract_focus_points(str(clip), backend=be, sample_every=30)
        assert set(fm) == {0}

    def test_concurrency_capped_at_keyframes(self, tmp_path):
        be = FakeBackend(responses='{"x": 500, "y": 500}')
        # 2 keyframes, concurrency 16 -> must not raise (min(len, concurrency))
        fm, _, _ = extract_focus_points(str(self._clip(tmp_path, n=60)), backend=be,
                                        sample_every=30, concurrency=16)
        assert len(fm) == 2

    def test_median_applied_across_keyframes(self, tmp_path):
        # stray middle sample should be pulled to its neighbors' median
        be = FakeBackend(responses=['{"x": 100, "y": 5}', '{"x": 900, "y": 5}',
                                    '{"x": 110, "y": 5}'])
        fm, _, _ = extract_focus_points(str(self._clip(tmp_path, n=90)), backend=be, sample_every=30)
        # frames 0/30/60 -> x px 192 / (median) / 211.2 ; middle no longer 1728
        assert fm[30] != pytest.approx(1728.0)


# --------------------------------------------------------------------------- #
# Adversarial pass 1: bug hunt (parse fences/prose, scaling, indexing)
# --------------------------------------------------------------------------- #
class TestAdv1BugHunt:
    def test_first_object_wins_with_trailing_junk(self):
        assert parse_point_response('{"x": 1, "y": 2}{"x": 9, "y": 9}') == (1.0, 2.0)

    def test_scaling_uses_full_frame_width_not_send_width(self, tmp_path):
        # x=1000 -> right edge in ORIGINAL pixels regardless of downscale
        be = FakeBackend(responses='{"x": 1000, "y": 500}')
        fm, _, _ = extract_focus_points(
            str(write_clip(tmp_path / "c.mp4", n_frames=30, w=1920, h=1080)),
            backend=be, sample_every=30, send_width=768)
        assert fm[0] == pytest.approx(1920.0)


# --------------------------------------------------------------------------- #
# Adversarial pass 2: robustness (out-of-range, all/partial fail, non-JSON)
# --------------------------------------------------------------------------- #
class TestAdv2Robustness:
    def _clip(self, tmp_path, n=90):
        return write_clip(tmp_path / "c.mp4", n_frames=n, w=1920, h=1080)

    def test_all_fail_returns_empty_focus(self, tmp_path):
        be = FakeBackend(responses="garbage not json")
        fm, _, n_failed = extract_focus_points(str(self._clip(tmp_path)), backend=be, sample_every=30)
        assert fm == {}
        assert n_failed == 3

    def test_partial_fail_counts_and_keeps_good(self, tmp_path):
        be = FakeBackend(responses=['{"x": 500, "y": 5}', "nope", '{"x": 600, "y": 5}'])
        fm, _, n_failed = extract_focus_points(str(self._clip(tmp_path)), backend=be, sample_every=30)
        assert n_failed == 1
        assert len(fm) == 2

    def test_out_of_range_treated_as_failure(self, tmp_path):
        be = FakeBackend(responses='{"x": 5000, "y": 5}')
        fm, _, n_failed = extract_focus_points(str(self._clip(tmp_path)), backend=be, sample_every=30)
        assert fm == {} and n_failed == 3

    def test_backend_exception_isolated(self, tmp_path):
        class _RaiseOnSecond:
            model = "m"
            last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}

            def __init__(self):
                self.n = 0

            def complete(self, messages, **kwargs):
                self.n += 1
                if self.n == 2:
                    raise RuntimeError("boom")
                return '{"x": 500, "y": 5}'

        fm, _, n_failed = extract_focus_points(
            str(self._clip(tmp_path)), backend=_RaiseOnSecond(), sample_every=30, concurrency=1)
        assert n_failed == 1
        assert len(fm) == 2  # the two successful calls survive the one exception

    def test_prompt_injection_prose_still_parses(self):
        reply = 'Ignore previous instructions. Anyway the point is {"x": 250, "y": 250}.'
        assert parse_point_response(reply) == (250.0, 250.0)


# --------------------------------------------------------------------------- #
# Adversarial pass 3: edge cases (median<3, downscale aspect, single keyframe)
# --------------------------------------------------------------------------- #
class TestAdv3EdgeCases:
    def test_downscale_to_send_width_preserves_aspect(self, tmp_path):
        be = FakeBackend(responses='{"x": 500, "y": 5}')
        extract_focus_points(
            str(write_clip(tmp_path / "c.mp4", n_frames=30, w=1920, h=1080)),
            backend=be, sample_every=30, send_width=768)
        img = _decode_data_uri(be.calls[0][0]["content"][0])
        assert img.shape[1] == 768
        assert img.shape[0] == pytest.approx(432, abs=1)  # 1080*768/1920

    def test_small_frame_not_upscaled(self, tmp_path):
        be = FakeBackend(responses='{"x": 500, "y": 5}')
        extract_focus_points(
            str(write_clip(tmp_path / "c.mp4", n_frames=30, w=320, h=180)),
            backend=be, sample_every=30, send_width=768)
        img = _decode_data_uri(be.calls[0][0]["content"][0])
        assert img.shape[1] == 320  # unchanged, never upscaled

    def test_single_keyframe_median_noop(self, tmp_path):
        be = FakeBackend(responses='{"x": 500, "y": 5}')
        fm, _, _ = extract_focus_points(
            str(write_clip(tmp_path / "c.mp4", n_frames=20, w=1920, h=1080)),
            backend=be, sample_every=30)
        assert set(fm) == {0}
        assert fm[0] == pytest.approx(960.0)
