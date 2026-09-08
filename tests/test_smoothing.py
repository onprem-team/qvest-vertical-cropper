"""Unit + adversarial tests for the critically-damped spring smoother."""
from __future__ import annotations

import math

import pytest

from v_cropper.smoothing import DEFAULT_SPRING_K, SpringPanner


def _reference_smooth_spring(xs, k=DEFAULT_SPRING_K):
    """The R&D smooth_spring, inlined as the ground-truth reference."""
    c = 2.0 * math.sqrt(k)
    out = []
    x = float(xs[0])
    v = 0.0
    for target in xs:
        a = k * (float(target) - x) - c * v
        v += a
        x += v
        out.append(x)
    return out


def _run(xs, k=DEFAULT_SPRING_K):
    sp = SpringPanner(k=k)
    return [sp.update(x) for x in xs]


def _mean_abs_jerk(xs):
    d1 = [b - a for a, b in zip(xs, xs[1:], strict=False)]
    d2 = [b - a for a, b in zip(d1, d1[1:], strict=False)]
    d3 = [b - a for a, b in zip(d2, d2[1:], strict=False)]
    return sum(abs(j) for j in d3) / len(d3) if d3 else 0.0


class TestSpringBasics:
    def test_center_before_first_target(self):
        assert SpringPanner().center_x(1000) == 500.0

    def test_first_target_inits_at_rest(self):
        sp = SpringPanner()
        assert sp.update(300.0) == 300.0
        assert sp.center_x(1000) == 300.0

    def test_none_holds_position(self):
        sp = SpringPanner()
        sp.update(200.0)
        assert sp.update(None) == 200.0
        assert sp.center_x(1000) == 200.0

    def test_none_before_any_target_stays_center(self):
        sp = SpringPanner()
        assert sp.update(None) is None
        assert sp.center_x(640) == 320.0

    def test_converges_to_constant_target(self):
        sp = SpringPanner()
        sp.update(0.0)
        for _ in range(2000):
            sp.update(1000.0)
        assert sp.center_x(4000) == pytest.approx(1000.0, abs=1.0)


class TestSpringEquivalence:
    def test_matches_reference_stream(self):
        xs = [100, 100, 400, 400, 400, 900, 300, 300, 700, 50, 50, 1000]
        sp = SpringPanner(k=DEFAULT_SPRING_K)
        got = [sp.update(x) for x in xs]
        assert got == pytest.approx(_reference_smooth_spring(xs))

    def test_deterministic(self):
        xs = [10, 500, 500, 20, 900, 900, 200]
        run1 = [v for v in _run(xs)]
        run2 = [v for v in _run(xs)]
        assert run1 == run2


class TestJerkGuardrail:
    def test_smooth_on_moving_target_under_threshold(self):
        # interpolated target: hold, linear pan, step, hold — like a real play
        target = ([500.0] * 30
                  + [500.0 + i * (900.0 / 120) for i in range(120)]
                  + [1400.0] * 30
                  + [700.0] * 120)
        sp = SpringPanner(k=DEFAULT_SPRING_K)
        smoothed = [sp.update(t) for t in target]
        assert _mean_abs_jerk(smoothed) < 2.0  # scorer JERK_THRESHOLD


# --- Adversarial pass 1: bug hunt (recurrence, clamp handled by render, init) ---
class TestAdv1Recurrence:
    def test_single_spring_step_math(self):
        k = DEFAULT_SPRING_K
        sp = SpringPanner(k=k)
        sp.update(0.0)             # x=0, v=0
        got = sp.update(100.0)     # a=k*100; v=a; x=a
        assert got == pytest.approx(k * 100.0)

    def test_c_is_critical_damping(self):
        sp = SpringPanner(k=0.04)
        assert sp.c == pytest.approx(2 * math.sqrt(0.04))


# --- Adversarial pass 2: robustness (blow-up bounds, extreme/empty targets) ---
class TestAdv2Robustness:
    def test_no_runaway_on_extreme_step(self):
        sp = SpringPanner()
        sp.update(0.0)
        vals = [sp.update(1_000_000.0) for _ in range(500)]
        # critically damped: approaches target from below, never overshoots far past it
        assert max(vals) <= 1_000_000.0 * 1.05
        assert vals[-1] == pytest.approx(1_000_000.0, rel=1e-3)

    def test_all_none_sequence_holds_center(self):
        sp = SpringPanner()
        for _ in range(10):
            sp.update(None)
        assert sp.center_x(800) == 400.0

    def test_negative_and_large_targets(self):
        sp = SpringPanner()
        out = [sp.update(t) for t in (-500.0, 500.0, -500.0, 500.0)]
        assert all(math.isfinite(v) for v in out)


# --- Adversarial pass 3: edge cases (varied k, monotonic step approach) ---
class TestAdv3EdgeCases:
    @pytest.mark.parametrize("k", [0.01, 0.05, 0.1, 0.25])
    def test_various_k_converge(self, k):
        sp = SpringPanner(k=k)
        sp.update(0.0)
        for _ in range(5000):
            sp.update(1000.0)
        assert sp.center_x(4000) == pytest.approx(1000.0, abs=1.0)

    def test_step_response_no_large_overshoot(self):
        sp = SpringPanner(k=DEFAULT_SPRING_K)
        sp.update(0.0)
        vals = [sp.update(100.0) for _ in range(400)]
        assert max(vals) <= 105.0  # <=5% overshoot for critical damping
