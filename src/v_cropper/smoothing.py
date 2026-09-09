# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest.US, LLC
# SPDX-License-Identifier: Apache-2.0
"""Critically-damped spring smoothing of the per-frame crop center.

Ported from the R&D v0013 winner (``smooth_spring``): a mass-spring that chases the
interpolated focus target without overshoot. At ``k=0.05`` the jerk stays well under the
scorer's 2.0 px/frame^3 guardrail while tracking fast action, giving a smooth broadcast
pan. Stateful/streaming so ``render`` applies it one frame at a time.
"""
from __future__ import annotations

import math

DEFAULT_SPRING_K = 0.05


class SpringPanner:
    """Stateful critically-damped spring: ``a = k*(target-x) - 2*sqrt(k)*v``.

    ``update(target)`` advances one frame toward ``target`` (``None`` = hold position);
    ``center_x(frame_w)`` returns the current center (frame center until first target).
    Equivalent, frame-for-frame, to running ``smooth_spring`` over the target sequence.
    """

    def __init__(self, k: float = DEFAULT_SPRING_K):
        self.k = float(k)
        self.c = 2.0 * math.sqrt(self.k)
        self.x: float | None = None
        self.v = 0.0

    def update(self, target: float | None) -> float | None:
        if target is None:
            return self.x  # hold (may still be None before the first valid target)
        if self.x is None:
            # First target: initialize at rest on it (matches smooth_spring out[0]=xs[0]).
            self.x = float(target)
            self.v = 0.0
        else:
            a = self.k * (float(target) - self.x) - self.c * self.v
            self.v += a
            self.x += self.v
        return self.x

    def center_x(self, frame_w: float) -> float:
        return self.x if self.x is not None else frame_w / 2.0
