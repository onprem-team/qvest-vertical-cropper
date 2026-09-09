# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest.US, LLC
# SPDX-License-Identifier: Apache-2.0
"""Scoreboard data models and majority-vote consensus merge."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class TeamInfo:
    name: str
    score: int | None = None


@dataclass
class ScoreboardReading:
    frame_idx: int
    sport: str | None = None
    event: str | None = None
    teams: list[TeamInfo] = field(default_factory=list)
    period: str | None = None
    clock: str | None = None
    extra_fields: list[dict[str, str]] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class GameState:
    sport: str
    event: str | None = None
    home_team: str = ""
    away_team: str = ""
    home_score: int | None = None
    away_score: int | None = None
    period: str | None = None
    clock: str | None = None
    extra_fields: list[dict[str, str]] = field(default_factory=list)


def is_matchup(state: GameState) -> bool:
    """Determine if the game state represents a head-to-head matchup."""
    has_teams = bool(state.home_team and state.away_team)
    has_score = state.home_score is not None or state.away_score is not None
    return has_teams and has_score


def _majority_vote(values: list[str]) -> str | None:
    """Return the most common non-empty value, or None."""
    filtered = [v.strip() for v in values if v and v.strip()]
    if not filtered:
        return None
    counter = Counter(filtered)
    return counter.most_common(1)[0][0]


def _majority_vote_int(values: list[int | None]) -> int | None:
    """Return the mode of non-None integer values."""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    counter = Counter(filtered)
    return counter.most_common(1)[0][0]


def _normalize_team_name(name: str) -> str:
    return name.strip().upper()


def merge_readings(readings: list[ScoreboardReading]) -> GameState | None:
    """Merge multiple VLM readings into a single consensus GameState.

    Thin wrapper over :func:`merge_readings_with_debug` (single source of truth).
    Returns None if there is insufficient data for a meaningful overlay
    (e.g., fewer than 50% of readings produced any valid data).
    """
    state, _debug = merge_readings_with_debug(readings)
    return state


@dataclass
class ScoreboardDebugInfo:
    """Debug info for the scoreboard consensus process."""
    total_readings: int = 0
    valid_readings: int = 0
    threshold: int = 0
    sport_votes: dict[str, int] = field(default_factory=dict)
    event_votes: dict[str, int] = field(default_factory=dict)
    field_votes: dict[str, dict[str, int]] = field(default_factory=dict)
    included_fields: list[str] = field(default_factory=list)
    excluded_fields: list[str] = field(default_factory=list)
    game_state_json: str = ""

    def to_lines(self) -> list[str]:
        """Format debug info as text lines for overlay rendering."""
        lines = [
            f"SCOREBOARD OCR: {self.valid_readings}/{self.total_readings} valid, threshold={self.threshold}",
            f"Sport: {self.sport_votes}" if self.sport_votes else "Sport: (none)",
        ]
        if self.event_votes:
            lines.append(f"Event: {self.event_votes}")
        lines.append("--- INCLUDED ---")
        for label in self.included_fields:
            votes = self.field_votes.get(label, {})
            top = max(votes, key=votes.get) if votes else "?"
            count = sum(votes.values())
            lines.append(f"  {label}: {top} ({count}/{self.valid_readings} frames)")
        if self.excluded_fields:
            lines.append("--- EXCLUDED (below threshold) ---")
            for label in self.excluded_fields:
                votes = self.field_votes.get(label, {})
                count = sum(votes.values())
                lines.append(f"  {label}: ({count}/{self.valid_readings} frames)")
        return lines


def merge_readings_with_debug(
    readings: list[ScoreboardReading],
) -> tuple[GameState | None, ScoreboardDebugInfo]:
    """Like merge_readings but also returns debug info about the consensus."""
    debug = ScoreboardDebugInfo(total_readings=len(readings))

    valid = [r for r in readings if r.confidence > 0]
    debug.valid_readings = len(valid)
    if not valid:
        return None, debug

    # Sport votes
    sports = [r.sport for r in valid if r.sport]
    debug.sport_votes = dict(Counter(sports).most_common())
    sport = _majority_vote(sports) or "unknown"

    events = [r.event for r in valid if r.event]
    debug.event_votes = dict(Counter(events).most_common())
    event = _majority_vote(events)

    home_names: list[str] = []
    away_names: list[str] = []
    home_scores: list[int | None] = []
    away_scores: list[int | None] = []
    for r in valid:
        if len(r.teams) >= 2:
            home_names.append(_normalize_team_name(r.teams[0].name))
            away_names.append(_normalize_team_name(r.teams[1].name))
            home_scores.append(r.teams[0].score)
            away_scores.append(r.teams[1].score)

    home_team = _majority_vote(home_names) or ""
    away_team = _majority_vote(away_names) or ""
    home_score = _majority_vote_int(home_scores)
    away_score = _majority_vote_int(away_scores)

    sorted_readings = sorted(valid, key=lambda r: r.frame_idx)
    period: str | None = None
    clock: str | None = None
    for r in reversed(sorted_readings):
        if r.period and period is None:
            period = r.period
        if r.clock and clock is None:
            clock = r.clock
        if period and clock:
            break

    # Extra fields with vote tracking
    label_values: dict[str, list[str]] = {}
    for r in valid:
        for ef in r.extra_fields:
            label = ef.get("label", "").strip()
            value = ef.get("value", "").strip()
            if label and value:
                label_values.setdefault(label, []).append(value)

    for label, values in label_values.items():
        debug.field_votes[label] = dict(Counter(values).most_common())

    threshold = max(1, -(-int(len(valid) * 3) // 10))
    if len(valid) > 1:
        threshold = max(2, threshold)
    debug.threshold = threshold

    extra_fields: list[dict[str, str]] = []
    for label, values in label_values.items():
        if len(values) >= threshold:
            consensus_value = _majority_vote(values)
            if consensus_value:
                extra_fields.append({"label": label, "value": consensus_value})
                debug.included_fields.append(label)
        else:
            debug.excluded_fields.append(label)

    if not home_team and not away_team and not extra_fields:
        return None, debug

    gs = GameState(
        sport=sport, event=event,
        home_team=home_team, away_team=away_team,
        home_score=home_score, away_score=away_score,
        period=period, clock=clock,
        extra_fields=extra_fields,
    )
    debug.game_state_json = json.dumps(
        {
            "sport": gs.sport, "event": gs.event,
            "home": f"{gs.home_team} {gs.home_score}", "away": f"{gs.away_team} {gs.away_score}",
            "period": gs.period, "clock": gs.clock,
            "extra_fields": gs.extra_fields,
        },
        indent=2,
    )
    return gs, debug
