"""Unit + adversarial tests for sport-configurable pointing prompts."""
from __future__ import annotations

import logging

import pytest

from v_cropper import prompts

# The football preset MUST stay byte-identical to the R&D v0013 winner (parity contract).
V0013_FOOTBALL = (
    "American football broadcast frame. Point at the football's CURRENT location: the "
    "player holding or carrying the ball (e.g. the quarterback in the pocket counts), or "
    "the ball itself if in flight or loose. Only if no ball is in play, point at the "
    "center of the player formation. Answer with ONLY this JSON, nothing else: "
    '{"x": <int 0-1000>, "y": <int 0-1000>} \u2014 normalized image coordinates '
    "(x: 0=left edge, 1000=right edge; y: 0=top)."
)


@pytest.fixture(autouse=True)
def _clear_prompt_env(monkeypatch):
    for var in ("VCROPPER_PROMPT", "VCROPPER_PROMPT_FILE", "VCROPPER_SPORT"):
        monkeypatch.delenv(var, raising=False)


class TestPresets:
    def test_football_is_default(self):
        assert prompts.resolve_prompt() == prompts.PRESETS["football"]

    def test_football_matches_v0013_byte_for_byte(self):
        assert prompts.PRESETS["football"] == V0013_FOOTBALL

    def test_all_presets_request_point_json(self):
        for name, text in prompts.PRESETS.items():
            assert '{"x": <int 0-1000>, "y": <int 0-1000>}' in text, name

    def test_available_sports_sorted(self):
        assert prompts.available_sports() == sorted(prompts.PRESETS)


class TestResolverPrecedence:
    def test_inline_beats_everything(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("file prompt with x y json")
        got = prompts.resolve_prompt(inline="inline x y json", prompt_file=str(f),
                                     sport="hockey")
        assert got == "inline x y json"

    def test_file_beats_sport(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("custom x y json")
        assert prompts.resolve_prompt(prompt_file=str(f), sport="hockey") == "custom x y json"

    def test_sport_beats_default(self):
        assert prompts.resolve_prompt(sport="hockey") == prompts.PRESETS["hockey"]

    def test_sport_case_insensitive(self):
        assert prompts.resolve_prompt(sport="Hockey") == prompts.PRESETS["hockey"]

    def test_unknown_sport_raises(self):
        with pytest.raises(ValueError, match="unknown sport"):
            prompts.resolve_prompt(sport="curling")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            prompts.resolve_prompt(prompt_file=str(tmp_path / "nope.txt"))

    def test_file_read_returns_contents(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("read me x y json")
        assert prompts.resolve_prompt(prompt_file=str(f)) == "read me x y json"


class TestEnvFallback:
    def test_env_prompt(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PROMPT", "env x y json")
        assert prompts.resolve_prompt() == "env x y json"

    def test_env_sport(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_SPORT", "soccer")
        assert prompts.resolve_prompt() == prompts.PRESETS["soccer"]

    def test_env_prompt_file(self, monkeypatch, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("env file x y json")
        monkeypatch.setenv("VCROPPER_PROMPT_FILE", str(f))
        assert prompts.resolve_prompt() == "env file x y json"

    def test_flag_arg_beats_env(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_SPORT", "soccer")
        assert prompts.resolve_prompt(sport="hockey") == prompts.PRESETS["hockey"]


# --- Adversarial pass 1: bug hunt (precedence conflicts, default selection) ---
class TestAdv1Precedence:
    def test_inline_over_env_file_and_env_sport(self, monkeypatch, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("file x y json")
        monkeypatch.setenv("VCROPPER_PROMPT_FILE", str(f))
        monkeypatch.setenv("VCROPPER_SPORT", "soccer")
        assert prompts.resolve_prompt(inline="win x y json") == "win x y json"

    def test_env_prompt_over_flag_sport(self, monkeypatch):
        # inline (env) sits above sport in precedence even when sport is passed as a flag
        monkeypatch.setenv("VCROPPER_PROMPT", "env-inline x y json")
        assert prompts.resolve_prompt(sport="hockey") == "env-inline x y json"

    def test_nothing_given_is_football(self):
        assert prompts.resolve_prompt() == prompts.PRESETS["football"]


# --- Adversarial pass 2: robustness (empty prompt, missing JSON instr, bad file) ---
class TestAdv2Robustness:
    def test_whitespace_inline_falls_through_to_sport(self):
        assert prompts.resolve_prompt(inline="   ", sport="hockey") == prompts.PRESETS["hockey"]

    def test_empty_env_prompt_falls_through(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PROMPT", "  \n ")
        assert prompts.resolve_prompt() == prompts.PRESETS["football"]

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("   \n")
        with pytest.raises(ValueError, match="empty"):
            prompts.resolve_prompt(prompt_file=str(f))

    def test_prompt_without_json_instruction_warns_but_returns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="v_cropper.prompts"):
            got = prompts.resolve_prompt(inline="just follow the ball")
        assert got == "just follow the ball"
        assert any("JSON" in r.message for r in caplog.records)

    def test_unreadable_dir_as_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            prompts.resolve_prompt(prompt_file=str(tmp_path))  # a directory, not a file


# --- Adversarial pass 3: edge cases (unicode, large file, trailing newline) + verify ---
class TestAdv3EdgeCases:
    def test_unicode_prompt_preserved(self, tmp_path):
        f = tmp_path / "u.txt"
        txt = 'Señala el balón — {"x": 0-1000, "y": 0-1000} json'
        f.write_text(txt, encoding="utf-8")
        assert prompts.resolve_prompt(prompt_file=str(f)) == txt

    def test_large_file(self, tmp_path):
        f = tmp_path / "big.txt"
        body = 'x y json ' + ("context line\n" * 5000)
        f.write_text(body)
        assert prompts.resolve_prompt(prompt_file=str(f)) == body

    def test_trailing_newline_preserved(self, tmp_path):
        f = tmp_path / "n.txt"
        f.write_text("point at ball x y json\n")
        assert prompts.resolve_prompt(prompt_file=str(f)).endswith("\n")

    def test_every_preset_resolves_nonempty(self):
        for sport in prompts.available_sports():
            out = prompts.resolve_prompt(sport=sport)
            assert out and out.strip()
