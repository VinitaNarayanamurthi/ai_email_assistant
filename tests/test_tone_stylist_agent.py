from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.tone_stylist_agent import (
    SYSTEM_DEFAULT_TONE,
    TONE_DIRECTIVES,
    TONE_INTENT_MODIFIERS,
    detect_tone_change_in_refinement,
    resolve_tone,
    resolve_tone_sample,
    tone_stylist_agent,
)


class TestResolveTone:
    def test_ui_selection_wins_over_prompt_hint(self) -> None:
        state = {
            "ui_tone_selection": "casual",
            "parsed_context": {"tone_hint": "formal"},
            "user_profile": {"default_tone": "assertive"},
        }
        tone, source = resolve_tone(state)
        assert tone == "casual"
        assert source == "ui_selection"

    def test_prompt_hint_wins_over_profile(self) -> None:
        state = {
            "ui_tone_selection": None,
            "parsed_context": {"tone_hint": "assertive"},
            "user_profile": {"default_tone": "casual"},
        }
        tone, source = resolve_tone(state)
        assert tone == "assertive"
        assert source == "prompt_hint"

    def test_profile_default_used_when_no_ui_or_hint(self) -> None:
        state = {
            "ui_tone_selection": None,
            "parsed_context": {"tone_hint": None},
            "user_profile": {"default_tone": "empathetic"},
        }
        tone, source = resolve_tone(state)
        assert tone == "empathetic"
        assert source == "profile_default"

    def test_system_default_when_nothing_set(self) -> None:
        state = {
            "ui_tone_selection": None,
            "parsed_context": {},
            "user_profile": {},
        }
        tone, source = resolve_tone(state)
        assert tone == SYSTEM_DEFAULT_TONE
        assert source == "system_default"

    def test_invalid_ui_selection_skipped(self) -> None:
        state = {
            "ui_tone_selection": "NOT_A_TONE",
            "parsed_context": {"tone_hint": "casual"},
            "user_profile": {},
        }
        tone, source = resolve_tone(state)
        assert tone == "casual"
        assert source == "prompt_hint"

    def test_invalid_prompt_hint_skipped(self) -> None:
        state = {
            "ui_tone_selection": None,
            "parsed_context": {"tone_hint": "INVALID"},
            "user_profile": {"default_tone": "formal"},
        }
        tone, source = resolve_tone(state)
        assert tone == "formal"
        assert source == "profile_default"


class TestResolveToneSample:
    def test_specific_tone_intent_file_found(self) -> None:
        path = resolve_tone_sample("formal", "apology")
        assert path is not None
        assert "formal_apology" in path

    def test_tone_only_fallback_when_specific_missing(self) -> None:
        path = resolve_tone_sample("casual", "other")
        assert path is not None
        assert "casual" in path

    def test_returns_none_when_no_file_exists(self) -> None:
        path = resolve_tone_sample("diplomatic", "internal_update")
        assert path is None

    def test_user_specific_file_takes_priority(self, tmp_path: Path) -> None:
        user_file = tmp_path / "formal_apology_user.txt"
        user_file.write_text("User approved sample", encoding="utf-8")

        with patch("src.agents.tone_stylist_agent.TONE_SAMPLES_DIR", tmp_path):
            generic_file = tmp_path / "formal_apology.txt"
            generic_file.write_text("Generic sample", encoding="utf-8")
            path = resolve_tone_sample("formal", "apology")

        assert path is not None
        assert "user" in path


class TestToneIntentModifiers:
    def test_formal_apology_adds_modifiers(self) -> None:
        state = {
            "ui_tone_selection": None,
            "parsed_context": {"tone_hint": "formal"},
            "user_profile": {},
            "intent": "apology",
        }
        result = tone_stylist_agent(state)
        directives = result["tone_directives"]
        assert isinstance(directives, list)
        assert len(directives) > len(TONE_DIRECTIVES["formal"])
        modifier_directives = TONE_INTENT_MODIFIERS[("formal", "apology")]
        for directive in modifier_directives:
            assert directive in directives

    def test_assertive_outreach_adds_modifiers(self) -> None:
        state = {
            "ui_tone_selection": "assertive",
            "parsed_context": {},
            "user_profile": {},
            "intent": "outreach",
        }
        result = tone_stylist_agent(state)
        directives = result["tone_directives"]
        modifier_directives = TONE_INTENT_MODIFIERS[("assertive", "outreach")]
        for directive in modifier_directives:
            assert directive in directives

    def test_no_modifier_for_unregistered_combo(self) -> None:
        state = {
            "ui_tone_selection": "formal",
            "parsed_context": {},
            "user_profile": {},
            "intent": "thank_you",
        }
        result = tone_stylist_agent(state)
        directives = result["tone_directives"]
        assert len(directives) == len(TONE_DIRECTIVES["formal"])


class TestDetectToneChange:
    def test_casual_signal_detected(self) -> None:
        assert detect_tone_change_in_refinement("make it more casual") == "casual"

    def test_formal_signal_detected(self) -> None:
        assert detect_tone_change_in_refinement("make it more professional") == "formal"

    def test_assertive_signal_detected(self) -> None:
        assert detect_tone_change_in_refinement("be more direct") == "assertive"

    def test_no_tone_signal_returns_none(self) -> None:
        assert detect_tone_change_in_refinement("make it shorter") is None


class TestToneStylistAgent:
    def test_agent_returns_all_required_keys(self) -> None:
        state = {
            "ui_tone_selection": "formal",
            "parsed_context": {},
            "user_profile": {},
            "intent": "other",
        }
        result = tone_stylist_agent(state)
        assert "tone" in result
        assert "tone_directives" in result
        assert "tone_resolution_log" in result

    def test_refinement_override_changes_tone(self) -> None:
        state = {
            "raw_input": "make it more casual",
            "tone": "formal",
            "ui_tone_selection": None,
            "parsed_context": {},
            "user_profile": {},
            "intent": "follow_up",
        }
        result = tone_stylist_agent(state)
        assert result["tone"] == "casual"
        assert "refinement_override" in str(result["tone_resolution_log"])

    def test_no_prior_tone_no_refinement_override(self) -> None:
        state = {
            "raw_input": "make it more casual",
            "ui_tone_selection": "formal",
            "parsed_context": {},
            "user_profile": {},
            "intent": "other",
        }
        result = tone_stylist_agent(state)
        assert result["tone"] == "formal"
