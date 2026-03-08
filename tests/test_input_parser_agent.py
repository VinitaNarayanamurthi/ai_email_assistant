from unittest.mock import MagicMock, patch

import pytest

from src.agents.input_parser_agent import (
    input_parser_agent,
    is_refinement_input,
    validate_parsed_context,
)

SAMPLE_PROFILE = {
    "user_id": "test",
    "default_tone": "formal",
    "name": "Test User",
}

SAMPLE_PARSED_CONTEXT = {
    "recipient_type": "client",
    "recipient_name": "Sarah",
    "subject_hint": "delayed shipment",
    "tone_hint": "formal",
    "constraints": ["apologetic"],
    "urgency": False,
    "context_notes": None,
}


class TestIsRefinementInput:
    def test_make_it_shorter_is_refinement(self) -> None:
        assert is_refinement_input("make it shorter") is True

    def test_change_tone_is_refinement(self) -> None:
        assert is_refinement_input("more formal please") is True

    def test_full_prompt_is_not_refinement(self) -> None:
        assert is_refinement_input(
            "Write a professional apology email to my client Sarah about the delayed shipment"
        ) is False

    def test_long_input_is_not_refinement(self) -> None:
        long_input = "make it shorter but also " + "x" * 100
        assert is_refinement_input(long_input) is False

    def test_empty_input_is_not_refinement(self) -> None:
        assert is_refinement_input("") is False

    def test_add_signal_is_refinement(self) -> None:
        assert is_refinement_input("add a P.S. about the discount") is True


class TestValidateParsedContext:
    def test_valid_context_passes(self) -> None:
        is_valid, issues = validate_parsed_context(
            {"subject_hint": "delayed shipment", "recipient_type": "client"}
        )
        assert is_valid is True
        assert issues == []

    def test_missing_subject_hint_fails(self) -> None:
        is_valid, issues = validate_parsed_context({"recipient_type": "client"})
        assert is_valid is False
        assert any("subject" in i.lower() for i in issues)

    def test_missing_recipient_type_fails(self) -> None:
        is_valid, issues = validate_parsed_context({"subject_hint": "quarterly report"})
        assert is_valid is False
        assert len(issues) > 0


class TestInputParserAgent:
    def test_empty_input_returns_error(self) -> None:
        result = input_parser_agent({"raw_input": ""})
        assert result["parsed_context"] is None
        assert result["parse_error"] is not None
        assert "empty" in str(result["parse_error"]).lower()

    def test_whitespace_only_input_returns_error(self) -> None:
        result = input_parser_agent({"raw_input": "   "})
        assert result["parsed_context"] is None

    def test_normal_input_parsed_correctly(self) -> None:
        with patch("src.agents.input_parser_agent.build_parser_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = dict(SAMPLE_PARSED_CONTEXT)
            mock_build.return_value = mock_chain

            result = input_parser_agent(
                {"raw_input": "Write an apology email to my client Sarah about delayed shipment"}
            )

        assert result["parse_error"] is None
        ctx = result["parsed_context"]
        assert isinstance(ctx, dict)
        assert ctx["recipient_type"] == "client"
        assert ctx["recipient_name"] == "Sarah"

    def test_tone_hint_fallback_from_profile(self) -> None:
        parsed_without_tone = {**SAMPLE_PARSED_CONTEXT, "tone_hint": None}
        with patch("src.agents.input_parser_agent.build_parser_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = dict(parsed_without_tone)
            mock_build.return_value = mock_chain

            result = input_parser_agent(
                {
                    "raw_input": "Write an email to my client",
                    "user_profile": SAMPLE_PROFILE,
                }
            )

        ctx = result["parsed_context"]
        assert isinstance(ctx, dict)
        assert ctx.get("tone_hint") == "formal"

    def test_no_profile_no_tone_hint_stays_none(self) -> None:
        parsed_without_tone = {**SAMPLE_PARSED_CONTEXT, "tone_hint": None}
        with patch("src.agents.input_parser_agent.build_parser_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = dict(parsed_without_tone)
            mock_build.return_value = mock_chain

            result = input_parser_agent({"raw_input": "Write an email to my client"})

        ctx = result["parsed_context"]
        assert isinstance(ctx, dict)
        assert ctx.get("tone_hint") is None

    def test_llm_failure_returns_soft_fallback(self) -> None:
        with patch("src.agents.input_parser_agent.build_parser_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = RuntimeError("API unavailable")
            mock_build.return_value = mock_chain

            result = input_parser_agent({"raw_input": "Write an email to my manager"})

        assert result["parsed_context"] is not None
        assert "Parsing degraded" in str(result["parse_error"])

    def test_llm_failure_fallback_uses_raw_input_as_subject(self) -> None:
        with patch("src.agents.input_parser_agent.build_parser_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = RuntimeError("API error")
            mock_build.return_value = mock_chain

            raw = "Write an email about the quarterly review"
            result = input_parser_agent({"raw_input": raw})

        ctx = result["parsed_context"]
        assert isinstance(ctx, dict)
        assert raw[:60] in ctx.get("subject_hint", "")

    def test_refinement_input_calls_apply_refinement(self) -> None:
        expected = {"parsed_context": dict(SAMPLE_PARSED_CONTEXT), "parse_error": None}
        with patch(
            "src.agents.input_parser_agent.apply_refinement_to_context"
        ) as mock_refine:
            mock_refine.return_value = expected

            result = input_parser_agent(
                {
                    "raw_input": "make it shorter",
                    "parsed_context": SAMPLE_PARSED_CONTEXT,
                }
            )

        mock_refine.assert_called_once()
        assert result["parsed_context"] == SAMPLE_PARSED_CONTEXT
