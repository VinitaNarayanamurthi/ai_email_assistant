from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.draft_writer_agent import (
    format_constraints,
    format_retry_instructions,
    format_structure_requirements,
    format_tone_directives,
    is_surgical_edit,
    load_tone_sample,
    draft_writer_agent,
)

SAMPLE_DRAFT = {
    "subject_line": "Test Subject",
    "salutation": "Dear Test,",
    "body_paragraphs": ["Paragraph one.", "Paragraph two."],
    "closing": "Please contact us.",
    "sign_off": "Sincerely,",
}

SAMPLE_STATE = {
    "raw_input": "Write an apology email",
    "intent": "apology",
    "tone_directives": ["Be formal", "No contractions"],
    "parsed_context": {
        "recipient_type": "client",
        "recipient_name": "Sarah",
        "subject_hint": "delayed shipment",
        "constraints": ["apologetic"],
        "urgency": False,
        "context_notes": None,
    },
    "retry_issues": [],
    "retry_count": 0,
    "active_model": "gpt-4o",
}


class TestFormatHelpers:
    def test_format_tone_directives_numbered(self) -> None:
        directives = ["Be formal", "No contractions", "Use full sentences"]
        result = format_tone_directives(directives)
        assert result.startswith("1. Be formal")
        assert "2. No contractions" in result
        assert "3. Use full sentences" in result

    def test_format_tone_directives_empty(self) -> None:
        assert format_tone_directives([]) == ""

    def test_format_structure_requirements_apology(self) -> None:
        result = format_structure_requirements("apology")
        assert "acknowledgment" in result.lower()
        assert "remedy" in result.lower()

    def test_format_structure_requirements_outreach_requires_cta(self) -> None:
        result = format_structure_requirements("outreach")
        assert "call-to-action" in result.lower() or "REQUIRED" in result

    def test_format_structure_requirements_fallback_for_unknown(self) -> None:
        result = format_structure_requirements("nonexistent_intent")
        assert "opener" in result.lower() or "body" in result.lower()

    def test_format_constraints_with_user_constraints(self) -> None:
        ctx = {"constraints": ["concise", "apologetic"], "urgency": False}
        result = format_constraints(ctx, [])
        assert "concise" in result
        assert "apologetic" in result

    def test_format_constraints_urgent(self) -> None:
        ctx = {"constraints": [], "urgency": True}
        result = format_constraints(ctx, [])
        assert "urgent" in result.lower()

    def test_format_constraints_no_constraints_returns_default(self) -> None:
        result = format_constraints({}, [])
        assert "No special constraints" in result

    def test_format_retry_instructions_empty_on_first_run(self) -> None:
        result = format_retry_instructions([], 0)
        assert result == ""

    def test_format_retry_instructions_with_issues(self) -> None:
        issues = ["Fix contractions", "Add CTA"]
        result = format_retry_instructions(issues, 1)
        assert "retry attempt 1" in result.lower()
        assert "Fix contractions" in result
        assert "Add CTA" in result

    def test_format_retry_instructions_zero_count_returns_empty(self) -> None:
        issues = ["Fix something"]
        result = format_retry_instructions(issues, 0)
        assert result == ""


class TestLoadToneSample:
    def test_existing_file_returns_formatted_section(self, tmp_path: Path) -> None:
        sample = tmp_path / "formal_apology.txt"
        sample.write_text("Dear Client,\nSorry for the delay.", encoding="utf-8")

        result = load_tone_sample(str(sample))
        assert "Reference example" in result
        assert "Dear Client" in result

    def test_missing_file_returns_empty_string(self) -> None:
        result = load_tone_sample("/nonexistent/path/sample.txt")
        assert result == ""

    def test_none_ref_returns_empty_string(self) -> None:
        assert load_tone_sample(None) == ""


class TestIsSurgicalEdit:
    def test_shorter_signal_is_surgical(self) -> None:
        assert is_surgical_edit("make it shorter", SAMPLE_DRAFT) is True

    def test_add_signal_is_surgical(self) -> None:
        assert is_surgical_edit("add a P.S. section", SAMPLE_DRAFT) is True

    def test_remove_signal_is_surgical(self) -> None:
        assert is_surgical_edit("remove the third paragraph", SAMPLE_DRAFT) is True

    def test_no_prior_draft_is_not_surgical(self) -> None:
        assert is_surgical_edit("make it shorter", None) is False

    def test_full_prompt_is_not_surgical(self) -> None:
        full_prompt = "Write a professional apology email to my client"
        assert is_surgical_edit(full_prompt, SAMPLE_DRAFT) is False


class TestDraftWriterAgent:
    def test_successful_generation_returns_draft(self) -> None:
        with patch("src.agents.draft_writer_agent._build_draft_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = dict(SAMPLE_DRAFT)
            mock_build.return_value = mock_chain

            result = draft_writer_agent(SAMPLE_STATE)

        assert result["draft_error"] is None
        draft = result["draft"]
        assert isinstance(draft, dict)
        assert draft["subject_line"] == "Test Subject"

    def test_llm_failure_after_retries_returns_draft_error(self) -> None:
        with patch("src.agents.draft_writer_agent._build_draft_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = RuntimeError("API error")
            mock_build.return_value = mock_chain

            result = draft_writer_agent(SAMPLE_STATE)

        assert result["draft"] is None
        assert result["draft_error"] is not None
        assert "failed" in str(result["draft_error"]).lower()

    def test_active_model_override_used(self) -> None:
        state = {**SAMPLE_STATE, "active_model": "claude-3-5-sonnet-20241022"}
        with patch("src.agents.draft_writer_agent._build_draft_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = dict(SAMPLE_DRAFT)
            mock_build.return_value = mock_chain

            draft_writer_agent(state)

        mock_build.assert_called_once()
        call_args = mock_build.call_args
        assert call_args[0][0] == "claude-3-5-sonnet-20241022"

    def test_retry_increases_temperature(self) -> None:
        state = {**SAMPLE_STATE, "retry_count": 2}
        with patch("src.agents.draft_writer_agent._build_draft_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = dict(SAMPLE_DRAFT)
            mock_build.return_value = mock_chain

            draft_writer_agent(state)

        call_args = mock_build.call_args
        temperature = call_args[0][1]
        assert temperature > 0.4

    def test_surgical_edit_path_invoked_on_signal(self) -> None:
        state = {
            **SAMPLE_STATE,
            "raw_input": "make it shorter",
            "draft": SAMPLE_DRAFT,
        }
        with patch(
            "src.agents.draft_writer_agent.apply_surgical_edit"
        ) as mock_surgical:
            mock_surgical.return_value = {"draft": SAMPLE_DRAFT, "draft_error": None}
            result = draft_writer_agent(state)

        mock_surgical.assert_called_once()

    def test_retry_instructions_injected_when_retry_issues_present(self) -> None:
        state = {
            **SAMPLE_STATE,
            "retry_count": 1,
            "retry_issues": ["Fix contractions in paragraph 2."],
        }
        with patch("src.agents.draft_writer_agent._build_draft_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = dict(SAMPLE_DRAFT)
            mock_build.return_value = mock_chain

            draft_writer_agent(state)

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert "retry_instructions" in call_kwargs
        assert "Fix contractions" in call_kwargs["retry_instructions"]
