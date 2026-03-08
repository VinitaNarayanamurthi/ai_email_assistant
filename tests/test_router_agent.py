import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.agents.router_agent import (
    assemble_final_draft,
    determine_next_action,
    extract_style_from_edits,
    log_draft_to_profile,
    router_memory_agent,
    update_style_notes,
)

SAMPLE_DRAFT = {
    "subject_line": "Apology for Delayed Shipment",
    "salutation": "Dear Sarah,",
    "body_paragraphs": [
        "We sincerely apologize for the delay in your recent shipment.",
        "We are taking immediate steps to resolve this issue.",
    ],
    "closing": "Please do not hesitate to contact us.",
    "sign_off": "Best regards,",
}

SAMPLE_PROFILE = {
    "user_id": "test",
    "name": "Vinit",
    "company": "Acme Corp",
    "role": "Manager",
    "default_tone": "formal",
    "sign_off_preference": "Best regards",
    "writing_style_notes": "",
    "prior_drafts": [],
    "known_contacts": {},
}

PASS_STATE = {
    "raw_input": "Write an apology email",
    "review_result": {
        "grammar_score": 0.95,
        "tone_alignment_score": 0.90,
        "coherence_score": 0.92,
        "structure_complete": True,
        "issues": [],
        "verdict": "PASS",
    },
    "retry_count": 0,
    "active_model": "gpt-4o",
    "personalized_draft": SAMPLE_DRAFT,
    "user_profile": SAMPLE_PROFILE,
    "parsed_context": {
        "recipient_type": "client",
        "recipient_name": "Sarah",
        "subject_hint": "delayed shipment",
    },
    "intent": "apology",
    "tone": "formal",
    "prior_review_issues": [],
}

MOCK_MCP_CONFIG = {
    "primary_model": "gpt-4o",
    "fallback_chain": ["claude-3-5-sonnet-20241022", "cohere/command-r-plus"],
    "max_retries": 3,
    "fallback_on_retry": 2,
}


def _patched_config(state: dict) -> dict:
    with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
        with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=MOCK_MCP_CONFIG["fallback_on_retry"]):
                return determine_next_action(state)


class TestAssembleFinalDraft:
    def test_all_sections_formatted_correctly(self) -> None:
        result = assemble_final_draft(SAMPLE_DRAFT, "Vinit")
        assert "Subject: Apology for Delayed Shipment" in result
        assert "Dear Sarah," in result
        assert "We sincerely apologize" in result
        assert "Please do not hesitate" in result
        assert "Best regards," in result
        assert "Vinit" in result

    def test_empty_subject_omitted(self) -> None:
        draft = {**SAMPLE_DRAFT, "subject_line": ""}
        result = assemble_final_draft(draft)
        assert "Subject:" not in result

    def test_no_sender_name_omitted(self) -> None:
        result = assemble_final_draft(SAMPLE_DRAFT, "")
        assert "Vinit" not in result

    def test_empty_paragraphs_skipped(self) -> None:
        draft = {
            **SAMPLE_DRAFT,
            "body_paragraphs": ["Real paragraph.", "", "   "],
        }
        result = assemble_final_draft(draft)
        assert "Real paragraph." in result
        lines = result.split("\n")
        empty_runs = sum(1 for i, l in enumerate(lines) if l == "" and i > 0 and lines[i - 1] == "")
        assert empty_runs == 0 or empty_runs < 3


class TestDetermineNextAction:
    def test_pass_verdict_returns_success(self) -> None:
        state = {**PASS_STATE, "retry_count": 0}
        with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=2):
                with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
                    result = determine_next_action(state)
        assert result == "success"

    def test_fail_low_retry_returns_retry(self) -> None:
        state = {
            **PASS_STATE,
            "review_result": {**PASS_STATE["review_result"], "verdict": "FAIL"},
            "retry_count": 0,
        }
        with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=2):
                with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
                    result = determine_next_action(state)
        assert result == "retry"

    def test_fail_at_threshold_returns_fallback(self) -> None:
        state = {
            **PASS_STATE,
            "review_result": {**PASS_STATE["review_result"], "verdict": "FAIL"},
            "retry_count": 2,
            "active_model": "gpt-4o",
        }
        with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=2):
                with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
                    result = determine_next_action(state)
        assert result == "fallback"

    def test_fail_all_fallbacks_exhausted_returns_warning(self) -> None:
        state = {
            **PASS_STATE,
            "review_result": {**PASS_STATE["review_result"], "verdict": "FAIL"},
            "retry_count": 5,
            "active_model": "cohere/command-r-plus",
        }
        with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=2):
                with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
                    result = determine_next_action(state)
        assert result == "warning"


class TestLogDraftToProfile:
    def test_entry_appended_to_prior_drafts(self) -> None:
        profile: dict = {**SAMPLE_PROFILE, "prior_drafts": []}
        updated = log_draft_to_profile(profile, PASS_STATE)
        assert len(updated["prior_drafts"]) == 1
        entry = updated["prior_drafts"][0]
        assert entry["intent"] == "apology"
        assert entry["recipient_type"] == "client"

    def test_original_profile_not_mutated(self) -> None:
        profile: dict = {**SAMPLE_PROFILE, "prior_drafts": []}
        original_len = len(profile["prior_drafts"])
        log_draft_to_profile(profile, PASS_STATE)
        assert len(profile["prior_drafts"]) == original_len

    def test_prior_drafts_capped_at_20(self) -> None:
        existing = [
            {"date": "2026-01-01", "intent": "other", "recipient_type": "client",
             "draft_summary": f"Draft {i}", "subject_hint": "test"}
            for i in range(20)
        ]
        profile: dict = {**SAMPLE_PROFILE, "prior_drafts": existing}
        updated = log_draft_to_profile(profile, PASS_STATE)
        assert len(updated["prior_drafts"]) == 20

    def test_draft_summary_truncated_at_200_chars(self) -> None:
        long_body = "x" * 500
        state = {
            **PASS_STATE,
            "personalized_draft": {
                **SAMPLE_DRAFT,
                "body_paragraphs": [long_body, long_body],
            },
        }
        profile: dict = {**SAMPLE_PROFILE, "prior_drafts": []}
        updated = log_draft_to_profile(profile, state)
        summary = updated["prior_drafts"][0]["draft_summary"]
        assert len(summary) <= 203


class TestUpdateStyleNotes:
    def test_observations_appended(self) -> None:
        profile: dict = {**SAMPLE_PROFILE, "writing_style_notes": "Existing note."}
        updated = update_style_notes(profile, ["User prefers short sentences."])
        assert "Existing note." in updated["writing_style_notes"]
        assert "User prefers short sentences." in updated["writing_style_notes"]

    def test_empty_observations_no_change(self) -> None:
        profile: dict = {**SAMPLE_PROFILE, "writing_style_notes": "Existing note."}
        updated = update_style_notes(profile, [])
        assert updated["writing_style_notes"] == "Existing note."


class TestRouterMemoryAgent:
    def test_success_path_returns_final_draft(self) -> None:
        with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=2):
                with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
                    with patch("src.agents.router_agent.save_user_profile"):
                        result = router_memory_agent(PASS_STATE)

        assert result["pipeline_status"] == "success"
        assert result["final_draft"] is not None
        assert result["pipeline_warning"] is None

    def test_retry_path_increments_retry_count(self) -> None:
        state = {
            **PASS_STATE,
            "review_result": {**PASS_STATE["review_result"], "verdict": "FAIL"},
            "retry_count": 0,
        }
        with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=2):
                with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
                    result = router_memory_agent(state)

        assert result["pipeline_status"] == "retrying"
        assert result["retry_count"] == 1

    def test_fallback_path_switches_model(self) -> None:
        state = {
            **PASS_STATE,
            "review_result": {**PASS_STATE["review_result"], "verdict": "FAIL"},
            "retry_count": 2,
            "active_model": "gpt-4o",
        }
        with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=2):
                with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
                    result = router_memory_agent(state)

        assert result["pipeline_status"] == "fallback"
        assert result["active_model"] == "claude-3-5-sonnet-20241022"

    def test_warning_path_sets_pipeline_warning(self) -> None:
        state = {
            **PASS_STATE,
            "review_result": {**PASS_STATE["review_result"], "verdict": "FAIL"},
            "retry_count": 5,
            "active_model": "cohere/command-r-plus",
        }
        with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=2):
                with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
                    result = router_memory_agent(state)

        assert result["pipeline_status"] == "warning"
        assert result["pipeline_warning"] is not None
        assert result["final_draft"] is not None

    def test_profile_save_failure_does_not_crash(self) -> None:
        with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=2):
                with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
                    with patch(
                        "src.agents.router_agent.save_user_profile",
                        side_effect=OSError("Disk full"),
                    ):
                        result = router_memory_agent(PASS_STATE)

        assert result["pipeline_status"] == "success"

    def test_user_edit_triggers_style_extraction(self) -> None:
        state = {**PASS_STATE, "user_edited_draft": "Subject: Apologies for the Delay\n\nDear Sarah,\nSorry."}
        with patch("src.agents.router_agent.get_mcp_config", return_value=MOCK_MCP_CONFIG):
            with patch("src.agents.router_agent.get_fallback_on_retry", return_value=2):
                with patch("src.agents.router_agent.get_fallback_chain", return_value=MOCK_MCP_CONFIG["fallback_chain"]):
                    with patch("src.agents.router_agent.save_user_profile"):
                        with patch(
                            "src.agents.router_agent.extract_style_from_edits",
                            return_value=["User prefers shorter emails."],
                        ) as mock_extract:
                            router_memory_agent(state)

        mock_extract.assert_called_once()


class TestExtractStyleFromEdits:
    def test_identical_texts_return_empty(self) -> None:
        result = extract_style_from_edits("Hello world.", "Hello world.")
        assert result == []

    def test_empty_original_returns_empty(self) -> None:
        result = extract_style_from_edits("", "Some edited text.")
        assert result == []

    def test_llm_failure_returns_empty(self) -> None:
        with patch("src.agents.router_agent._build_style_diff_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = RuntimeError("LLM error")
            mock_build.return_value = mock_chain

            result = extract_style_from_edits("Original text here.", "Edited text here, shorter.")

        assert result == []

    def test_successful_extraction_returns_list(self) -> None:
        with patch("src.agents.router_agent._build_style_diff_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = ["User prefers shorter sentences."]
            mock_build.return_value = mock_chain

            result = extract_style_from_edits("Original long text here.", "Short text.")

        assert result == ["User prefers shorter sentences."]
