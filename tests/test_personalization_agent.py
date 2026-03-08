from copy import deepcopy
from unittest.mock import MagicMock, patch

from src.agents.personalization_agent import (
    apply_slot_filling,
    find_prior_context,
    personalization_agent,
)

SAMPLE_DRAFT = {
    "subject_line": "Apology for Delay",
    "salutation": "Dear [Name],",
    "body_paragraphs": [
        "We at [Company] sincerely apologize for the delay.",
        "We are taking immediate corrective action.",
    ],
    "closing": "Please contact us if you have questions.",
    "sign_off": "Sincerely,",
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
    "known_contacts": {"client": "Sarah Chen"},
}

SAMPLE_CONTEXT = {
    "recipient_type": "client",
    "recipient_name": "Sarah",
    "subject_hint": "delayed shipment",
    "constraints": ["apologetic"],
    "urgency": False,
}


class TestApplySlotFilling:
    def test_recipient_name_injected_into_salutation(self) -> None:
        draft = deepcopy(SAMPLE_DRAFT)
        ctx = {**SAMPLE_CONTEXT}
        result, log = apply_slot_filling(draft, ctx, {})
        assert "Sarah" in result["salutation"]
        assert "[Name]" not in result["salutation"]

    def test_no_recipient_name_leaves_placeholder(self) -> None:
        draft = deepcopy(SAMPLE_DRAFT)
        ctx = {**SAMPLE_CONTEXT, "recipient_name": None}
        result, log = apply_slot_filling(draft, ctx, {})
        assert "[Name]" in result["salutation"]

    def test_sign_off_preference_applied(self) -> None:
        draft = deepcopy(SAMPLE_DRAFT)
        result, log = apply_slot_filling(draft, SAMPLE_CONTEXT, SAMPLE_PROFILE)
        assert "Best regards" in result["sign_off"]
        assert any("sign-off" in entry.lower() for entry in log)

    def test_company_placeholder_replaced(self) -> None:
        draft = deepcopy(SAMPLE_DRAFT)
        result, log = apply_slot_filling(draft, SAMPLE_CONTEXT, SAMPLE_PROFILE)
        body = " ".join(result["body_paragraphs"])
        assert "Acme Corp" in body
        assert "[Company]" not in body

    def test_name_resolved_from_known_contacts(self) -> None:
        draft = deepcopy(SAMPLE_DRAFT)
        ctx = {**SAMPLE_CONTEXT, "recipient_name": None}
        result, log = apply_slot_filling(draft, ctx, SAMPLE_PROFILE)
        assert "Sarah Chen" in result["salutation"]

    def test_no_profile_no_changes(self) -> None:
        draft = deepcopy(SAMPLE_DRAFT)
        result, log = apply_slot_filling(draft, {}, {})
        assert result["salutation"] == SAMPLE_DRAFT["salutation"]
        assert result["sign_off"] == SAMPLE_DRAFT["sign_off"]

    def test_original_draft_not_mutated(self) -> None:
        draft = deepcopy(SAMPLE_DRAFT)
        original_salutation = draft["salutation"]
        apply_slot_filling(draft, SAMPLE_CONTEXT, SAMPLE_PROFILE)
        assert draft["salutation"] == original_salutation


class TestFindPriorContext:
    def test_same_intent_and_recipient_found(self) -> None:
        profile = {
            "prior_drafts": [
                {
                    "recipient_type": "client",
                    "intent": "apology",
                    "draft_summary": "We apologize for the delay...",
                }
            ]
        }
        result = find_prior_context(profile, "apology", "client")
        assert result is not None
        assert "apologize" in result

    def test_same_recipient_different_intent_fallback(self) -> None:
        profile = {
            "prior_drafts": [
                {
                    "recipient_type": "client",
                    "intent": "follow_up",
                    "draft_summary": "Following up on our meeting...",
                }
            ]
        }
        result = find_prior_context(profile, "apology", "client")
        assert result is not None
        assert "Following up" in result

    def test_different_recipient_type_not_returned(self) -> None:
        profile = {
            "prior_drafts": [
                {
                    "recipient_type": "manager",
                    "intent": "apology",
                    "draft_summary": "I apologize for missing the deadline.",
                }
            ]
        }
        result = find_prior_context(profile, "apology", "client")
        assert result is None

    def test_empty_profile_returns_none(self) -> None:
        assert find_prior_context({}, "apology", "client") is None

    def test_most_recent_match_returned(self) -> None:
        profile = {
            "prior_drafts": [
                {
                    "recipient_type": "client",
                    "intent": "apology",
                    "draft_summary": "Old apology.",
                },
                {
                    "recipient_type": "client",
                    "intent": "apology",
                    "draft_summary": "Recent apology.",
                },
            ]
        }
        result = find_prior_context(profile, "apology", "client")
        assert result == "Recent apology."


class TestPersonalizationAgent:
    def test_no_draft_passes_through(self) -> None:
        state = {"draft": None, "parsed_context": {}, "user_profile": {}, "intent": "other"}
        result = personalization_agent(state)
        assert result["personalized_draft"] is None
        assert result["personalization_log"] == ["No draft to personalize."]

    def test_slot_filling_always_runs(self) -> None:
        state = {
            "draft": deepcopy(SAMPLE_DRAFT),
            "parsed_context": SAMPLE_CONTEXT,
            "user_profile": SAMPLE_PROFILE,
            "intent": "apology",
        }
        result = personalization_agent(state)
        personalized = result["personalized_draft"]
        assert isinstance(personalized, dict)
        assert "Sarah" in personalized["salutation"]

    def test_voice_matching_skipped_when_no_style_notes(self) -> None:
        profile = {**SAMPLE_PROFILE, "writing_style_notes": ""}
        state = {
            "draft": deepcopy(SAMPLE_DRAFT),
            "parsed_context": SAMPLE_CONTEXT,
            "user_profile": profile,
            "intent": "apology",
        }
        with patch(
            "src.agents.personalization_agent._build_voice_match_chain"
        ) as mock_build:
            personalization_agent(state)

        mock_build.assert_not_called()

    def test_voice_matching_runs_with_sufficient_style_notes(self) -> None:
        profile = {
            **SAMPLE_PROFILE,
            "writing_style_notes": "Prefers short sentences and active voice throughout emails.",
        }
        state = {
            "draft": deepcopy(SAMPLE_DRAFT),
            "parsed_context": SAMPLE_CONTEXT,
            "user_profile": profile,
            "intent": "apology",
        }
        refined_draft = {**SAMPLE_DRAFT, "salutation": "Dear Sarah,"}
        with patch(
            "src.agents.personalization_agent._build_voice_match_chain"
        ) as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = refined_draft
            mock_build.return_value = mock_chain

            result = personalization_agent(state)

        mock_build.assert_called_once()
        assert any("voice matching" in entry.lower() for entry in result["personalization_log"])

    def test_voice_matching_failure_returns_original_draft(self) -> None:
        profile = {
            **SAMPLE_PROFILE,
            "writing_style_notes": "Prefers concise emails with active voice preferred.",
        }
        state = {
            "draft": deepcopy(SAMPLE_DRAFT),
            "parsed_context": SAMPLE_CONTEXT,
            "user_profile": profile,
            "intent": "apology",
        }
        with patch(
            "src.agents.personalization_agent._build_voice_match_chain"
        ) as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = RuntimeError("LLM error")
            mock_build.return_value = mock_chain

            result = personalization_agent(state)

        assert result["personalized_draft"] is not None
        assert any("skipped" in entry.lower() for entry in result["personalization_log"])
