from unittest.mock import MagicMock, patch

from src.agents.intent_detection_agent import (
    ALLOWED_INTENTS,
    CONFIDENCE_THRESHOLD,
    format_prior_hint,
    get_intent_prior,
    intent_detection_agent,
)

SAMPLE_CONTEXT = {
    "recipient_type": "client",
    "subject_hint": "delayed shipment",
    "constraints": ["apologetic"],
    "context_notes": None,
}


def _make_state(intent: str | None = None, raw_input: str = "") -> dict:
    state: dict = {
        "raw_input": raw_input,
        "parsed_context": SAMPLE_CONTEXT,
    }
    if intent:
        state["intent"] = intent
    return state


class TestGetIntentPrior:
    def test_empty_prior_drafts_returns_empty(self) -> None:
        result = get_intent_prior({}, "client")
        assert result == {}

    def test_single_draft_correct_distribution(self) -> None:
        profile = {
            "prior_drafts": [
                {"recipient_type": "client", "intent": "apology"},
                {"recipient_type": "client", "intent": "apology"},
                {"recipient_type": "client", "intent": "follow_up"},
            ]
        }
        result = get_intent_prior(profile, "client")
        assert abs(result["apology"] - 2 / 3) < 0.01
        assert abs(result["follow_up"] - 1 / 3) < 0.01

    def test_different_recipient_type_ignored(self) -> None:
        profile = {
            "prior_drafts": [{"recipient_type": "manager", "intent": "request"}]
        }
        result = get_intent_prior(profile, "client")
        assert result == {}


class TestFormatPriorHint:
    def test_empty_prior_returns_empty_string(self) -> None:
        assert format_prior_hint({}) == ""

    def test_hint_includes_top_intents(self) -> None:
        prior = {"apology": 0.7, "follow_up": 0.3}
        hint = format_prior_hint(prior)
        assert "apology" in hint
        assert "70%" in hint


class TestIntentDetectionAgent:
    def test_clear_apology_classified_correctly(self) -> None:
        with patch("src.agents.intent_detection_agent.build_intent_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "intent": "apology",
                "confidence": 0.95,
                "reasoning": "Subject indicates an apology for a delay.",
            }
            mock_build.return_value = mock_chain

            result = intent_detection_agent(_make_state())

        assert result["intent"] == "apology"
        assert result["intent_fallback"] is False
        assert result["intent_confidence"] == 0.95

    def test_low_confidence_falls_back_to_other(self) -> None:
        with patch("src.agents.intent_detection_agent.build_intent_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "intent": "outreach",
                "confidence": 0.50,
                "reasoning": "Ambiguous context.",
            }
            mock_build.return_value = mock_chain

            result = intent_detection_agent(_make_state())

        assert result["intent"] == "other"
        assert result["intent_fallback"] is True

    def test_invalid_intent_label_remapped_to_other(self) -> None:
        with patch("src.agents.intent_detection_agent.build_intent_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "intent": "unknown_label",
                "confidence": 0.90,
                "reasoning": "Some reasoning.",
            }
            mock_build.return_value = mock_chain

            result = intent_detection_agent(_make_state())

        assert result["intent"] == "other"
        assert result["intent_fallback"] is True

    def test_llm_failure_returns_other_with_zero_confidence(self) -> None:
        with patch("src.agents.intent_detection_agent.build_intent_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = RuntimeError("API error")
            mock_build.return_value = mock_chain

            result = intent_detection_agent(_make_state())

        assert result["intent"] == "other"
        assert result["intent_confidence"] == 0.0
        assert result["intent_fallback"] is True

    def test_refinement_preserves_prior_intent(self) -> None:
        state = _make_state(intent="apology", raw_input="make it shorter")
        result = intent_detection_agent(state)

        assert result["intent"] == "apology"
        assert result["intent_confidence"] == 1.0
        assert result["intent_fallback"] is False

    def test_all_valid_intents_accepted(self) -> None:
        for intent_label in ALLOWED_INTENTS - {"other"}:
            with patch(
                "src.agents.intent_detection_agent.build_intent_chain"
            ) as mock_build:
                mock_chain = MagicMock()
                mock_chain.invoke.return_value = {
                    "intent": intent_label,
                    "confidence": 0.90,
                    "reasoning": "Clear signal.",
                }
                mock_build.return_value = mock_chain

                result = intent_detection_agent(_make_state())

            assert result["intent"] == intent_label
            assert result["intent_fallback"] is False

    def test_confidence_at_threshold_is_accepted(self) -> None:
        with patch("src.agents.intent_detection_agent.build_intent_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "intent": "request",
                "confidence": CONFIDENCE_THRESHOLD,
                "reasoning": "At threshold.",
            }
            mock_build.return_value = mock_chain

            result = intent_detection_agent(_make_state())

        assert result["intent"] == "request"
        assert result["intent_fallback"] is False

    def test_confidence_just_below_threshold_falls_back(self) -> None:
        with patch("src.agents.intent_detection_agent.build_intent_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "intent": "request",
                "confidence": CONFIDENCE_THRESHOLD - 0.01,
                "reasoning": "Below threshold.",
            }
            mock_build.return_value = mock_chain

            result = intent_detection_agent(_make_state())

        assert result["intent"] == "other"
        assert result["intent_fallback"] is True
