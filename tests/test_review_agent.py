from unittest.mock import MagicMock, patch

from src.agents.review_agent import (
    PASS_THRESHOLDS,
    deduplicate_issues,
    get_thresholds,
    review_validator_agent,
    validate_structure,
)

VALID_DRAFT = {
    "subject_line": "Sincere Apologies for the Delay",
    "salutation": "Dear Sarah,",
    "body_paragraphs": [
        "We sincerely apologize for the delay in your shipment.",
        "The delay was due to an unexpected logistics issue.",
    ],
    "closing": "Please do not hesitate to contact us.",
    "sign_off": "Best regards,",
}

PASS_REVIEW = {
    "grammar_score": 0.95,
    "tone_alignment_score": 0.90,
    "coherence_score": 0.92,
    "structure_complete": True,
    "issues": [],
    "verdict": "PASS",
}

FAIL_REVIEW_TONE = {
    "grammar_score": 0.92,
    "tone_alignment_score": 0.50,
    "coherence_score": 0.88,
    "structure_complete": True,
    "issues": ["Draft uses contractions, violating the formal tone."],
    "verdict": "FAIL",
}

BASE_STATE = {
    "personalized_draft": VALID_DRAFT,
    "tone": "formal",
    "intent": "apology",
    "parsed_context": {"subject_hint": "delayed shipment", "constraints": ["apologetic"]},
    "retry_count": 0,
    "raw_input": "Write an apology email",
}


class TestValidateStructure:
    def test_complete_draft_passes(self) -> None:
        ok, issues = validate_structure(VALID_DRAFT)
        assert ok is True
        assert issues == []

    def test_missing_subject_line_fails(self) -> None:
        draft = {**VALID_DRAFT, "subject_line": ""}
        ok, issues = validate_structure(draft)
        assert ok is False
        assert any("subject" in i.lower() for i in issues)

    def test_missing_salutation_fails(self) -> None:
        draft = {**VALID_DRAFT, "salutation": ""}
        ok, issues = validate_structure(draft)
        assert ok is False
        assert any("salutation" in i.lower() for i in issues)

    def test_single_paragraph_fails(self) -> None:
        draft = {**VALID_DRAFT, "body_paragraphs": ["Only one paragraph."]}
        ok, issues = validate_structure(draft)
        assert ok is False
        assert any("paragraph" in i.lower() for i in issues)

    def test_empty_body_fails(self) -> None:
        draft = {**VALID_DRAFT, "body_paragraphs": []}
        ok, issues = validate_structure(draft)
        assert ok is False

    def test_missing_closing_fails(self) -> None:
        draft = {**VALID_DRAFT, "closing": ""}
        ok, issues = validate_structure(draft)
        assert ok is False
        assert any("closing" in i.lower() for i in issues)

    def test_missing_sign_off_fails(self) -> None:
        draft = {**VALID_DRAFT, "sign_off": ""}
        ok, issues = validate_structure(draft)
        assert ok is False
        assert any("sign" in i.lower() for i in issues)


class TestGetThresholds:
    def test_retry_count_zero_uses_base_thresholds(self) -> None:
        thresholds = get_thresholds(0)
        assert thresholds["tone_alignment_score"] == PASS_THRESHOLDS["tone_alignment_score"]
        assert thresholds["coherence_score"] == PASS_THRESHOLDS["coherence_score"]

    def test_retry_count_one_lowers_thresholds(self) -> None:
        thresholds_0 = get_thresholds(0)
        thresholds_1 = get_thresholds(1)
        assert thresholds_1["tone_alignment_score"] < thresholds_0["tone_alignment_score"]
        assert thresholds_1["coherence_score"] < thresholds_0["coherence_score"]

    def test_thresholds_do_not_go_below_floor(self) -> None:
        thresholds = get_thresholds(100)
        assert thresholds["tone_alignment_score"] >= 0.60
        assert thresholds["coherence_score"] >= 0.65


class TestDeduplicateIssues:
    def test_new_issues_all_returned(self) -> None:
        issues = ["Fix grammar.", "Add CTA."]
        result = deduplicate_issues(issues, {})
        assert result == issues

    def test_prior_issues_filtered_out(self) -> None:
        issues = ["Fix grammar.", "Add CTA."]
        state = {"prior_review_issues": ["Fix grammar."]}
        result = deduplicate_issues(issues, state)
        assert "Fix grammar." not in result
        assert "Add CTA." in result

    def test_all_repeats_escalated(self) -> None:
        issues = ["Fix grammar."]
        state = {"prior_review_issues": ["Fix grammar."]}
        result = deduplicate_issues(issues, state)
        assert len(result) > 0
        assert any("[Repeat issue" in i for i in result)


class TestReviewValidatorAgent:
    def test_pass_verdict_returned_correctly(self) -> None:
        with patch("src.agents.review_agent._build_review_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = dict(PASS_REVIEW)
            mock_build.return_value = mock_chain

            result = review_validator_agent(BASE_STATE)

        assert result["review_result"]["verdict"] == "PASS"
        assert result["retry_issues"] == []

    def test_fail_on_low_tone_score(self) -> None:
        with patch("src.agents.review_agent._build_review_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = dict(FAIL_REVIEW_TONE)
            mock_build.return_value = mock_chain

            result = review_validator_agent(BASE_STATE)

        assert result["review_result"]["verdict"] == "FAIL"
        assert len(result["retry_issues"]) > 0

    def test_structural_fast_fail_skips_llm(self) -> None:
        broken_draft = {
            "subject_line": "",
            "salutation": "",
            "body_paragraphs": [],
            "closing": "",
            "sign_off": "",
        }
        state = {**BASE_STATE, "personalized_draft": broken_draft}

        with patch("src.agents.review_agent._build_review_chain") as mock_build:
            result = review_validator_agent(state)
            mock_build.assert_not_called()

        assert result["review_result"]["verdict"] == "FAIL"

    def test_no_draft_returns_fail(self) -> None:
        state = {**BASE_STATE, "personalized_draft": None, "draft": None}
        result = review_validator_agent(state)
        assert result["review_result"]["verdict"] == "FAIL"
        assert "No draft" in str(result["retry_issues"])

    def test_llm_crash_fails_open(self) -> None:
        with patch("src.agents.review_agent._build_review_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = RuntimeError("LLM unavailable")
            mock_build.return_value = mock_chain

            result = review_validator_agent(BASE_STATE)

        assert result["review_result"]["verdict"] == "PASS"
        assert result["retry_issues"] == []

    def test_retry_count_relaxes_tone_threshold(self) -> None:
        borderline_review = {
            **PASS_REVIEW,
            "tone_alignment_score": 0.65,
            "verdict": "PASS",
        }
        state_retry = {**BASE_STATE, "retry_count": 2}
        with patch("src.agents.review_agent._build_review_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = borderline_review
            mock_build.return_value = mock_chain

            result = review_validator_agent(state_retry)

        assert result["review_result"]["verdict"] == "PASS"

    def test_low_grammar_score_causes_fail(self) -> None:
        low_grammar = {**PASS_REVIEW, "grammar_score": 0.50, "verdict": "PASS"}
        with patch("src.agents.review_agent._build_review_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = low_grammar
            mock_build.return_value = mock_chain

            result = review_validator_agent(BASE_STATE)

        assert result["review_result"]["verdict"] == "FAIL"
        assert any("grammar" in i.lower() for i in result["retry_issues"])

    def test_prior_review_issues_tracked_on_fail(self) -> None:
        with patch("src.agents.review_agent._build_review_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = dict(FAIL_REVIEW_TONE)
            mock_build.return_value = mock_chain

            result = review_validator_agent(BASE_STATE)

        assert "prior_review_issues" in result
        assert len(result["prior_review_issues"]) > 0
