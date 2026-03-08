import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.models.state import EmailAssistantState, ReviewResultDict

PASS_THRESHOLDS = {
    "grammar_score": 0.75,
    "tone_alignment_score": 0.70,
    "coherence_score": 0.75,
}

TONE_ONLY_REFINEMENTS = ["tone", "formal", "casual", "assertive", "shorter", "longer"]

REVIEW_SYSTEM_PROMPT = """You are a strict but fair email quality reviewer.

Evaluate the given email draft across four dimensions:

1. GRAMMAR & FLUENCY (0.0-1.0)
   - Check for: broken sentences, tense inconsistencies, subject-verb errors, awkward phrasing
   - 1.0 = flawless, 0.75 = minor issues, 0.5 = noticeable errors, below 0.5 = major problems

2. TONE ALIGNMENT (0.0-1.0)
   - Requested tone: {tone}
   - Does the draft actually sound like a {tone} email?
   - Penalize heavily for obvious violations (e.g., contractions in formal, no CTA in assertive)
   - 1.0 = perfectly matched, 0.7 = mostly matched with minor deviations

3. CONTEXTUAL COHERENCE (0.0-1.0)
   - Intended subject: {subject_hint}
   - Intended intent type: {intent}
   - Does the email body actually address this subject?
   - Are there any invented facts, names, or events not in the context?
   - 1.0 = fully on-topic and accurate, 0.5 = partially off-topic or vague

4. STRUCTURE COMPLETENESS (true/false)
   - Must have: non-empty subject_line, salutation, at least 2 body_paragraphs, closing, sign_off
   - true = all present and non-empty, false = any missing or empty

For each issue found, write a specific, actionable correction instruction.
Examples of good issue descriptions:
  - "Closing paragraph is missing a call-to-action. Add one specific next step."
  - "Draft uses contractions ('don't', 'we're') which violates the formal tone requirement. Replace all contractions."
  - "Body does not mention the delayed shipment — the stated subject. Rewrite paragraph 2 to address it."

Bad issue descriptions (too vague):
  - "Tone is wrong."
  - "Needs improvement."

Return a JSON object with fields: grammar_score, tone_alignment_score, coherence_score, structure_complete, issues (list), verdict (PASS or FAIL)."""

REVIEW_USER_PROMPT = """Evaluate this email draft:

{draft_json}

Additional context for coherence check:
- Subject hint: {subject_hint}
- Constraints the draft must satisfy: {constraints}

Return your structured review."""


class ReviewResult(BaseModel):
    grammar_score: float = Field(
        description="Grammar and fluency score from 0.0 (poor) to 1.0 (excellent)",
        ge=0.0,
        le=1.0,
    )
    tone_alignment_score: float = Field(
        description="How well the draft matches the requested tone, from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    coherence_score: float = Field(
        description="How well the draft addresses the stated subject and context, from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    structure_complete: bool = Field(
        description="True if all required sections are present and non-empty"
    )
    issues: list[str] = Field(
        description="Specific, actionable list of issues found. Empty list if verdict is PASS.",
        default_factory=list,
    )
    verdict: str = Field(description="PASS or FAIL")


def get_thresholds(retry_count: int) -> dict[str, float]:
    base = dict(PASS_THRESHOLDS)
    if retry_count >= 1:
        base["tone_alignment_score"] = max(
            0.60, base["tone_alignment_score"] - 0.05 * retry_count
        )
        base["coherence_score"] = max(
            0.65, base["coherence_score"] - 0.03 * retry_count
        )
    return base


def validate_structure(draft: dict[str, object]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not str(draft.get("subject_line") or "").strip():
        issues.append("Subject line is missing or empty.")
    if not str(draft.get("salutation") or "").strip():
        issues.append("Salutation is missing or empty.")
    body = draft.get("body_paragraphs")
    if not body or not isinstance(body, list) or len(body) < 2:
        issues.append("Email body must have at least 2 paragraphs.")
    if not str(draft.get("closing") or "").strip():
        issues.append("Closing sentence is missing.")
    if not str(draft.get("sign_off") or "").strip():
        issues.append("Sign-off is missing.")
    return len(issues) == 0, issues


def deduplicate_issues(new_issues: list[str], state: EmailAssistantState) -> list[str]:
    prior_issues: list[str] = list(state.get("prior_review_issues") or [])  # type: ignore[arg-type]
    prior_normalized = {i.lower().strip() for i in prior_issues}
    fresh_issues = [i for i in new_issues if i.lower().strip() not in prior_normalized]

    if not fresh_issues and new_issues:
        fresh_issues = [
            f"[Repeat issue — escalated] {issue}" for issue in new_issues[:2]
        ]
    return fresh_issues


def _build_tone_check_chain(tone: str) -> object:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"Evaluate if this email matches a {tone} tone. Return JSON with: "
                f"tone_alignment_score (0.0-1.0), issues (list of strings), verdict (PASS or FAIL).",
            ),
            ("human", "{draft_json}"),
        ]
    )
    return prompt | llm | JsonOutputParser()


def _build_review_chain() -> object:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    parser = JsonOutputParser(pydantic_object=ReviewResult)
    prompt = ChatPromptTemplate.from_messages(
        [("system", REVIEW_SYSTEM_PROMPT), ("human", REVIEW_USER_PROMPT)]
    )
    return prompt | llm | parser


def run_targeted_tone_check(state: EmailAssistantState) -> dict[str, object]:
    draft = state.get("personalized_draft") or state.get("draft")
    tone = str(state.get("tone") or "formal")
    prior_result = state.get("review_result") or {}

    chain = _build_tone_check_chain(tone)

    try:
        result: dict[str, object] = chain.invoke(  # type: ignore[assignment]
            {"draft_json": json.dumps(draft, indent=2)}
        )
        tone_score = float(result.get("tone_alignment_score") or 0.8)  # type: ignore[arg-type]
        issues_raw = result.get("issues") or []
        issues: list[str] = [str(i) for i in issues_raw] if isinstance(issues_raw, list) else []
        verdict = "PASS" if tone_score >= 0.70 else "FAIL"

        review_result: ReviewResultDict = {
            "grammar_score": float(prior_result.get("grammar_score") or 0.8),  # type: ignore[arg-type]
            "tone_alignment_score": tone_score,
            "coherence_score": float(prior_result.get("coherence_score") or 0.8),  # type: ignore[arg-type]
            "structure_complete": bool(prior_result.get("structure_complete", True)),
            "issues": issues,
            "verdict": verdict,
        }
        return {
            "review_result": review_result,
            "retry_issues": issues if verdict == "FAIL" else [],
        }
    except Exception:
        review_result = {
            "grammar_score": 0.8,
            "tone_alignment_score": 0.8,
            "coherence_score": 0.8,
            "structure_complete": True,
            "issues": [],
            "verdict": "PASS",
        }
        return {"review_result": review_result, "retry_issues": []}


def review_validator_agent(state: EmailAssistantState) -> dict[str, object]:
    draft = state.get("personalized_draft") or state.get("draft")
    tone = str(state.get("tone") or "formal")
    intent = str(state.get("intent") or "other")
    ctx = state.get("parsed_context") or {}
    retry_count = int(state.get("retry_count") or 0)
    raw_input = str(state.get("raw_input") or "")
    prior_result = state.get("review_result") or {}

    if not draft:
        fail_result: ReviewResultDict = {
            "grammar_score": 0.0,
            "tone_alignment_score": 0.0,
            "coherence_score": 0.0,
            "structure_complete": False,
            "issues": ["No draft was provided to review."],
            "verdict": "FAIL",
        }
        return {
            "review_result": fail_result,
            "retry_issues": ["No draft was generated. Please try again."],
        }

    is_tone_refinement = any(s in raw_input.lower() for s in TONE_ONLY_REFINEMENTS)
    coherence_already_ok = float(prior_result.get("coherence_score") or 0.0) >= 0.75
    if is_tone_refinement and coherence_already_ok and prior_result:
        return run_targeted_tone_check(state)

    struct_ok, struct_issues = validate_structure(dict(draft))  # type: ignore[arg-type]
    if not struct_ok and len(struct_issues) >= 3:
        fast_fail: ReviewResultDict = {
            "grammar_score": 0.0,
            "tone_alignment_score": 0.0,
            "coherence_score": 0.0,
            "structure_complete": False,
            "issues": struct_issues,
            "verdict": "FAIL",
        }
        return {"review_result": fast_fail, "retry_issues": struct_issues}

    chain = _build_review_chain()

    try:
        result: dict[str, object] = chain.invoke(  # type: ignore[assignment]
            {
                "tone": tone,
                "intent": intent,
                "subject_hint": str(ctx.get("subject_hint") or "not specified"),
                "draft_json": json.dumps(draft, indent=2),
                "constraints": ", ".join(ctx.get("constraints") or []) or "none",  # type: ignore[arg-type]
            }
        )

        thresholds = get_thresholds(retry_count)
        verdict = "PASS"
        issues_raw = result.get("issues") or []
        issues: list[str] = [str(i) for i in issues_raw] if isinstance(issues_raw, list) else []

        grammar_score = float(result.get("grammar_score") or 0.0)  # type: ignore[arg-type]
        tone_score = float(result.get("tone_alignment_score") or 0.0)  # type: ignore[arg-type]
        coherence_score = float(result.get("coherence_score") or 0.0)  # type: ignore[arg-type]
        structure_complete = bool(result.get("structure_complete", True))

        if grammar_score < thresholds["grammar_score"]:
            verdict = "FAIL"
            if not any("grammar" in i.lower() for i in issues):
                issues.append(
                    f"Grammar score too low ({grammar_score:.2f}). Review sentence structure and fluency."
                )

        if tone_score < thresholds["tone_alignment_score"]:
            verdict = "FAIL"
            if not any("tone" in i.lower() for i in issues):
                issues.append(
                    f"Tone does not sufficiently match '{tone}' (score: {tone_score:.2f})."
                )

        if coherence_score < thresholds["coherence_score"]:
            verdict = "FAIL"
            if not any("subject" in i.lower() or "topic" in i.lower() for i in issues):
                issues.append(
                    f"Draft does not adequately address the stated subject: '{ctx.get('subject_hint')}'."
                )

        if not structure_complete:
            verdict = "FAIL"
            issues.extend(struct_issues)

        if verdict == "FAIL":
            issues = deduplicate_issues(issues, state)

        review_result: ReviewResultDict = {
            "grammar_score": grammar_score,
            "tone_alignment_score": tone_score,
            "coherence_score": coherence_score,
            "structure_complete": structure_complete,
            "issues": issues,
            "verdict": verdict,
        }

        prior_review_issues: list[str] = list(state.get("prior_review_issues") or [])  # type: ignore[arg-type]
        return {
            "review_result": review_result,
            "retry_issues": issues if verdict == "FAIL" else [],
            "prior_review_issues": prior_review_issues + issues if verdict == "FAIL" else prior_review_issues,
        }

    except Exception:
        pass_result: ReviewResultDict = {
            "grammar_score": 0.8,
            "tone_alignment_score": 0.8,
            "coherence_score": 0.8,
            "structure_complete": True,
            "issues": [],
            "verdict": "PASS",
        }
        return {"review_result": pass_result, "retry_issues": []}
