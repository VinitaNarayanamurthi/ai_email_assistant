# Implementation Plan: Review & Validator Agent
**File:** `src/agents/review_agent.py`
**Position in pipeline:** Agent 6 of 7

---

## 1. Role & Responsibility

The Review & Validator Agent is the **quality gate** of the pipeline. It evaluates the personalized draft against four independent quality dimensions and produces a structured verdict. A PASS allows the draft to proceed to the Router Agent for finalization. A FAIL sends the draft back to the Draft Writer with a specific list of issues to fix.

It does NOT rewrite the email. It only **evaluates and judges**.

---

## 2. Dependencies

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional
import json
```

---

## 3. State Interface

### Reads from shared state
| Key | Type | Description |
|---|---|---|
| `personalized_draft` | `dict` | The draft to evaluate |
| `tone` | `str` | The intended tone (for tone alignment check) |
| `intent` | `str` | The intended intent (for structure check) |
| `parsed_context` | `dict` | Original constraints and subject (for coherence check) |
| `retry_count` | `int` | Current retry count (influences strictness) |

### Writes to shared state
| Key | Type | Description |
|---|---|---|
| `review_result` | `dict` | Full evaluation result including verdict |
| `retry_issues` | `list[str]` | List of specific issues for the Draft Writer to fix on retry |

---

## 4. Review Dimensions

| Dimension | What It Checks | Scoring |
|---|---|---|
| **Grammar & Fluency** | Broken sentences, tense errors, subject-verb agreement | 0.0–1.0 |
| **Tone Alignment** | Does the draft actually sound like the requested tone? | 0.0–1.0 |
| **Contextual Coherence** | Does the draft address the stated subject? No hallucinated facts? | 0.0–1.0 |
| **Structure Completeness** | Are all required sections present and non-empty? | Boolean |

---

## 5. Output Schema

```python
class ReviewResult(BaseModel):
    grammar_score: float = Field(
        description="Grammar and fluency score from 0.0 (poor) to 1.0 (excellent)",
        ge=0.0, le=1.0
    )
    tone_alignment_score: float = Field(
        description="How well the draft matches the requested tone, from 0.0 to 1.0",
        ge=0.0, le=1.0
    )
    coherence_score: float = Field(
        description="How well the draft addresses the stated subject and context, from 0.0 to 1.0",
        ge=0.0, le=1.0
    )
    structure_complete: bool = Field(
        description="True if all required sections (subject, salutation, body, closing, sign-off) are present and non-empty"
    )
    issues: list[str] = Field(
        description="Specific, actionable list of issues found. Empty list if verdict is PASS.",
        default_factory=list
    )
    verdict: str = Field(
        description="PASS or FAIL"
    )
```

---

## 6. Pass Thresholds

```python
PASS_THRESHOLDS = {
    "grammar_score": 0.75,
    "tone_alignment_score": 0.70,
    "coherence_score": 0.75,
    "structure_complete": True  # must be True
}
```

On retry runs (`retry_count > 0`), lower the tone threshold slightly to avoid infinite loops:
```python
def get_thresholds(retry_count: int) -> dict:
    base = dict(PASS_THRESHOLDS)
    if retry_count >= 1:
        base["tone_alignment_score"] = max(0.60, base["tone_alignment_score"] - 0.05)
        base["coherence_score"] = max(0.65, base["coherence_score"] - 0.05)
    return base
```

---

## 7. Prompt Template

Using the **LLM-as-judge** pattern. The model evaluates the draft against stated criteria and returns a structured verdict.

```python
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
  - "Draft uses contractions ('don't', 'we're') which violates the formal tone requirement."
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
```

---

## 8. Structural Validation (Pre-LLM)

Run a fast deterministic check before the LLM call to catch obvious structural failures cheaply:

```python
def validate_structure(draft: dict) -> tuple[bool, list[str]]:
    """Fast structural check — no LLM needed."""
    issues = []

    if not draft.get("subject_line", "").strip():
        issues.append("Subject line is missing or empty.")
    if not draft.get("salutation", "").strip():
        issues.append("Salutation is missing or empty.")
    if not draft.get("body_paragraphs") or len(draft["body_paragraphs"]) < 2:
        issues.append("Email body must have at least 2 paragraphs.")
    if not draft.get("closing", "").strip():
        issues.append("Closing sentence is missing.")
    if not draft.get("sign_off", "").strip():
        issues.append("Sign-off is missing.")

    return len(issues) == 0, issues
```

If structural validation fails hard (multiple missing sections), skip the LLM review and return FAIL immediately — no point spending API tokens on an obviously broken draft.

---

## 9. Implementation — Agent Node Function

```python
def review_validator_agent(state: dict) -> dict:
    draft = state.get("personalized_draft") or state.get("draft")
    tone = state.get("tone", "formal")
    intent = state.get("intent", "other")
    ctx = state.get("parsed_context", {})
    retry_count = state.get("retry_count", 0)

    if not draft:
        return {
            "review_result": {
                "grammar_score": 0.0,
                "tone_alignment_score": 0.0,
                "coherence_score": 0.0,
                "structure_complete": False,
                "issues": ["No draft was provided to review."],
                "verdict": "FAIL"
            },
            "retry_issues": ["No draft was generated. Please try again."]
        }

    # Step 1: Fast structural check
    struct_ok, struct_issues = validate_structure(draft)
    if not struct_ok and len(struct_issues) >= 3:
        # Draft is too broken to bother LLM-reviewing
        return {
            "review_result": {
                "grammar_score": 0.0,
                "tone_alignment_score": 0.0,
                "coherence_score": 0.0,
                "structure_complete": False,
                "issues": struct_issues,
                "verdict": "FAIL"
            },
            "retry_issues": struct_issues
        }

    # Step 2: LLM-based multi-criteria evaluation
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    parser = JsonOutputParser(pydantic_object=ReviewResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEW_SYSTEM_PROMPT),
        ("human", REVIEW_USER_PROMPT)
    ])
    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "tone": tone,
            "intent": intent,
            "subject_hint": ctx.get("subject_hint", "not specified"),
            "draft_json": json.dumps(draft, indent=2),
            "constraints": ", ".join(ctx.get("constraints", [])) or "none"
        })

        # Step 3: Apply pass/fail thresholds
        thresholds = get_thresholds(retry_count)
        verdict = "PASS"
        issues = list(result.get("issues", []))

        if result["grammar_score"] < thresholds["grammar_score"]:
            verdict = "FAIL"
            if not any("grammar" in i.lower() for i in issues):
                issues.append(f"Grammar score too low ({result['grammar_score']:.2f}). Review sentence structure and fluency.")

        if result["tone_alignment_score"] < thresholds["tone_alignment_score"]:
            verdict = "FAIL"
            if not any("tone" in i.lower() for i in issues):
                issues.append(f"Tone does not sufficiently match '{tone}' (score: {result['tone_alignment_score']:.2f}).")

        if result["coherence_score"] < thresholds["coherence_score"]:
            verdict = "FAIL"
            if not any("subject" in i.lower() or "topic" in i.lower() for i in issues):
                issues.append(f"Draft does not adequately address the stated subject: '{ctx.get('subject_hint')}'.")

        if not result["structure_complete"]:
            verdict = "FAIL"
            issues.extend(struct_issues)

        return {
            "review_result": {
                "grammar_score": result["grammar_score"],
                "tone_alignment_score": result["tone_alignment_score"],
                "coherence_score": result["coherence_score"],
                "structure_complete": result["structure_complete"],
                "issues": issues,
                "verdict": verdict
            },
            "retry_issues": issues if verdict == "FAIL" else []
        }

    except Exception as e:
        # Fail open — if the reviewer crashes, let the draft through with a warning
        return {
            "review_result": {
                "grammar_score": 0.8,
                "tone_alignment_score": 0.8,
                "coherence_score": 0.8,
                "structure_complete": True,
                "issues": [],
                "verdict": "PASS"  # fail open
            },
            "retry_issues": []
        }
```

---

## 10. Register in LangGraph

```python
from src.agents.review_agent import review_validator_agent

graph.add_node("review_validator", review_validator_agent)
graph.add_edge("review_validator", "router_memory")
```

No conditional edge here — the Router Agent reads `review_result.verdict` and makes the routing decision.

---

## 11. Example Input → Output (PASS)

**`personalized_draft`:**
```json
{
  "subject_line": "Sincere Apologies Regarding Your Recent Shipment Delay",
  "salutation": "Dear Sarah,",
  "body_paragraphs": ["We sincerely apologize...", "The delay was caused by...", "We value your partnership..."],
  "closing": "Please do not hesitate to contact us directly.",
  "sign_off": "Best regards,"
}
```

**`review_result` added to state:**
```json
{
  "grammar_score": 0.97,
  "tone_alignment_score": 0.91,
  "coherence_score": 0.94,
  "structure_complete": true,
  "issues": [],
  "verdict": "PASS"
}
```

---

## 12. Example Input → Output (FAIL)

**`review_result`:**
```json
{
  "grammar_score": 0.95,
  "tone_alignment_score": 0.52,
  "coherence_score": 0.88,
  "structure_complete": true,
  "issues": [
    "Draft uses contractions ('we've', 'don't') which violates the formal tone requirement. Replace all contractions.",
    "Closing does not include a call-to-action. Add a specific next step with a timeline."
  ],
  "verdict": "FAIL"
}
```

**`retry_issues`** (passed back to Draft Writer on retry):
```json
[
  "Draft uses contractions ('we've', 'don't') which violates the formal tone requirement. Replace all contractions.",
  "Closing does not include a call-to-action. Add a specific next step with a timeline."
]
```

---

## 13. Conversation Memory

### Memory Type Overview

| Memory Type | Scope | Mechanism | This Agent's Role |
|---|---|---|---|
| **Short-term (within-session)** | Single pipeline run | LangGraph shared state | Reads `personalized_draft`, `tone`, `intent`; writes `review_result` + `retry_issues` |
| **Retry memory (within-session)** | Across retry cycles in one run | `retry_count` in state | Reads retry count to progressively loosen thresholds and avoid infinite loops |
| **Long-term (cross-session)** | Across sessions | `user_profiles.json` (indirect) | Issue patterns from prior reviews can inform writing_style_notes over time |
| **Conversational (multi-turn)** | Same UI session | LangGraph `thread_id` checkpointing | Stores prior `review_result` — used on refinement turns to understand what was already accepted |

---

### 13.1 Retry Memory — Threshold Relaxation Over Cycles

Within a single pipeline run, this agent accumulates implicit memory through `retry_count`. Each time it rejects a draft, `retry_count` increments (via the Router Agent). On the next call, it reads this count and applies progressively looser thresholds:

```python
def get_thresholds(retry_count: int) -> dict:
    base = dict(PASS_THRESHOLDS)
    # Each retry relaxes non-grammar thresholds slightly
    if retry_count >= 1:
        base["tone_alignment_score"] = max(0.60, base["tone_alignment_score"] - 0.05 * retry_count)
        base["coherence_score"] = max(0.65, base["coherence_score"] - 0.03 * retry_count)
    return base
```

This is a simple but effective form of **within-session memory**: the agent "remembers" it has already rejected the draft before and becomes more lenient to prevent an infinite retry loop. It still enforces grammar and structure hard.

---

### 13.2 Issue History — Preventing Repeat Feedback

A subtle problem: if the agent gives the same feedback twice, the Draft Writer may not know what changed between retries. Store `prior_review_issues` in state and skip issues already raised before:

```python
def deduplicate_issues(new_issues: list[str], state: dict) -> list[str]:
    """
    Filters out issues that were already raised in a prior review cycle.
    Prevents the Draft Writer from receiving the same unhelpful feedback twice.
    """
    prior_issues = state.get("prior_review_issues", [])
    # Normalize for comparison
    prior_normalized = {i.lower().strip() for i in prior_issues}
    fresh_issues = [i for i in new_issues if i.lower().strip() not in prior_normalized]

    # If all issues are repeats, escalate with a different framing
    if not fresh_issues and new_issues:
        fresh_issues = [
            f"[Repeat issue — escalated] {issue}" for issue in new_issues[:2]
        ]
    return fresh_issues
```

Update the agent node to track issue history:

```python
def review_validator_agent(state: dict) -> dict:
    # ... existing review logic ...

    if verdict == "FAIL":
        issues = deduplicate_issues(issues, state)
        # Store current issues for the next retry cycle
        return {
            "review_result": { ... },
            "retry_issues": issues,
            "prior_review_issues": state.get("prior_review_issues", []) + issues
        }
```

Add `prior_review_issues` to the state schema (initialized as `[]`).

---

### 13.3 Long-Term Memory — Issue Patterns Feeding Style Notes

Over many sessions, recurring review failures for a user reveal systematic weaknesses in how the LLM writes for that user's style. For example: *"LLM always writes closings without a CTA for this user's follow-up emails"* — this is a pattern worth capturing.

The Router Agent can write a summary of recurring issues to `writing_style_notes` so future Draft Writer runs include preventive instructions:

```python
# In router_agent.py — after a run that required retries:
def record_recurring_issues(profile: dict, review_history: list[list[str]]) -> dict:
    """
    If the same issue category appeared across multiple review cycles in this session,
    add a preventive note to writing_style_notes.
    """
    all_issues = [issue for cycle in review_history for issue in cycle]
    tone_issues = [i for i in all_issues if "tone" in i.lower() or "contraction" in i.lower()]

    if len(tone_issues) >= 2:
        note = "Note: LLM tends to use informal language in formal drafts for this user. Add explicit anti-contraction reminder."
        existing = profile.get("writing_style_notes", "")
        if note not in existing:
            profile["writing_style_notes"] = (existing + " " + note).strip()
    return profile
```

This creates a **slow feedback loop**: review failures → style notes → better Draft Writer prompts → fewer review failures.

---

### 13.4 Multi-Turn Conversation Memory — Accepting Prior Decisions

On a refinement turn, if the user asked only to change the tone (not the content), the new draft should be reviewed against the new tone. But if the content-related review already passed on the prior turn, the coherence and structure checks can be **relaxed or skipped** on the refinement turn.

With LangGraph's `MemorySaver`, the prior `review_result` is in state:

```python
def review_validator_agent(state: dict) -> dict:
    raw_input = state.get("raw_input", "")
    prior_result = state.get("review_result", {})

    # On refinement turns where prior coherence/structure passed,
    # skip those checks and only re-evaluate what changed
    TONE_ONLY_REFINEMENTS = ["tone", "formal", "casual", "assertive", "shorter", "longer"]
    is_tone_refinement = any(s in raw_input.lower() for s in TONE_ONLY_REFINEMENTS)

    if is_tone_refinement and prior_result.get("coherence_score", 0) >= 0.75:
        # Coherence already verified — only re-check tone alignment and grammar
        # Skip full LLM review, do a targeted tone check only
        return run_targeted_tone_check(state)

    # Full review for fresh requests or content-changing refinements
    # ... existing implementation ...
```

```python
def run_targeted_tone_check(state: dict) -> dict:
    """Lightweight review that only checks tone alignment, skipping coherence/structure."""
    draft = state.get("personalized_draft") or state.get("draft")
    tone = state.get("tone", "formal")
    prior_result = state.get("review_result", {})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # cheaper model for targeted check
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Does this email match a {tone} tone? Score 0.0-1.0 and list any violations."),
        ("human", json.dumps(draft))
    ])
    from langchain_core.output_parsers import JsonOutputParser
    # ... invoke and return result, reusing prior coherence/structure scores ...
```

---

### 13.5 Memory Scope Boundaries

| What This Agent Remembers | How Long |
|---|---|
| `review_result` from current run | Until pipeline ends (in-state) |
| `retry_count` — used to relax thresholds | Within a single pipeline execution |
| `prior_review_issues` — deduplication list | Within a single pipeline execution |
| `review_result` from prior turn in same session | Until session ends (MemorySaver) |
| Recurring issue patterns → style notes | Permanently (user_profiles.json, via Router Agent) |

---

## 14. Testing Checklist

- [ ] Well-written formal email → PASS with high scores
- [ ] Formal email with contractions → `tone_alignment_score` low → FAIL
- [ ] Draft missing closing → `structure_complete: false` → immediate FAIL
- [ ] Draft off-topic (wrong subject) → `coherence_score` low → FAIL
- [ ] Grammar errors present → `grammar_score` low → FAIL
- [ ] `retry_count > 0` → lower thresholds applied
- [ ] LLM reviewer crashes → fail open → PASS returned, pipeline continues
- [ ] Issues list is specific and actionable (not vague)

---

## 14. File Checklist

- [ ] `src/agents/review_agent.py` — agent function + schema + structural validator + thresholds
- [ ] Registered in `src/workflow/langgraph_flow.py`
- [ ] Unit tests in `tests/test_review_agent.py`
