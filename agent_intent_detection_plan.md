# Implementation Plan: Intent Detection Agent
**File:** `src/agents/intent_detection_agent.py`
**Position in pipeline:** Agent 2 of 7

---

## 1. Role & Responsibility

The Intent Detection Agent reads the structured `parsed_context` produced by the Input Parsing Agent and classifies the email's purpose into one of a fixed set of intent labels. This label is the structural blueprint for the Draft Writer Agent — it determines which email template, rhetorical approach, and structural conventions will be applied.

It does NOT generate text. It only **classifies**.

---

## 2. Dependencies

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
```

---

## 3. State Interface

### Reads from shared state
| Key | Type | Description |
|---|---|---|
| `parsed_context` | `dict` | Structured output from Input Parsing Agent |

### Writes to shared state
| Key | Type | Description |
|---|---|---|
| `intent` | `str` | One of the supported intent labels |
| `intent_confidence` | `float` | Confidence score 0.0–1.0 |
| `intent_fallback` | `bool` | True if low confidence forced a fallback |

---

## 4. Supported Intent Labels

| Label | Description | Structural Signals |
|---|---|---|
| `outreach` | Cold contact, intro, proposal | Hook opener, value proposition, CTA |
| `follow_up` | Checking in after prior contact | Reference prior exchange, next step ask |
| `apology` | Acknowledging mistake or delay | Lead with acknowledgment, not justification |
| `informational` | Update, announcement, status share | Structured facts, no strong CTA |
| `internal_update` | Team/org communication | Direct, no fluff, action items |
| `request` | Asking for something specific | Clear ask, context, deadline |
| `thank_you` | Expressing gratitude | Personal, warm, brief |
| `other` | Catch-all fallback | Generic professional template |

---

## 5. Output Schema

```python
class IntentResult(BaseModel):
    intent: str = Field(
        description=(
            "Exactly one of: outreach, follow_up, apology, informational, "
            "internal_update, request, thank_you, other"
        )
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="One sentence explaining why this intent was chosen"
    )
```

---

## 6. Prompt Template

This agent uses a **classification prompt** — the LLM is constrained to pick from a list, not generate freely.

```python
SYSTEM_PROMPT = """You are an expert email intent classifier.

Your only job is to determine the primary purpose of the email being requested.

Supported intents:
- outreach: Cold contact, introduction, partnership or sales proposal
- follow_up: Checking in after a previous email, meeting, or conversation
- apology: Acknowledging a mistake, delay, or failure
- informational: Sharing an update, announcement, or status report
- internal_update: Communication directed at teammates or within an organization
- request: Asking someone for something specific (a document, a meeting, approval)
- thank_you: Expressing gratitude or appreciation
- other: Does not clearly fit any of the above

Rules:
- Choose the SINGLE most dominant intent
- Do not combine intents
- If uncertain between two, pick the one with higher confidence and note it in reasoning
- Return confidence as a decimal between 0.0 and 1.0
"""

USER_PROMPT = """Based on this parsed email context, classify the intent:

Recipient type: {recipient_type}
Subject hint: {subject_hint}
Constraints: {constraints}
Context notes: {context_notes}

Return a JSON object with fields: intent, confidence, reasoning."""
```

---

## 7. Implementation Steps

### Step 1 — Build the classification chain

```python
def build_intent_chain():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    parser = JsonOutputParser(pydantic_object=IntentResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ])

    return prompt | llm | parser
```

### Step 2 — Define the confidence threshold

```python
CONFIDENCE_THRESHOLD = 0.65  # below this → treat as "other"
```

### Step 3 — Write the agent node function

```python
def intent_detection_agent(state: dict) -> dict:
    ctx = state.get("parsed_context", {})

    chain = build_intent_chain()

    try:
        result = chain.invoke({
            "recipient_type": ctx.get("recipient_type", "unknown"),
            "subject_hint": ctx.get("subject_hint", ""),
            "constraints": ", ".join(ctx.get("constraints", [])) or "none",
            "context_notes": ctx.get("context_notes") or "none"
        })

        intent = result["intent"]
        confidence = result["confidence"]
        fallback_used = False

        # Low confidence fallback
        if confidence < CONFIDENCE_THRESHOLD:
            intent = "other"
            fallback_used = True

        # Validate the intent label is in the allowed set
        allowed = {
            "outreach", "follow_up", "apology", "informational",
            "internal_update", "request", "thank_you", "other"
        }
        if intent not in allowed:
            intent = "other"
            fallback_used = True

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "intent_fallback": fallback_used
        }

    except Exception as e:
        # Hard fallback — default to "other" so the pipeline does not die
        return {
            "intent": "other",
            "intent_confidence": 0.0,
            "intent_fallback": True
        }
```

### Step 4 — Register in LangGraph

```python
from src.agents.intent_detection_agent import intent_detection_agent

graph.add_node("intent_detection", intent_detection_agent)
graph.add_edge("intent_detection", "tone_stylist")
```

---

## 8. Intent → Template Mapping

The `intent` label written to state is consumed by the Draft Writer Agent to select the structural template. Define this mapping in `src/agents/draft_writer_agent.py` or in a shared constants file:

```python
INTENT_TEMPLATES = {
    "outreach": {
        "structure": ["hook", "value_proposition", "social_proof", "cta"],
        "opener_style": "attention-grabbing",
        "cta_required": True
    },
    "follow_up": {
        "structure": ["reference_prior", "update_or_ask", "next_step"],
        "opener_style": "referential",
        "cta_required": True
    },
    "apology": {
        "structure": ["acknowledgment", "explanation_brief", "remedy", "reassurance"],
        "opener_style": "empathetic",
        "cta_required": False
    },
    "informational": {
        "structure": ["context", "key_points", "implications", "optional_cta"],
        "opener_style": "direct",
        "cta_required": False
    },
    "internal_update": {
        "structure": ["summary", "details", "action_items"],
        "opener_style": "direct",
        "cta_required": False
    },
    "request": {
        "structure": ["context", "specific_ask", "rationale", "deadline", "cta"],
        "opener_style": "respectful",
        "cta_required": True
    },
    "thank_you": {
        "structure": ["gratitude", "specific_what", "forward_looking"],
        "opener_style": "warm",
        "cta_required": False
    },
    "other": {
        "structure": ["opener", "body", "closing"],
        "opener_style": "neutral",
        "cta_required": False
    }
}
```

Store this in `src/agents/intent_templates.py` and import it in the Draft Writer Agent.

---

## 9. Example Input → Output

**`parsed_context` input:**
```json
{
  "recipient_type": "client",
  "subject_hint": "delayed shipment",
  "constraints": ["apologetic"],
  "context_notes": null
}
```

**Output added to state:**
```json
{
  "intent": "apology",
  "intent_confidence": 0.96,
  "intent_fallback": false
}
```

---

## 10. Conversation Memory

### Memory Type Overview

| Memory Type | Scope | Mechanism | This Agent's Role |
|---|---|---|---|
| **Short-term (within-session)** | Single pipeline run | LangGraph shared state | Writes `intent` + `intent_confidence` — consumed by Tone Stylist and Draft Writer |
| **Long-term (cross-session)** | Across sessions | `user_profiles.json` | Reads `prior_drafts` to bias classification toward the user's typical email patterns |
| **Conversational (multi-turn)** | Same UI session | LangGraph `thread_id` checkpointing | Reads prior `intent` on refinement turns — avoids reclassification if intent hasn't changed |

---

### 10.1 Short-Term Memory — Within-Session State

The `intent` label written to state acts as **shared short-term memory** for the rest of the pipeline. It drives structural templates in the Draft Writer and modifier selection in the Tone Stylist. All agents downstream treat it as resolved.

On a retry loop (after the Review Agent rejects a draft), this agent does **not re-run** — the intent is already resolved and stored in state. The Router Agent routes back only to the Draft Writer, not all the way back to Intent Detection.

---

### 10.2 Long-Term Memory — Prior Intent Patterns

The user profile's `prior_drafts` list is a cross-session history of past email intent/recipient combinations. This agent can use it to apply a **prior probability bias** to the classifier — making it more likely to choose the correct intent for this user's typical communication patterns.

```python
def get_intent_prior(profile: dict, recipient_type: str) -> dict[str, float]:
    """
    Analyzes prior_drafts to build a probability distribution over intents
    for a given recipient type. Used to bias the classifier on ambiguous inputs.
    """
    prior_drafts = profile.get("prior_drafts", [])
    counts: dict[str, int] = {}

    for draft in prior_drafts:
        if draft.get("recipient_type") == recipient_type:
            intent = draft.get("intent", "other")
            counts[intent] = counts.get(intent, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return {}  # No prior data — don't bias

    return {intent: count / total for intent, count in counts.items()}
```

Inject this prior into the classification prompt as a soft hint:

```python
def format_prior_hint(prior: dict[str, float]) -> str:
    if not prior:
        return ""
    top = sorted(prior.items(), key=lambda x: x[1], reverse=True)[:2]
    lines = [f"  - {intent}: {prob:.0%} of past emails to this recipient type"
             for intent, prob in top]
    return "Historical context (soft signal only — do not override clear intent signals):\n" + "\n".join(lines)
```

---

### 10.3 Multi-Turn Conversation Memory — Preserving Intent Across Refinements

When the user submits a refinement command (*"make it shorter"*, *"change the tone to casual"*), the intent of the email has not changed. Re-running Intent Detection wastes an LLM call and risks drift.

With LangGraph's `MemorySaver` and `thread_id`, the prior `intent` is already in state. The agent should detect a refinement turn and skip reclassification:

```python
def intent_detection_agent(state: dict) -> dict:
    # On a refinement turn, prior intent is already resolved — preserve it
    prior_intent = state.get("intent")
    raw_input = state.get("raw_input", "")

    REFINEMENT_SIGNALS = ["shorter", "longer", "tone", "make it", "change it", "less", "more"]
    is_refinement = prior_intent and any(s in raw_input.lower() for s in REFINEMENT_SIGNALS)

    if is_refinement:
        return {
            "intent": prior_intent,
            "intent_confidence": 1.0,
            "intent_fallback": False
        }

    # Normal classification flow for fresh requests
    ctx = state.get("parsed_context", {})
    # ... rest of implementation ...
```

---

### 10.4 What the Checkpointer Stores for This Agent

```python
# State snapshot after Intent Detection runs (stored by MemorySaver under thread_id):
{
    "intent": "apology",
    "intent_confidence": 0.96,
    "intent_fallback": False,
    # ... all prior fields from Input Parsing Agent ...
}
```

On the next turn (refinement), this state is loaded and `state["intent"]` is available immediately — no reclassification needed unless the intent itself changes.

---

### 10.5 Memory Scope Boundaries

| What This Agent Remembers | How Long |
|---|---|
| `intent` for current run | Until pipeline ends (in-state) |
| `intent` from prior turn in same session | Until session ends (MemorySaver) |
| User's historical intent distribution per recipient type | Permanently (user_profiles.json via prior_drafts) |

---

## 11. Testing Checklist

- [ ] Clear apology context → `apology` with high confidence
- [ ] Cold sales pitch context → `outreach` with high confidence
- [ ] Ambiguous input → confidence < threshold → falls back to `other`
- [ ] Unknown/invalid intent label from LLM → remapped to `other`
- [ ] LLM API failure → returns `other` with confidence 0.0, pipeline continues
- [ ] All 7 non-other intents covered by at least one test case

---

## 11. File Checklist

- [ ] `src/agents/intent_detection_agent.py` — agent function + chain + schema
- [ ] `src/agents/intent_templates.py` — `INTENT_TEMPLATES` dict
- [ ] Registered in `src/workflow/langgraph_flow.py`
- [ ] Unit tests in `tests/test_intent_detection_agent.py`
