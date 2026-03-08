# Implementation Plan: Draft Writer Agent
**File:** `src/agents/draft_writer_agent.py`
**Position in pipeline:** Agent 4 of 7

---

## 1. Role & Responsibility

The Draft Writer Agent is the **core generative agent** — the only agent that produces the actual email text. By the time it runs, all decisions have been made upstream (what kind of email, what tone, what constraints). Its only job is to write a well-structured, coherent draft using those fully-resolved parameters.

It uses LangChain prompt templates with LLM function calling to enforce a structured JSON output, making the result easy for downstream agents to operate on.

---

## 2. Dependencies

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
import json

from src.agents.intent_templates import INTENT_TEMPLATES
```

---

## 3. State Interface

### Reads from shared state
| Key | Type | Description |
|---|---|---|
| `parsed_context` | `dict` | Recipient, subject, constraints from Agent 1 |
| `intent` | `str` | Email type label from Agent 2 |
| `tone` | `str` | Resolved tone label from Agent 3 |
| `tone_directives` | `list[str]` | Writing instructions from Agent 3 |
| `tone_sample_ref` | `str \| None` | Path to few-shot sample file |
| `retry_issues` | `list[str]` | Issues from the Review Agent on retry (empty on first run) |
| `retry_count` | `int` | Number of retry attempts so far |
| `active_model` | `str \| None` | Override model name set by Router Agent on fallback |

### Writes to shared state
| Key | Type | Description |
|---|---|---|
| `draft` | `dict` | Structured email draft object |
| `draft_error` | `str \| None` | Error if generation failed |

---

## 4. Output Schema (Structured Draft)

```python
class EmailDraft(BaseModel):
    subject_line: str = Field(
        description="A clear, specific subject line for the email (5-12 words)"
    )
    salutation: str = Field(
        description="Opening greeting (e.g., 'Dear Mr. Smith,' or 'Hi Sarah,')"
    )
    body_paragraphs: list[str] = Field(
        description=(
            "List of 2-4 body paragraphs. Each paragraph is a single string. "
            "Do not include salutation or sign-off here."
        ),
        min_length=2,
        max_length=4
    )
    closing: str = Field(
        description="A single closing sentence or call-to-action (1-2 sentences)"
    )
    sign_off: str = Field(
        description="Sign-off phrase only (e.g., 'Best regards,' or 'Sincerely,'). Do not include name."
    )
```

---

## 5. Prompt Template

The prompt is assembled from multiple upstream signals. This composite construction is the Email Template Engine role.

```python
SYSTEM_PROMPT = """You are an expert professional email writer.

Your task: write a complete, polished {intent_label} email.

TONE RULES — follow these strictly:
{tone_directives_formatted}

STRUCTURAL REQUIREMENTS for a {intent_label} email:
{structure_requirements}

CONSTRAINTS:
{constraints_formatted}

{retry_instructions}

Output format: Return a JSON object with exactly these fields:
- subject_line (string)
- salutation (string)
- body_paragraphs (list of 2-4 strings, one per paragraph)
- closing (string, 1-2 sentences)
- sign_off (string)

Do not include markdown formatting inside field values. Plain text only."""

USER_PROMPT = """Write the email with these details:

Recipient type: {recipient_type}
Recipient name: {recipient_name}
Subject / topic: {subject_hint}
Additional context: {context_notes}

{few_shot_section}

Now write the email:"""
```

---

## 6. Prompt Assembly Helpers

### Format tone directives as a numbered list
```python
def format_tone_directives(directives: list[str]) -> str:
    return "\n".join(f"{i+1}. {d}" for i, d in enumerate(directives))
```

### Format structure requirements from intent template
```python
def format_structure_requirements(intent: str) -> str:
    template = INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["other"])
    structure = template["structure"]
    lines = [f"- Include a section for: {section.replace('_', ' ')}" for section in structure]
    if template.get("cta_required"):
        lines.append("- A clear call-to-action is REQUIRED.")
    return "\n".join(lines)
```

### Format constraints
```python
def format_constraints(ctx: dict, retry_issues: list[str]) -> str:
    parts = []
    if ctx.get("constraints"):
        parts.append("User constraints: " + ", ".join(ctx["constraints"]))
    if ctx.get("urgency"):
        parts.append("This is an urgent email — keep it concise and action-oriented.")
    if not parts:
        parts.append("No special constraints.")
    return "\n".join(parts)
```

### Format retry instructions (only on retry runs)
```python
def format_retry_instructions(retry_issues: list[str], retry_count: int) -> str:
    if not retry_issues or retry_count == 0:
        return ""
    issues_str = "\n".join(f"- {issue}" for issue in retry_issues)
    return f"""IMPORTANT — This is retry attempt {retry_count}.
The previous draft was rejected for these reasons:
{issues_str}
Fix all of these issues in this new draft."""
```

### Load few-shot sample
```python
def load_tone_sample(tone_sample_ref: Optional[str]) -> str:
    if not tone_sample_ref:
        return ""
    path = Path(tone_sample_ref)
    if path.exists():
        sample_text = path.read_text(encoding="utf-8")
        return f"Reference example (match this tone and style, NOT the content):\n\"\"\"\n{sample_text}\n\"\"\""
    return ""
```

---

## 7. Implementation — Agent Node Function

```python
MAX_GENERATION_RETRIES = 2  # LLM-level retries for malformed output

def draft_writer_agent(state: dict) -> dict:
    ctx = state.get("parsed_context", {})
    intent = state.get("intent", "other")
    tone_directives = state.get("tone_directives", [])
    tone_sample_ref = state.get("tone_sample_ref")
    retry_issues = state.get("retry_issues", [])
    retry_count = state.get("retry_count", 0)
    active_model = state.get("active_model", "gpt-4o")

    # Assemble prompt components
    tone_directives_formatted = format_tone_directives(tone_directives)
    structure_requirements = format_structure_requirements(intent)
    constraints_formatted = format_constraints(ctx, retry_issues)
    retry_instructions = format_retry_instructions(retry_issues, retry_count)
    few_shot_section = load_tone_sample(tone_sample_ref)

    # Build chain
    llm = ChatOpenAI(model=active_model, temperature=0.4)
    parser = JsonOutputParser(pydantic_object=EmailDraft)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ])
    chain = prompt | llm | parser

    # Invoke with retry on malformed output
    for attempt in range(MAX_GENERATION_RETRIES):
        try:
            draft = chain.invoke({
                "intent_label": intent.replace("_", " "),
                "tone_directives_formatted": tone_directives_formatted,
                "structure_requirements": structure_requirements,
                "constraints_formatted": constraints_formatted,
                "retry_instructions": retry_instructions,
                "recipient_type": ctx.get("recipient_type", "recipient"),
                "recipient_name": ctx.get("recipient_name") or "[Name]",
                "subject_hint": ctx.get("subject_hint", ""),
                "context_notes": ctx.get("context_notes") or "None provided.",
                "few_shot_section": few_shot_section
            })

            return {
                "draft": draft,
                "draft_error": None
            }

        except Exception as e:
            if attempt == MAX_GENERATION_RETRIES - 1:
                return {
                    "draft": None,
                    "draft_error": f"Draft generation failed after {MAX_GENERATION_RETRIES} attempts: {str(e)}"
                }
            # else: retry the LLM call
```

---

## 8. Register in LangGraph

```python
from src.agents.draft_writer_agent import draft_writer_agent

graph.add_node("draft_writer", draft_writer_agent)
graph.add_edge("draft_writer", "personalization")
```

### Conditional edge: handle draft generation failure

```python
def after_draft_writer(state: dict) -> str:
    if state.get("draft") is None:
        return "error_terminal"
    return "personalization"

graph.add_conditional_edges("draft_writer", after_draft_writer, {
    "personalization": "personalization",
    "error_terminal": END
})
```

---

## 9. Temperature Strategy

| Scenario | Temperature | Reason |
|---|---|---|
| First attempt, normal | `0.4` | Some creativity, but consistent |
| Retry attempt | `0.5` | Slightly higher to break out of the failing pattern |
| MCP fallback model | `0.3` | Lower for a different model's calibration |

Adjust temperature based on retry count:
```python
temperature = 0.4 + (0.1 * min(retry_count, 2))
```

---

## 10. Example Input → Output

**State inputs:**
```json
{
  "intent": "apology",
  "tone": "formal",
  "tone_directives": ["Use precise vocabulary...", "No contractions...", "Begin with acknowledgment..."],
  "parsed_context": {
    "recipient_type": "client",
    "recipient_name": "[Name]",
    "subject_hint": "delayed shipment",
    "constraints": ["apologetic"],
    "urgency": false
  },
  "retry_issues": [],
  "retry_count": 0
}
```

**Output `draft` added to state:**
```json
{
  "subject_line": "Sincere Apologies Regarding Your Recent Shipment Delay",
  "salutation": "Dear [Name],",
  "body_paragraphs": [
    "We sincerely apologize for the delay in your recent shipment. We understand the inconvenience this has caused and take full responsibility for this lapse in our standard of service.",
    "The delay was caused by an unexpected disruption in our logistics chain. We have taken immediate steps to rectify the situation and ensure your order is dispatched at the earliest opportunity.",
    "We value your continued partnership and are committed to making this right. Please find attached a revised delivery timeline for your reference."
  ],
  "closing": "Should you have any questions or require further assistance, please do not hesitate to contact us directly.",
  "sign_off": "Sincerely,"
}
```

---

## 11. Conversation Memory

### Memory Type Overview

| Memory Type | Scope | Mechanism | This Agent's Role |
|---|---|---|---|
| **Short-term (within-session)** | Single pipeline run | LangGraph shared state | Reads all upstream decisions; writes `draft` — the central artifact of the run |
| **Retry memory (within-session)** | Across retry cycles in one run | `retry_issues` + `retry_count` in state | Remembers what the Review Agent rejected and incorporates fixes into the next draft |
| **Long-term (cross-session)** | Across sessions | `data/tone_samples/` files | Few-shot samples informed by patterns from prior successful drafts |
| **Conversational (multi-turn)** | Same UI session | LangGraph `thread_id` checkpointing | Reads prior `draft` to apply surgical edits on refinement turns instead of full rewrites |

---

### 11.1 Short-Term Memory — Retry Context Within a Single Run

The most unique form of memory for this agent is **retry context memory**: when the Review Agent rejects a draft and the pipeline loops back here, this agent reads `retry_issues` (what was wrong) and `retry_count` (how many times it has tried) from state. This is short-term memory that lives within a single pipeline execution.

```python
# On first run:
state["retry_issues"] = []
state["retry_count"] = 0

# After a Review Agent rejection:
state["retry_issues"] = [
    "Draft uses contractions — violates formal tone.",
    "Closing is missing a call-to-action."
]
state["retry_count"] = 1
```

The agent incorporates this into its prompt via `format_retry_instructions()`. Effectively, it "remembers" its past mistakes and is explicitly told to avoid them.

---

### 11.2 Multi-Turn Conversation Memory — Surgical Edits vs Full Rewrites

On refinement turns (*"make it shorter"*, *"add a P.S. about the discount"*), a full rewrite is wasteful and risks losing parts of the draft the user liked. With LangGraph's `MemorySaver` and `thread_id`, the prior `draft` is available in state.

Detect surgical edit requests and apply them to the existing draft rather than generating from scratch:

```python
SURGICAL_EDIT_SIGNALS = [
    "shorter", "longer", "add", "remove", "change paragraph",
    "rewrite only", "keep the opening", "just the closing", "p.s.", "postscript"
]

def is_surgical_edit(raw_input: str, prior_draft: dict | None) -> bool:
    if not prior_draft:
        return False
    lowered = raw_input.lower()
    return any(signal in lowered for signal in SURGICAL_EDIT_SIGNALS)
```

```python
SURGICAL_EDIT_SYSTEM_PROMPT = """You are editing a specific part of an existing email draft.

Apply ONLY the changes described in the edit instruction.
Preserve all other parts of the draft exactly as they are.
Return the complete updated draft as the same JSON structure.

Existing draft:
{existing_draft}"""

SURGICAL_EDIT_USER_PROMPT = """Edit instruction: \"{edit_instruction}\"

Return the updated draft JSON."""

def apply_surgical_edit(state: dict) -> dict:
    prior_draft = state.get("draft") or state.get("personalized_draft")
    raw_input = state.get("raw_input", "")
    active_model = state.get("active_model", "gpt-4o")

    llm = ChatOpenAI(model=active_model, temperature=0.2)
    from langchain_core.output_parsers import JsonOutputParser
    prompt = ChatPromptTemplate.from_messages([
        ("system", SURGICAL_EDIT_SYSTEM_PROMPT),
        ("human", SURGICAL_EDIT_USER_PROMPT)
    ])
    chain = prompt | llm | JsonOutputParser()

    try:
        updated_draft = chain.invoke({
            "existing_draft": json.dumps(prior_draft, indent=2),
            "edit_instruction": raw_input
        })
        return {"draft": updated_draft, "draft_error": None}
    except Exception as e:
        return {"draft": prior_draft, "draft_error": f"Surgical edit failed: {str(e)}"}
```

Updated agent node entry point to handle both modes:

```python
def draft_writer_agent(state: dict) -> dict:
    raw_input = state.get("raw_input", "")
    prior_draft = state.get("draft") or state.get("personalized_draft")

    # Check if this is a surgical edit on a refinement turn
    if is_surgical_edit(raw_input, prior_draft):
        return apply_surgical_edit(state)

    # Full generation (first run or non-surgical refinement)
    # ... existing implementation ...
```

---

### 11.3 Long-Term Memory — Few-Shot Samples from Past Successful Drafts

The `data/tone_samples/` directory is seeded initially with handcrafted examples, but over time it can be **extended with real user-approved drafts** as new few-shot references. When the user approves a draft, the Router Agent can write a cleaned version to `data/tone_samples/{tone}_{intent}_user.txt`. On the next run of the same tone/intent combo, this agent's `load_tone_sample()` finds it and uses it as a personalized reference.

```python
# Tone sample resolution order (in tone_stylist_agent.py):
# 1. {tone}_{intent}_user.txt  ← user-approved, most personalized
# 2. {tone}_{intent}.txt       ← generic handcrafted sample
# 3. {tone}.txt                ← tone-only fallback
# 4. None                      ← no few-shot section
```

---

### 11.4 What the Checkpointer Stores for This Agent

```python
# State snapshot after Draft Writer runs (stored by MemorySaver under thread_id):
{
    "draft": {
        "subject_line": "Sincere Apologies...",
        "salutation": "Dear Sarah,",
        "body_paragraphs": ["...", "...", "..."],
        "closing": "...",
        "sign_off": "Sincerely,"
    },
    "retry_count": 0,
    "retry_issues": [],
    # + all upstream fields
}
```

On a refinement turn, `state["draft"]` is the starting point for surgical edits, not a blank slate.

---

### 11.5 Memory Scope Boundaries

| What This Agent Remembers | How Long |
|---|---|
| `draft` from current run | Until pipeline ends (in-state) |
| `retry_issues` / `retry_count` from this run's retry cycles | Within a single pipeline execution |
| `draft` from prior turn in same session | Until session ends (MemorySaver) |
| Few-shot tone samples | Permanently (data/tone_samples/*.txt files) |

---

## 12. Testing Checklist

- [ ] Normal first-run generation → structured `draft` object returned
- [ ] Each of 7 intent types → correct structural elements present
- [ ] Formal tone → no contractions in output
- [ ] Assertive tone → active voice, CTA in closing
- [ ] Retry with `retry_issues` → retry instructions injected into prompt
- [ ] LLM returns malformed JSON → retries up to `MAX_GENERATION_RETRIES`, then returns `draft_error`
- [ ] `active_model` override → uses specified model instead of default
- [ ] `tone_sample_ref` file present → few-shot section included in prompt

---

## 12. File Checklist

- [ ] `src/agents/draft_writer_agent.py` — agent function + all prompt helpers + schema
- [ ] `src/agents/intent_templates.py` — `INTENT_TEMPLATES` dict (shared with Intent Detection)
- [ ] `data/tone_samples/*.txt` — sample files for few-shot reference
- [ ] Registered in `src/workflow/langgraph_flow.py`
- [ ] Unit tests in `tests/test_draft_writer_agent.py`
