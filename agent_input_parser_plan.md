# Implementation Plan: Input Parsing Agent
**File:** `src/agents/input_parser_agent.py`
**Position in pipeline:** Agent 1 of 7 — Entry point

---

## 1. Role & Responsibility

The Input Parsing Agent is the first node in the LangGraph pipeline. Its sole job is to convert raw, unstructured user input into a clean, validated, structured `parsed_context` dictionary that every downstream agent can consume without ambiguity.

It does NOT classify intent, choose tone, or generate content. It only **normalizes and validates**.

---

## 2. Dependencies

```toml
# pyproject.toml / requirements
langchain-core
langchain-openai
pydantic>=2.0
```

```python
# imports inside the file
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
| `raw_input` | `str` | The user's freeform prompt from the UI |
| `user_profile` | `dict` | Loaded user profile (for fallback defaults) |

### Writes to shared state
| Key | Type | Description |
|---|---|---|
| `parsed_context` | `dict` | Structured extraction of input |
| `parse_error` | `str \| None` | Error message if parsing failed |

---

## 4. Output Schema (Pydantic Model)

Define the expected output shape using Pydantic so it can be used both for function calling enforcement and for validation:

```python
class ParsedContext(BaseModel):
    recipient_type: str = Field(
        description="Type of recipient: client, colleague, manager, partner, vendor, other"
    )
    recipient_name: Optional[str] = Field(
        default=None,
        description="Specific name of recipient if mentioned, else null"
    )
    subject_hint: str = Field(
        description="The core topic or subject matter of the email in 3-8 words"
    )
    tone_hint: Optional[str] = Field(
        default=None,
        description="Tone explicitly requested by user: formal, casual, assertive, or null"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Any explicit constraints: concise, apologetic, urgent, avoid-blame, etc."
    )
    urgency: bool = Field(
        default=False,
        description="True if user indicated time-sensitivity (ASAP, urgent, today, etc.)"
    )
    context_notes: Optional[str] = Field(
        default=None,
        description="Any additional context that does not fit above fields"
    )
```

---

## 5. Prompt Template

```python
SYSTEM_PROMPT = """You are an expert at parsing natural language instructions for email composition.

Extract structured information from the user's email request. Be conservative:
- Only populate fields that are clearly stated or strongly implied
- Use null for any field not mentioned
- Do not infer or hallucinate details the user did not provide

Recipient types: client, colleague, manager, partner, vendor, other
Tone hints (only if user explicitly states): formal, casual, assertive
"""

USER_PROMPT = """Parse the following email request:

\"{raw_input}\"

Return a JSON object matching the schema exactly. Do not add extra fields."""
```

---

## 6. Implementation Steps

### Step 1 — Define the LLM call with function calling

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def build_parser_chain():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)  # temperature=0 for determinism

    parser = JsonOutputParser(pydantic_object=ParsedContext)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ]).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser
```

### Step 2 — Write the agent node function

LangGraph agent nodes are plain Python functions that take `state: dict` and return a `dict` of state updates.

```python
def input_parser_agent(state: dict) -> dict:
    raw_input = state.get("raw_input", "").strip()

    # Guard: reject empty input
    if not raw_input:
        return {
            "parsed_context": None,
            "parse_error": "Input is empty. Please describe the email you want to write."
        }

    chain = build_parser_chain()

    try:
        parsed = chain.invoke({"raw_input": raw_input})

        # Apply fallbacks from user profile for missing optional fields
        user_profile = state.get("user_profile", {})
        if parsed.get("tone_hint") is None and user_profile.get("default_tone"):
            parsed["tone_hint"] = user_profile["default_tone"]

        return {
            "parsed_context": parsed,
            "parse_error": None
        }

    except Exception as e:
        # Soft failure: return a minimal parsed context so the pipeline can continue
        return {
            "parsed_context": {
                "recipient_type": "recipient",
                "recipient_name": None,
                "subject_hint": raw_input[:60],  # use raw input as subject hint
                "tone_hint": state.get("user_profile", {}).get("default_tone", "formal"),
                "constraints": [],
                "urgency": False,
                "context_notes": raw_input
            },
            "parse_error": f"Parsing degraded: {str(e)}"
        }
```

### Step 3 — Validation helper

```python
def validate_parsed_context(parsed: dict) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_issues)."""
    issues = []

    if not parsed.get("subject_hint"):
        issues.append("Could not determine email subject. Please be more specific.")

    if not parsed.get("recipient_type"):
        issues.append("Could not determine who this email is for.")

    return len(issues) == 0, issues
```

### Step 4 — Register as a LangGraph node

In `src/workflow/langgraph_flow.py`:

```python
from src.agents.input_parser_agent import input_parser_agent

graph.add_node("input_parser", input_parser_agent)
graph.set_entry_point("input_parser")
graph.add_edge("input_parser", "intent_detection")
```

---

## 7. Conditional Edge After This Agent

After input parsing, check for a hard failure (empty input with no fallback). If `parse_error` is set AND `parsed_context` is None, route to an error terminal. Otherwise, proceed to Intent Detection.

```python
def after_input_parser(state: dict) -> str:
    if state.get("parsed_context") is None:
        return "error_terminal"
    return "intent_detection"

graph.add_conditional_edges("input_parser", after_input_parser, {
    "intent_detection": "intent_detection",
    "error_terminal": END
})
```

---

## 8. Example Input → Output

**User prompt:** `"Write an email to my client Sarah about the delayed shipment, keep it professional and apologetic"`

**Output `parsed_context`:**
```json
{
  "recipient_type": "client",
  "recipient_name": "Sarah",
  "subject_hint": "delayed shipment",
  "tone_hint": "formal",
  "constraints": ["apologetic"],
  "urgency": false,
  "context_notes": null
}
```

---

## 9. Conversation Memory

### Memory Type Overview

| Memory Type | Scope | Mechanism | This Agent's Role |
|---|---|---|---|
| **Short-term (within-session)** | Single pipeline run | LangGraph shared state dict | Writes `parsed_context` — available to all downstream agents in this run |
| **Long-term (cross-session)** | Across separate user sessions | `user_profiles.json` + LangGraph `MemorySaver` | Reads `user_profile.default_tone` as fallback when tone is not stated |
| **Conversational (multi-turn)** | Multiple invocations in one UI session | LangGraph `thread_id` checkpointing | Reads prior parsed context from the same thread if user refines their request |

---

### 9.1 Short-Term Memory — Within-Session State

The `parsed_context` dict this agent writes is the **foundation of all downstream agents** for the current run. It is stored in the LangGraph state object and lives for the duration of one pipeline execution:

```python
# This is what "short-term memory" looks like for this agent
state["parsed_context"] = {
    "recipient_type": "client",
    "recipient_name": "Sarah",
    "subject_hint": "delayed shipment",
    "tone_hint": "formal",
    "constraints": ["apologetic"],
    "urgency": False,
    "context_notes": None
}
```

Every downstream agent reads from `state["parsed_context"]` — none of them re-parse the raw input. This is why this agent is the single source of truth for the current request.

---

### 9.2 Long-Term Memory — Cross-Session Profile

When this agent finds a missing optional field (like `tone_hint`), it reads from the user's persistent profile as a fallback:

```python
# In input_parser_agent() — already shown in implementation
user_profile = state.get("user_profile", {})
if parsed.get("tone_hint") is None and user_profile.get("default_tone"):
    parsed["tone_hint"] = user_profile["default_tone"]
```

The profile was written by the Router & Memory Agent in a previous session and loaded before the graph starts. From this agent's perspective, it is **read-only long-term memory**.

---

### 9.3 Multi-Turn Conversation Memory — LangGraph Thread Persistence

The most important memory feature for this agent is support for **conversational refinement**: the user generates an email, reads it, and says *"actually make it shorter"* or *"change the recipient to my manager"* — without re-typing the full original prompt.

#### Setup: Enable LangGraph checkpointing

In `src/workflow/langgraph_flow.py`, attach a checkpointer when building the graph:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # in-memory (dev)
# from langgraph.checkpoint.sqlite import SqliteSaver  # persistent (prod)

checkpointer = MemorySaver()

graph = StateGraph(EmailAssistantState)
# ... add nodes and edges ...
compiled_graph = graph.compile(checkpointer=checkpointer)
```

#### Invoking with a thread_id

Each unique user session gets a `thread_id`. All graph invocations under the same `thread_id` share state history:

```python
config = {"configurable": {"thread_id": "user_vinit_session_001"}}

# First invocation — full prompt
result1 = compiled_graph.invoke(
    {"raw_input": "Write an apology email to Sarah about the delayed shipment"},
    config=config
)

# Second invocation — refinement only (user typed "make it shorter")
result2 = compiled_graph.invoke(
    {"raw_input": "make it shorter"},  # sparse input
    config=config
)
```

#### How this agent handles sparse refinement input

When `raw_input` is a refinement command rather than a full prompt, the Input Parsing Agent must detect this and **merge with the prior state** rather than overwriting it:

```python
REFINEMENT_SIGNALS = [
    "make it", "change it", "shorter", "longer", "more formal",
    "less formal", "add", "remove", "different tone", "update"
]

def is_refinement_input(raw_input: str) -> bool:
    lowered = raw_input.lower().strip()
    return any(signal in lowered for signal in REFINEMENT_SIGNALS) and len(lowered) < 100

def input_parser_agent(state: dict) -> dict:
    raw_input = state.get("raw_input", "").strip()

    # Check if this is a refinement of a prior request
    prior_context = state.get("parsed_context")  # from checkpointed prior run
    if prior_context and is_refinement_input(raw_input):
        return apply_refinement_to_context(raw_input, prior_context)

    # ... normal parsing flow ...
```

```python
def apply_refinement_to_context(refinement: str, prior_context: dict) -> dict:
    """
    Merges a short refinement command into the existing parsed_context.
    Only fields explicitly changed by the refinement are updated.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are updating a structured email request context based on a user's refinement command.
         Only change the fields explicitly mentioned in the refinement.
         Return the full updated JSON context with all original fields preserved unless changed.
         Original context: {prior_context}"""),
        ("human", "Refinement command: \"{refinement}\"\n\nReturn the updated JSON context.")
    ])

    from langchain_core.output_parsers import JsonOutputParser
    chain = prompt | llm | JsonOutputParser()

    try:
        updated = chain.invoke({
            "prior_context": str(prior_context),
            "refinement": refinement
        })
        return {"parsed_context": updated, "parse_error": None}
    except Exception:
        # If merge fails, re-parse from scratch treating refinement as new input
        return {"parsed_context": prior_context, "parse_error": None}
```

#### State the checkpointer persists between turns

```python
# After turn 1, the checkpointer stores the full state including:
{
    "raw_input": "Write an apology email to Sarah about the delayed shipment",
    "parsed_context": { "recipient_name": "Sarah", "subject_hint": "delayed shipment", ... },
    "intent": "apology",
    "tone": "formal",
    "final_draft": "Subject: Sincere Apologies...",
    ...
}

# Turn 2 starts with this state already populated.
# The agent reads prior parsed_context, detects refinement, merges changes.
```

---

### 9.4 Memory Scope Boundaries

| What This Agent Remembers | How Long |
|---|---|
| `parsed_context` from current run | Until pipeline ends (in-state) |
| Prior `parsed_context` from same UI session | Until session ends (MemorySaver) |
| User's `default_tone` from profile | Permanently (user_profiles.json) |
| Full history of prior runs | Not directly — delegated to Router Agent |

---

## 10. Testing Checklist

- [ ] Normal input with all fields → full structured output
- [ ] Input with no recipient name → `recipient_name: null`, recipient_type populated
- [ ] Input with no tone hint → falls back to profile default
- [ ] Empty string input → `parse_error` returned, pipeline halted gracefully
- [ ] Input with urgency signal ("ASAP") → `urgency: true`
- [ ] Very long input (>500 chars) → truncation or summary handled without error
- [ ] LLM API failure → soft fallback object returned, `parse_error` populated

---

## 10. File Checklist

- [ ] `src/agents/input_parser_agent.py` — agent node function + chain builder + validator
- [ ] `ParsedContext` Pydantic model defined in this file or in `src/models/schemas.py`
- [ ] Registered in `src/workflow/langgraph_flow.py`
- [ ] Unit tests in `tests/test_input_parser_agent.py`
