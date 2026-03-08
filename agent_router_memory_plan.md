# Implementation Plan: Routing & Memory Agent
**File:** `src/agents/router_agent.py`
**Position in pipeline:** Agent 7 of 7

---

## 1. Role & Responsibility

The Routing & Memory Agent is the **control plane** of the pipeline. It has two distinct responsibilities that are both executed in a single node:

1. **Routing** — Reads the `review_result.verdict` and decides what happens next:
   - PASS → assemble the final draft and terminate the pipeline
   - FAIL + retries remaining → route back to the Draft Writer with the issue list
   - FAIL + retries exhausted → trigger MCP fallback (switch model) and retry once
   - All fallbacks exhausted → surface draft to user with a warning flag

2. **Memory** — On a successful completion, persists state to `user_profiles.json`:
   - Logs the current draft as a prior draft summary
   - Updates writing style notes if the user edited the draft in the UI
   - Saves the updated profile back to disk

It is both the **memory producer** and the **terminal orchestrator** of the pipeline.

---

## 2. Dependencies

```python
import json
from datetime import date
from pathlib import Path
from typing import Optional
from copy import deepcopy

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import yaml
```

---

## 3. State Interface

### Reads from shared state
| Key | Type | Description |
|---|---|---|
| `review_result` | `dict` | Verdict + scores + issues from Review Agent |
| `retry_count` | `int` | Number of draft retries so far |
| `personalized_draft` | `dict` | The draft that was just reviewed |
| `user_profile` | `dict` | Current user profile |
| `intent` | `str` | Email intent (for draft logging) |
| `parsed_context` | `dict` | For logging recipient type and subject |
| `active_model` | `str \| None` | Current LLM model in use |
| `user_edited_draft` | `str \| None` | If user edited the draft in UI, the edited text |

### Writes to shared state
| Key | Type | Description |
|---|---|---|
| `final_draft` | `str` | Fully assembled plain-text email |
| `retry_count` | `int` | Incremented if retrying |
| `retry_issues` | `list[str]` | Issue list forwarded to Draft Writer |
| `active_model` | `str` | Updated if model switched via MCP |
| `pipeline_warning` | `str \| None` | Warning shown in UI if max retries hit |
| `pipeline_status` | `str` | `"success"`, `"retrying"`, `"fallback"`, `"warning"` |

---

## 4. Configuration — MCP Model Chain

Read the fallback model chain from `config/mcp.yaml`:

```yaml
# config/mcp.yaml
primary_model: gpt-4o
fallback_chain:
  - claude-3-5-sonnet-20241022
  - command-r-plus        # Cohere via LiteLLM
max_retries: 3
fallback_on_retry: 2     # Switch model after this many standard retries
```

```python
CONFIG_PATH = Path("config/mcp.yaml")
PROFILES_PATH = Path("src/memory/user_profiles.json")

def load_mcp_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

MCP_CONFIG = load_mcp_config()
MAX_RETRIES = MCP_CONFIG.get("max_retries", 3)
FALLBACK_ON_RETRY = MCP_CONFIG.get("fallback_on_retry", 2)
FALLBACK_CHAIN = MCP_CONFIG.get("fallback_chain", [])
PRIMARY_MODEL = MCP_CONFIG.get("primary_model", "gpt-4o")
```

---

## 5. Routing Logic

```python
def determine_next_action(state: dict) -> str:
    """
    Returns the routing decision string consumed by LangGraph conditional edges.

    Returns:
        "success"   → pipeline ends, draft is finalized
        "retry"     → route back to draft_writer with issues
        "fallback"  → switch LLM model, route back to draft_writer
        "warning"   → all retries and fallbacks exhausted, output draft with warning
    """
    verdict = state.get("review_result", {}).get("verdict", "FAIL")
    retry_count = state.get("retry_count", 0)
    active_model = state.get("active_model", PRIMARY_MODEL)

    if verdict == "PASS":
        return "success"

    # Not passed: check retry budget
    if retry_count < FALLBACK_ON_RETRY:
        return "retry"

    # Retry budget hit: try switching model
    current_index = (
        FALLBACK_CHAIN.index(active_model)
        if active_model in FALLBACK_CHAIN
        else -1
    )
    next_index = current_index + 1
    if next_index < len(FALLBACK_CHAIN):
        return "fallback"

    # All fallbacks exhausted
    return "warning"
```

---

## 6. Draft Assembly — Plain Text

The `draft` object from the Draft Writer is structured JSON. Convert it to a readable plain-text email string for display in the UI.

```python
def assemble_final_draft(draft: dict, sender_name: str = "") -> str:
    """Converts the structured draft dict into a plain-text email string."""
    lines = []

    subject = draft.get("subject_line", "")
    if subject:
        lines.append(f"Subject: {subject}")
        lines.append("")

    salutation = draft.get("salutation", "")
    if salutation:
        lines.append(salutation)
        lines.append("")

    for para in draft.get("body_paragraphs", []):
        if para.strip():
            lines.append(para.strip())
            lines.append("")

    closing = draft.get("closing", "")
    if closing:
        lines.append(closing)
        lines.append("")

    sign_off = draft.get("sign_off", "")
    if sign_off:
        lines.append(sign_off)
        if sender_name:
            lines.append(sender_name)

    return "\n".join(lines).strip()
```

---

## 7. Memory Operations

### 7.1 Log the draft to user profile

```python
def log_draft_to_profile(profile: dict, state: dict) -> dict:
    """Appends a summary of the current draft to the user's prior_drafts list."""
    profile = deepcopy(profile)
    ctx = state.get("parsed_context", {})
    draft = state.get("personalized_draft", {})

    # Summarize the draft (first 200 chars of assembled text)
    assembled = assemble_final_draft(draft)
    summary = assembled[:200] + "..." if len(assembled) > 200 else assembled

    entry = {
        "date": str(date.today()),
        "intent": state.get("intent", "other"),
        "recipient_type": ctx.get("recipient_type", "unknown"),
        "recipient_name": ctx.get("recipient_name"),
        "subject_hint": ctx.get("subject_hint", ""),
        "draft_summary": summary
    }

    if "prior_drafts" not in profile:
        profile["prior_drafts"] = []

    profile["prior_drafts"].append(entry)

    # Keep only the 20 most recent drafts to prevent unbounded growth
    profile["prior_drafts"] = profile["prior_drafts"][-20:]

    return profile
```

### 7.2 Update style notes from user edits

If the user edited the draft in the UI, diff the original and edited versions to extract style preferences:

```python
STYLE_DIFF_SYSTEM_PROMPT = """You are analyzing the difference between an AI-generated email draft and a human-edited version.

Your task: extract 1-3 concise observations about the STYLE differences (not content).
These will be saved as writing style notes to improve future drafts.

Focus on:
- Formality level adjustments
- Sentence length preferences
- Specific phrases added or removed
- Structural preferences (e.g., "user always adds a subject restatement in the opener")

Do NOT describe content changes (changed facts, different subjects, etc.)
Return a list of 1-3 short style observation strings."""

def extract_style_from_edits(original: str, edited: str) -> list[str]:
    if not original or not edited or original.strip() == edited.strip():
        return []

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", STYLE_DIFF_SYSTEM_PROMPT),
        ("human", "Original:\n{original}\n\nEdited:\n{edited}\n\nReturn a JSON list of style observations.")
    ])

    from langchain_core.output_parsers import JsonOutputParser
    chain = prompt | llm | JsonOutputParser()

    try:
        observations = chain.invoke({"original": original, "edited": edited})
        return observations if isinstance(observations, list) else []
    except Exception:
        return []

def update_style_notes(profile: dict, new_observations: list[str]) -> dict:
    profile = deepcopy(profile)
    existing = profile.get("writing_style_notes", "")
    if new_observations:
        combined = existing + " " + " ".join(new_observations)
        profile["writing_style_notes"] = combined.strip()
    return profile
```

### 7.3 Persist profile to disk

```python
def save_user_profile(user_id: str, profile: dict) -> None:
    if not PROFILES_PATH.exists():
        PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        all_profiles = {}
    else:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            all_profiles = json.load(f)

    all_profiles[user_id] = profile

    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, indent=2, ensure_ascii=False)
```

---

## 8. Implementation — Agent Node Function

```python
def router_memory_agent(state: dict) -> dict:
    action = determine_next_action(state)
    profile = state.get("user_profile", {})
    user_id = profile.get("user_id", "default")

    # ── SUCCESS PATH ──────────────────────────────────────────────────────────
    if action == "success":
        draft = state.get("personalized_draft") or state.get("draft", {})
        sender_name = profile.get("name", "")
        final_text = assemble_final_draft(draft, sender_name)

        # Memory: log draft
        updated_profile = log_draft_to_profile(profile, state)

        # Memory: update style notes from user edits (if any)
        user_edited = state.get("user_edited_draft")
        if user_edited:
            style_obs = extract_style_from_edits(final_text, user_edited)
            updated_profile = update_style_notes(updated_profile, style_obs)

        # Memory: persist to disk (non-blocking — don't fail on write error)
        try:
            save_user_profile(user_id, updated_profile)
        except Exception:
            pass  # Log warning but don't crash the pipeline

        return {
            "final_draft": final_text,
            "user_profile": updated_profile,
            "pipeline_status": "success",
            "pipeline_warning": None
        }

    # ── RETRY PATH ────────────────────────────────────────────────────────────
    if action == "retry":
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "retry_issues": state.get("review_result", {}).get("issues", []),
            "pipeline_status": "retrying"
        }

    # ── MODEL FALLBACK PATH ───────────────────────────────────────────────────
    if action == "fallback":
        current_model = state.get("active_model", PRIMARY_MODEL)
        current_index = (
            FALLBACK_CHAIN.index(current_model)
            if current_model in FALLBACK_CHAIN
            else -1
        )
        next_model = FALLBACK_CHAIN[current_index + 1]

        return {
            "active_model": next_model,
            "retry_count": state.get("retry_count", 0) + 1,
            "retry_issues": state.get("review_result", {}).get("issues", []),
            "pipeline_status": "fallback"
        }

    # ── WARNING PATH (all retries & fallbacks exhausted) ─────────────────────
    if action == "warning":
        draft = state.get("personalized_draft") or state.get("draft", {})
        sender_name = profile.get("name", "")
        final_text = assemble_final_draft(draft, sender_name)

        return {
            "final_draft": final_text,
            "pipeline_status": "warning",
            "pipeline_warning": (
                "This draft could not be fully validated after multiple attempts. "
                "Please review it carefully before sending."
            )
        }
```

---

## 9. Register in LangGraph — Conditional Edges

```python
from langgraph.graph import END
from src.agents.router_agent import router_memory_agent, determine_next_action

graph.add_node("router_memory", router_memory_agent)

# Conditional edges based on routing decision
graph.add_conditional_edges(
    "router_memory",
    determine_next_action,
    {
        "success": END,
        "retry": "draft_writer",      # Loop back to Draft Writer with issues
        "fallback": "draft_writer",   # Also goes to Draft Writer, but with new model
        "warning": END                # Surface draft with warning
    }
)
```

---

## 10. MCP Model Switching — Integration Detail

When `active_model` is updated in state and the graph loops back to `draft_writer`, the Draft Writer Agent reads `state["active_model"]` when building its LLM:

```python
# In draft_writer_agent.py
active_model = state.get("active_model", "gpt-4o")
llm = ChatOpenAI(model=active_model, temperature=0.4)
```

For Cohere models via LiteLLM, the integration is in `src/integrations/cohere_client.py` and uses LiteLLM's `completion()` wrapper, which makes Cohere API-compatible with the OpenAI interface. The `active_model` string for Cohere would be `"cohere/command-r-plus"` and LiteLLM routes it correctly.

```python
# src/integrations/cohere_client.py
from litellm import completion

def call_cohere(model: str, messages: list[dict], temperature: float = 0.4) -> str:
    response = completion(
        model=model,  # e.g., "cohere/command-r-plus"
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content
```

---

## 11. Pipeline Status → UI Display

The Streamlit UI should read `pipeline_status` and `pipeline_warning` from state to display appropriate feedback:

| `pipeline_status` | UI Behavior |
|---|---|
| `"success"` | Display draft normally, no warning |
| `"retrying"` | Show spinner with "Improving draft..." message |
| `"fallback"` | Show spinner with "Switching model, retrying..." message |
| `"warning"` | Display draft with yellow warning banner showing `pipeline_warning` text |

---

## 12. Example State Transitions

**Scenario: PASS on first attempt**
```
review_result.verdict = "PASS"
retry_count = 0
→ action = "success"
→ final_draft assembled, profile logged, pipeline ends
```

**Scenario: FAIL on first attempt, retry**
```
review_result.verdict = "FAIL"
retry_count = 0  (< FALLBACK_ON_RETRY=2)
→ action = "retry"
→ retry_count = 1, retry_issues forwarded to draft_writer
→ pipeline loops back to draft_writer
```

**Scenario: FAIL after 2 retries, switch model**
```
review_result.verdict = "FAIL"
retry_count = 2  (== FALLBACK_ON_RETRY=2)
active_model = "gpt-4o" (not in fallback_chain)
→ action = "fallback"
→ active_model = "claude-3-5-sonnet-20241022"
→ retry_count = 3
→ pipeline loops back to draft_writer with new model
```

**Scenario: FAIL after all fallbacks**
```
review_result.verdict = "FAIL"
retry_count = 4
active_model = "command-r-plus" (last in chain)
→ action = "warning"
→ final_draft assembled from current (imperfect) draft
→ pipeline_warning set → UI shows warning banner
```

---

## 13. Conversation Memory

### Memory Type Overview

| Memory Type | Scope | Mechanism | This Agent's Role |
|---|---|---|---|
| **Short-term (within-session)** | Single pipeline run | LangGraph shared state | Reads entire state; writes `final_draft`, `pipeline_status`, `pipeline_warning` |
| **Long-term (cross-session)** | Across sessions | `user_profiles.json` | Primary memory producer — writes draft logs, style notes, contact updates, approved samples |
| **Conversational (multi-turn)** | Same UI session | LangGraph `thread_id` checkpointing | Manages session-level state: persists `final_draft` so refinement turns have context |
| **Episodic (cross-session)** | Across sessions | `prior_drafts` in profile | Appends each completed draft as an episode — feeds Personalization Agent in future runs |

This agent is the **sole memory writer** of the entire pipeline. All other agents that "read" long-term memory are reading what this agent has written in past sessions.

---

### 13.1 The Full Memory Write Cycle

Every successful pipeline completion triggers a cascade of memory writes:

```
Pipeline PASS
    ↓
1. Assemble final_draft (plain text)
    ↓
2. Log draft to prior_drafts  (episodic memory)
    ↓
3. Extract style from user edits  (behavioral memory)
    ↓
4. Update known_contacts  (relational memory)
    ↓
5. Save approved draft as tone sample  (few-shot memory)
    ↓
6. Persist updated profile to user_profiles.json
    ↓
7. Update state with final_draft + updated user_profile
```

All of this runs in the `"success"` branch of `router_memory_agent()`.

---

### 13.2 LangGraph MemorySaver — Session-Level Conversation Memory

The `MemorySaver` (or `SqliteSaver` for production) attached to the compiled graph stores the full state dict after every node execution. This is the foundation for multi-turn conversation within a UI session.

#### Graph setup with checkpointing

```python
# src/workflow/langgraph_flow.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional

class EmailAssistantState(TypedDict, total=False):
    raw_input: str
    ui_tone_selection: Optional[str]
    parsed_context: Optional[dict]
    parse_error: Optional[str]
    intent: Optional[str]
    intent_confidence: Optional[float]
    intent_fallback: Optional[bool]
    tone: Optional[str]
    tone_directives: Optional[list]
    tone_sample_ref: Optional[str]
    tone_resolution_log: Optional[str]
    draft: Optional[dict]
    draft_error: Optional[str]
    personalized_draft: Optional[dict]
    personalization_log: Optional[list]
    review_result: Optional[dict]
    retry_issues: Optional[list]
    prior_review_issues: Optional[list]
    retry_count: int
    active_model: Optional[str]
    user_profile: Optional[dict]
    user_edited_draft: Optional[str]
    final_draft: Optional[str]
    pipeline_status: Optional[str]
    pipeline_warning: Optional[str]

checkpointer = MemorySaver()
graph = StateGraph(EmailAssistantState)
# ... add_node / add_edge calls ...
compiled_graph = graph.compile(checkpointer=checkpointer)
```

#### Invoking with thread_id

```python
# src/ui/streamlit_app.py

import streamlit as st
import uuid

# Generate a thread_id per browser session (persists while tab is open)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# First turn
result = compiled_graph.invoke(
    {
        "raw_input": st.session_state.user_prompt,
        "ui_tone_selection": st.session_state.tone_selection,
        "user_profile": load_user_profile(st.session_state.user_id),
        "retry_count": 0
    },
    config=config
)

# Refinement turn — same thread_id means prior state is loaded automatically
result = compiled_graph.invoke(
    {"raw_input": "make it shorter"},
    config=config
)
# At this point, state already contains: intent, tone, draft, personalized_draft, etc.
# from the prior turn. Only raw_input is new.
```

---

### 13.3 Production Memory: SqliteSaver for Cross-Process Persistence

`MemorySaver` is in-process (lost when the app restarts). For production, use `SqliteSaver` to persist conversation threads to disk:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = "data/conversation_memory.db"

with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    compiled_graph = graph.compile(checkpointer=checkpointer)
```

With `SqliteSaver`, a user can close the app and resume the same thread later. The graph restores the full state from the DB and continues as if the session never ended.

Thread lifecycle management:

```python
def start_new_thread(user_id: str) -> str:
    """Generate a new thread_id for a fresh email composition."""
    return f"{user_id}_{uuid.uuid4().hex[:8]}"

def list_user_threads(user_id: str, checkpointer) -> list[str]:
    """List all active threads for a user (for session history UI)."""
    # Thread IDs are namespaced by user_id prefix
    return [tid for tid in checkpointer.list_threads() if tid.startswith(user_id)]
```

---

### 13.4 Writing All Memory Layers in the Success Path

The expanded success branch that writes all memory layers:

```python
if action == "success":
    draft = state.get("personalized_draft") or state.get("draft", {})
    sender_name = profile.get("name", "")
    final_text = assemble_final_draft(draft, sender_name)
    updated_profile = deepcopy(profile)

    # Layer 1 — Episodic: log this draft
    updated_profile = log_draft_to_profile(updated_profile, state)

    # Layer 2 — Behavioral: extract style from user edits
    user_edited = state.get("user_edited_draft")
    if user_edited:
        style_obs = extract_style_from_edits(final_text, user_edited)
        updated_profile = update_style_notes(updated_profile, style_obs)

    # Layer 3 — Relational: update known_contacts with recipient name
    ctx = state.get("parsed_context", {})
    recipient_type = ctx.get("recipient_type")
    recipient_name = ctx.get("recipient_name")
    if recipient_type and recipient_name:
        updated_profile.setdefault("known_contacts", {})[recipient_type] = recipient_name

    # Layer 4 — Few-shot: save approved draft as a tone sample
    tone = state.get("tone", "")
    intent = state.get("intent", "")
    review = state.get("review_result", {})
    if tone and intent and review.get("tone_alignment_score", 0) >= 0.90:
        # High-quality draft — save as future few-shot reference
        sample_path = Path(f"data/tone_samples/{tone}_{intent}_user.txt")
        try:
            sample_path.write_text(final_text, encoding="utf-8")
        except Exception:
            pass  # Non-critical

    # Layer 5 — Issue patterns: record recurring review failures to style notes
    prior_review_issues = state.get("prior_review_issues", [])
    if state.get("retry_count", 0) >= 1 and prior_review_issues:
        updated_profile = record_recurring_issues(updated_profile, [prior_review_issues])

    # Persist everything to disk
    try:
        save_user_profile(profile.get("user_id", "default"), updated_profile)
    except Exception:
        pass

    return {
        "final_draft": final_text,
        "user_profile": updated_profile,
        "pipeline_status": "success",
        "pipeline_warning": None
    }
```

---

### 13.5 Memory Scope Boundaries

| What This Agent Writes | Where | How Long |
|---|---|---|
| `final_draft` plain text | LangGraph state | Until session ends (MemorySaver) |
| Prior draft entry | `user_profiles.json` prior_drafts | Permanently (last 20 kept) |
| Style observations from edits | `user_profiles.json` writing_style_notes | Permanently, appended |
| Known contact associations | `user_profiles.json` known_contacts | Permanently, updated |
| User-approved tone sample | `data/tone_samples/{tone}_{intent}_user.txt` | Permanently (file on disk) |
| Recurring issue notes | `user_profiles.json` writing_style_notes | Permanently, appended |
| Full pipeline state | LangGraph MemorySaver / SqliteSaver | Session or cross-session |

---

## 14. Testing Checklist

- [ ] PASS verdict → `final_draft` assembled correctly, profile saved
- [ ] FAIL + retry_count < threshold → `retry_count` incremented, `retry_issues` forwarded
- [ ] FAIL + retry_count == threshold → `active_model` switched to first fallback
- [ ] FAIL + all fallbacks exhausted → `pipeline_warning` set, draft returned
- [ ] Profile JSON write failure → pipeline continues, no crash
- [ ] User edited draft provided → style observations extracted and saved to profile
- [ ] `assemble_final_draft` → correct plain-text formatting with all sections
- [ ] `log_draft_to_profile` → prior_drafts capped at 20 entries
- [ ] MCP config loaded correctly from `mcp.yaml`

---

## 14. File Checklist

- [ ] `src/agents/router_agent.py` — agent function + routing logic + memory operations + draft assembly
- [ ] `config/mcp.yaml` — model chain and retry configuration
- [ ] `src/integrations/cohere_client.py` — LiteLLM-based Cohere wrapper
- [ ] `src/memory/user_profiles.json` — initialized with at least one sample profile
- [ ] Conditional edges registered in `src/workflow/langgraph_flow.py`
- [ ] Unit tests in `tests/test_router_agent.py`
