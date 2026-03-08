# Implementation Plan: Personalization Agent
**File:** `src/agents/personalization_agent.py`
**Position in pipeline:** Agent 5 of 7

---

## 1. Role & Responsibility

The Personalization Agent is the **memory consumer** of the pipeline. It takes the generic email draft from the Draft Writer Agent and makes it feel like it was written by and for a specific person. It does this in two phases:

1. **Slot-filling** — Direct substitutions using known facts (name, company, sign-off)
2. **LLM-assisted rewriting** — Subtle voice matching using writing history from the user profile (only when history exists)

It does NOT invent facts. It only injects what is known from the user profile and session state.

---

## 2. Dependencies

```python
import json
from pathlib import Path
from typing import Optional
from copy import deepcopy

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
```

---

## 3. State Interface

### Reads from shared state
| Key | Type | Description |
|---|---|---|
| `draft` | `dict` | Structured email draft from Draft Writer Agent |
| `parsed_context` | `dict` | Recipient name/type from Input Parsing Agent |
| `user_profile` | `dict` | Full user profile loaded from `user_profiles.json` |
| `intent` | `str` | Email intent — used for historical reference lookup |

### Writes to shared state
| Key | Type | Description |
|---|---|---|
| `personalized_draft` | `dict` | Draft with personalization applied |
| `personalization_log` | `list[str]` | What was injected and why |

---

## 4. User Profile Schema

The user profile is stored in `src/memory/user_profiles.json`. Define the expected schema:

```python
# Example profile structure
USER_PROFILE_SCHEMA = {
    "user_id": str,              # Unique identifier
    "name": str,                 # Sender's full name
    "company": str,              # Sender's company name
    "role": str,                 # Sender's job title
    "default_tone": str,         # Preferred tone label
    "sign_off_preference": str,  # e.g., "Best regards", "Thanks"
    "writing_style_notes": str,  # Free-text style observations
    "prior_drafts": [            # List of past draft summaries
        {
            "date": str,
            "intent": str,
            "recipient_type": str,
            "recipient_name": str,
            "subject_hint": str,
            "draft_summary": str  # Short summary of what was written
        }
    ]
}
```

### Loading the profile

```python
PROFILES_PATH = Path("src/memory/user_profiles.json")

def load_user_profile(user_id: str) -> dict:
    if not PROFILES_PATH.exists():
        return {}
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    return profiles.get(user_id, {})
```

---

## 5. Phase 1 — Slot-Filling

Direct, deterministic substitutions. No LLM involved.

```python
def apply_slot_filling(draft: dict, ctx: dict, profile: dict) -> tuple[dict, list[str]]:
    """
    Replaces placeholder tokens in the draft with known values.
    Returns (updated_draft, log_entries).
    """
    result = deepcopy(draft)
    log = []

    # Resolve recipient name
    recipient_name = ctx.get("recipient_name") or profile.get("known_contacts", {}).get(
        ctx.get("recipient_type"), None
    )
    if recipient_name and recipient_name != "[Name]":
        result["salutation"] = result["salutation"].replace("[Name]", recipient_name)
        log.append(f"Injected recipient name: {recipient_name}")

    # Inject sender sign-off preference
    sign_off_pref = profile.get("sign_off_preference")
    if sign_off_pref:
        result["sign_off"] = sign_off_pref + ","
        log.append(f"Applied preferred sign-off: {sign_off_pref}")

    # Replace company placeholders in body
    company = profile.get("company")
    if company:
        result["body_paragraphs"] = [
            p.replace("[Company]", company).replace("[Your Company]", company)
            for p in result["body_paragraphs"]
        ]
        if company in str(result["body_paragraphs"]):
            log.append(f"Injected company name: {company}")

    # Replace sender name placeholder in sign-off area (if present)
    sender_name = profile.get("name")
    if sender_name:
        result["sign_off"] = result.get("sign_off", "").rstrip(",") + ","
        log.append(f"Sign-off set, sender name available: {sender_name}")

    return result, log
```

---

## 6. Phase 2 — Historical Reference Injection

If prior drafts exist with the same recipient type or intent, reference them in the body.

```python
def find_prior_context(profile: dict, intent: str, recipient_type: str) -> Optional[str]:
    """
    Looks for a prior draft that matches the current intent and recipient.
    Returns a brief context string if found, else None.
    """
    prior_drafts = profile.get("prior_drafts", [])
    if not prior_drafts:
        return None

    # Look for same recipient + same intent
    for draft in reversed(prior_drafts):  # most recent first
        if draft.get("recipient_type") == recipient_type and draft.get("intent") == intent:
            return draft.get("draft_summary")

    # Fallback: same recipient any intent
    for draft in reversed(prior_drafts):
        if draft.get("recipient_type") == recipient_type:
            return draft.get("draft_summary")

    return None
```

---

## 7. Phase 2 — LLM-Assisted Voice Matching

Only runs if `writing_style_notes` is present in the profile (indicating enough history to calibrate).

```python
VOICE_MATCH_SYSTEM_PROMPT = """You are helping refine an email draft to better match a specific person's writing style.

Do NOT change the content, facts, intent, or tone category.
Only make subtle adjustments to match the style notes below.
Keep all changes minimal — if in doubt, leave text unchanged.

Writer's style notes:
{style_notes}

Prior context about this recipient (for reference only, do not copy verbatim):
{prior_context}"""

VOICE_MATCH_USER_PROMPT = """Refine this email draft to better match the writer's voice.
Return the same JSON structure with the same fields. Only modify text, not structure.

Current draft:
{draft_json}"""

def apply_voice_matching(draft: dict, profile: dict, prior_context: Optional[str]) -> tuple[dict, list[str]]:
    style_notes = profile.get("writing_style_notes", "")
    log = []

    # Skip if no style notes to calibrate against
    if not style_notes or len(style_notes) < 20:
        return draft, log

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)  # cheaper model, subtle task
    prompt = ChatPromptTemplate.from_messages([
        ("system", VOICE_MATCH_SYSTEM_PROMPT),
        ("human", VOICE_MATCH_USER_PROMPT)
    ])

    try:
        from langchain_core.output_parsers import JsonOutputParser
        chain = prompt | llm | JsonOutputParser()
        refined = chain.invoke({
            "style_notes": style_notes,
            "prior_context": prior_context or "No prior interactions on record.",
            "draft_json": json.dumps(draft, indent=2)
        })
        log.append("Applied LLM voice matching based on writing style notes.")
        return refined, log
    except Exception as e:
        log.append(f"Voice matching skipped (error): {str(e)}")
        return draft, log  # Fail gracefully — return original draft
```

---

## 8. Implementation — Agent Node Function

```python
def personalization_agent(state: dict) -> dict:
    draft = state.get("draft")
    if not draft:
        # No draft to personalize — pass through
        return {"personalized_draft": draft, "personalization_log": ["No draft to personalize."]}

    ctx = state.get("parsed_context", {})
    profile = state.get("user_profile", {})
    intent = state.get("intent", "other")
    log = []

    # Phase 1: Slot-filling (always runs)
    draft, slot_log = apply_slot_filling(draft, ctx, profile)
    log.extend(slot_log)

    # Find prior context for this recipient/intent combo
    prior_context = find_prior_context(
        profile,
        intent=intent,
        recipient_type=ctx.get("recipient_type", "")
    )
    if prior_context:
        log.append(f"Found prior interaction context for {ctx.get('recipient_type')} + {intent}.")

    # Phase 2: Voice matching (only if style notes exist)
    draft, voice_log = apply_voice_matching(draft, profile, prior_context)
    log.extend(voice_log)

    return {
        "personalized_draft": draft,
        "personalization_log": log
    }
```

---

## 9. Register in LangGraph

```python
from src.agents.personalization_agent import personalization_agent

graph.add_node("personalization", personalization_agent)
graph.add_edge("personalization", "review_validator")
```

---

## 10. Profile Loading — Where It Happens

The user profile should be loaded **before the pipeline starts**, not inside this agent, so all agents can access it from state. Do this at the entry point (e.g., the Streamlit UI or a pre-graph setup step):

```python
# In src/workflow/langgraph_flow.py or in the UI before graph invocation
def load_profile_into_state(user_id: str, initial_state: dict) -> dict:
    profile = load_user_profile(user_id)
    initial_state["user_profile"] = profile
    return initial_state
```

---

## 11. Example Input → Output

**State `draft` (from Draft Writer):**
```json
{
  "subject_line": "Sincere Apologies Regarding Your Recent Shipment Delay",
  "salutation": "Dear [Name],",
  "body_paragraphs": ["We sincerely apologize...", "The delay was caused by...", "We value your partnership..."],
  "closing": "Please do not hesitate to contact us.",
  "sign_off": "Sincerely,"
}
```

**Profile:**
```json
{
  "name": "Vinit",
  "company": "Acme Corp",
  "sign_off_preference": "Best regards",
  "writing_style_notes": "Prefers concise emails. Rarely uses 'hereby' or overly formal language.",
  "prior_drafts": []
}
```

**Output `personalized_draft`:**
```json
{
  "subject_line": "Sincere Apologies Regarding Your Recent Shipment Delay",
  "salutation": "Dear Sarah,",
  "body_paragraphs": ["We sincerely apologize...", "The delay was caused by...", "We value your partnership..."],
  "closing": "Please do not hesitate to contact us.",
  "sign_off": "Best regards,"
}
```

**`personalization_log`:**
```json
[
  "Injected recipient name: Sarah",
  "Applied preferred sign-off: Best regards",
  "Voice matching skipped: no sufficient style history."
]
```

---

## 12. Conversation Memory

### Memory Type Overview

| Memory Type | Scope | Mechanism | This Agent's Role |
|---|---|---|---|
| **Short-term (within-session)** | Single pipeline run | LangGraph shared state | Reads `draft` + `parsed_context`; writes `personalized_draft` |
| **Long-term (cross-session)** | Across sessions | `user_profiles.json` | Primary memory consumer — reads `prior_drafts`, `writing_style_notes`, `known_contacts` |
| **Conversational (multi-turn)** | Same UI session | LangGraph `thread_id` checkpointing | On refinement turns, re-applies updated personalization to the modified draft without reloading all profile data |

---

### 12.1 Long-Term Memory — The Profile as a Living Memory Store

This agent is the **deepest user of long-term memory** in the entire pipeline. Everything it does — slot-filling, prior context injection, voice matching — is driven by data that accumulated over multiple past sessions.

The `user_profiles.json` is structured as a memory store with three distinct memory layers:

```python
{
    "user_id": "vinit_123",

    # Layer 1: Factual identity memory (permanent, manually set or onboarded)
    "name": "Vinit",
    "company": "Acme Corp",
    "role": "Software Engineer",
    "sign_off_preference": "Best regards",

    # Layer 2: Behavioral preference memory (updated by Router Agent over time)
    "default_tone": "formal",
    "writing_style_notes": "Prefers concise emails. Avoids overly formal openers.",

    # Layer 3: Episodic memory (rolling log of past drafts — most recent 20)
    "prior_drafts": [
        {
            "date": "2026-03-01",
            "intent": "follow_up",
            "recipient_type": "client",
            "recipient_name": "Sarah",
            "subject_hint": "Q1 proposal review",
            "draft_summary": "Dear Sarah, Following up on our Q1 proposal..."
        }
    ],

    # Layer 4: Relational memory (known contacts with their names)
    "known_contacts": {
        "client": "Sarah Chen",
        "manager": "David Kim",
        "partner": "Alex Rivera"
    }
}
```

**Layer 4 — Relational memory** is a new addition not previously shown. It allows the Personalization Agent to resolve recipient names even when the user does not mention them in their prompt (e.g., "email my client" → looks up "Sarah Chen").

```python
def apply_slot_filling(draft: dict, ctx: dict, profile: dict) -> tuple[dict, list[str]]:
    # Resolve recipient name from known_contacts if not in parsed_context
    recipient_name = ctx.get("recipient_name")
    if not recipient_name:
        known_contacts = profile.get("known_contacts", {})
        recipient_type = ctx.get("recipient_type", "")
        recipient_name = known_contacts.get(recipient_type)  # e.g., "client" → "Sarah Chen"

    if recipient_name and recipient_name != "[Name]":
        result["salutation"] = result["salutation"].replace("[Name]", recipient_name)
        log.append(f"Injected recipient name from known_contacts: {recipient_name}")
    # ... rest of slot-filling ...
```

---

### 12.2 Episodic Memory — Prior Drafts as Conversation History

The `prior_drafts` list is the **episodic memory** of this agent. It stores what was said in past "conversations" (email generation sessions) with specific recipient types. This agent uses it to make current drafts feel like a continuation of an ongoing relationship.

```python
def find_prior_context(profile: dict, intent: str, recipient_type: str) -> str | None:
    """
    Retrieves the most relevant prior interaction as context.
    Prioritizes: same intent + same recipient > same recipient only.
    """
    prior_drafts = profile.get("prior_drafts", [])
    # most recent first
    for draft in reversed(prior_drafts):
        if draft.get("recipient_type") == recipient_type and draft.get("intent") == intent:
            return draft.get("draft_summary")
    for draft in reversed(prior_drafts):
        if draft.get("recipient_type") == recipient_type:
            return draft.get("draft_summary")
    return None
```

When `prior_context` is found, it is passed to the LLM voice matching phase so the model can naturally reference the prior interaction in the new draft body — e.g., *"As mentioned in my last email..."* — without the user specifying this explicitly.

---

### 12.3 Multi-Turn Conversation Memory — Re-personalizing on Refinement

When a user refines a draft on a second turn (*"change the tone"*, *"make it shorter"*), the Draft Writer produces an updated `draft`. The Personalization Agent must **re-apply personalization** to the new draft without redundantly re-loading the full profile.

With LangGraph's `MemorySaver`, the `user_profile` loaded in turn 1 is still in state for turn 2. The agent re-runs slot-filling and voice matching on the new draft using the already-loaded profile:

```python
def personalization_agent(state: dict) -> dict:
    draft = state.get("draft")
    profile = state.get("user_profile", {})  # already in state from turn 1

    # Profile is already loaded — no file I/O needed on refinement turns
    # Just re-apply personalization to the new/updated draft
    # ... existing implementation ...
```

This is efficient because the profile file is only read **once per session** (at graph initialization), not once per agent invocation.

---

### 12.4 Updating Known Contacts from Conversation

Over time, as the user generates emails to specific named recipients, the relational memory can be updated automatically. The Router Agent (or a separate post-processing step) can extract new contact associations from `parsed_context` and add them to `known_contacts`:

```python
# In router_agent.py — after a successful run:
def update_known_contacts(profile: dict, ctx: dict) -> dict:
    recipient_type = ctx.get("recipient_type")
    recipient_name = ctx.get("recipient_name")
    if recipient_type and recipient_name:
        if "known_contacts" not in profile:
            profile["known_contacts"] = {}
        # Only update if this is a new or different name
        if profile["known_contacts"].get(recipient_type) != recipient_name:
            profile["known_contacts"][recipient_type] = recipient_name
    return profile
```

This creates a **growing relational memory** that makes future personalization more automatic.

---

### 12.5 Memory Scope Boundaries

| What This Agent Remembers | How Long |
|---|---|
| `personalized_draft` from current run | Until pipeline ends (in-state) |
| `user_profile` (loaded at start of session) | Until session ends (MemorySaver) |
| Factual identity (name, company, sign-off) | Permanently (user_profiles.json, Layer 1) |
| Behavioral preferences (tone, style notes) | Permanently, updated over time (Layer 2) |
| Episodic history (prior drafts) | Last 20 entries permanently (Layer 3) |
| Known contacts (name ↔ recipient type) | Permanently, grows over time (Layer 4) |

---

## 13. Testing Checklist

- [ ] Recipient name in `parsed_context` → injected into salutation
- [ ] No recipient name → `[Name]` left as-is (Review Agent catches if needed)
- [ ] `sign_off_preference` in profile → applied to sign-off
- [ ] `company` in profile → replaces `[Company]` in body paragraphs
- [ ] `writing_style_notes` present → LLM voice matching runs
- [ ] `writing_style_notes` absent/short → LLM phase skipped, no error
- [ ] Prior drafts with matching intent/recipient → `prior_context` passed to voice matching
- [ ] LLM voice matching fails → original draft returned without crashing pipeline
- [ ] No profile at all → slot-filling returns original draft with no changes

---

## 13. File Checklist

- [ ] `src/agents/personalization_agent.py` — agent function + all phases + profile loader
- [ ] `src/memory/user_profiles.json` — at least one sample profile for testing
- [ ] Registered in `src/workflow/langgraph_flow.py`
- [ ] Profile loaded into initial state before graph invocation
- [ ] Unit tests in `tests/test_personalization_agent.py`
