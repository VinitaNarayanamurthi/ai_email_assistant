# Implementation Plan: Tone Stylist Agent
**File:** `src/agents/tone_stylist_agent.py`
**Position in pipeline:** Agent 3 of 7

---

## 1. Role & Responsibility

The Tone Stylist Agent resolves the final tone for the email and converts it into a concrete set of writing directives (`tone_directives`) that the Draft Writer Agent injects directly into its prompt. It also optionally resolves a reference to a few-shot tone sample from `data/tone_samples/`.

It does NOT generate email content. It only **selects, resolves, and encodes tone parameters**.

---

## 2. Dependencies

```python
import json
import os
from pathlib import Path
from typing import Optional
```

This agent is **mostly deterministic** — it primarily does lookup and resolution logic, not LLM calls. An optional LLM call is used only for tone/intent cross-product adjustments when needed.

```python
# Only needed for the optional LLM cross-product adjustment
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
```

---

## 3. State Interface

### Reads from shared state
| Key | Type | Description |
|---|---|---|
| `parsed_context` | `dict` | Contains `tone_hint` from user prompt |
| `intent` | `str` | Intent label — affects tone modifier selection |
| `user_profile` | `dict` | Contains `default_tone` as final fallback |
| `ui_tone_selection` | `str \| None` | Explicit tone chosen in the UI dropdown (highest priority) |

### Writes to shared state
| Key | Type | Description |
|---|---|---|
| `tone` | `str` | The resolved tone label |
| `tone_directives` | `list[str]` | List of writing instruction strings |
| `tone_sample_ref` | `str \| None` | Path to few-shot sample file if available |
| `tone_resolution_log` | `str` | Which source won the priority resolution |

---

## 4. Tone Directives Library

Store this as a constant dictionary in `src/agents/tone_stylist_agent.py`. It is the core of this agent.

```python
TONE_DIRECTIVES = {
    "formal": [
        "Use precise, professional vocabulary. Avoid slang and colloquialisms.",
        "Do not use contractions (use 'do not' instead of 'don't').",
        "Write in complete sentences with clear structure.",
        "Use a respectful salutation (e.g., 'Dear Mr./Ms. [Name],').",
        "Close with a professional sign-off (e.g., 'Sincerely,' or 'Best regards,').",
        "Maintain a neutral, objective tone throughout."
    ],
    "casual": [
        "Use a warm, conversational tone.",
        "Contractions are encouraged (e.g., 'I'd', 'we're', 'you've').",
        "Keep sentences short and easy to read.",
        "Use a friendly opener (e.g., 'Hope you're doing well!').",
        "Sign off informally (e.g., 'Cheers,' or 'Thanks,').",
        "It is okay to use light positive language, but avoid exclamation marks in every sentence."
    ],
    "assertive": [
        "Use direct, active voice throughout.",
        "State the main point or ask in the first or second sentence.",
        "Avoid hedging phrases like 'perhaps', 'might', 'kind of', 'just wanted to'.",
        "Each paragraph should have a single, clear purpose.",
        "End with a specific, actionable call-to-action with a clear deadline if applicable.",
        "Do not apologize unnecessarily or over-explain."
    ],
    "empathetic": [
        "Lead with acknowledgment of the recipient's feelings or situation.",
        "Use inclusive language ('we understand', 'together we can').",
        "Avoid clinical or detached language.",
        "Validate the recipient's experience before moving to solutions.",
        "Close with a reassuring and supportive statement."
    ],
    "urgent": [
        "Open with the urgency context immediately — do not bury the lead.",
        "Use bold or clear formatting cues (e.g., 'Action Required:', 'Time-Sensitive:') in the subject line.",
        "Keep the body concise — remove all non-essential information.",
        "State the deadline explicitly (day and time if possible).",
        "Close with a clear, specific next step and contact information."
    ],
    "diplomatic": [
        "Use balanced, neutral language that does not assign blame.",
        "Acknowledge multiple perspectives before proposing a resolution.",
        "Avoid absolute statements (never, always, impossible).",
        "Use softening phrases where appropriate ('It would be helpful if...', 'We appreciate your patience...').",
        "End on a collaborative, forward-looking note."
    ]
}
```

---

## 5. Tone + Intent Cross-Product Modifiers

Some combinations of tone and intent require additional directives beyond the base tone. Define these as additive modifiers:

```python
TONE_INTENT_MODIFIERS = {
    ("formal", "apology"): [
        "Begin by directly acknowledging the issue without deflecting responsibility.",
        "Do not use passive voice to obscure accountability.",
        "Offer a concrete remedy or next step."
    ],
    ("assertive", "outreach"): [
        "Open with a specific value proposition — not a generic introduction.",
        "Reference a relevant pain point the recipient likely has.",
        "End with a single, specific call-to-action (not multiple options)."
    ],
    ("casual", "follow_up"): [
        "Reference the prior interaction naturally and briefly.",
        "Keep the tone light — this is a check-in, not a demand.",
        "Make it easy for the recipient to say yes or respond simply."
    ],
    ("empathetic", "apology"): [
        "Validate any inconvenience caused before offering solutions.",
        "Use first-person accountability language ('I', 'we') — not passive.",
        "Do not rush to solutions before acknowledging impact."
    ],
    ("urgent", "request"): [
        "State the deadline and consequence of missing it in the first paragraph.",
        "Remove all pleasantries if the urgency is extreme.",
        "Offer to assist or provide whatever is needed to speed resolution."
    ]
}
```

---

## 6. Priority Resolution Logic

```python
PRIORITY_ORDER = ["ui_selection", "prompt_hint", "profile_default", "system_default"]
SYSTEM_DEFAULT_TONE = "formal"
VALID_TONES = set(TONE_DIRECTIVES.keys())
```

```python
def resolve_tone(state: dict) -> tuple[str, str]:
    """
    Returns (resolved_tone, resolution_source) based on priority order.
    Priority: UI dropdown > prompt hint > user profile > system default
    """
    # 1. Highest priority: explicit UI dropdown selection
    ui_tone = state.get("ui_tone_selection")
    if ui_tone and ui_tone in VALID_TONES:
        return ui_tone, "ui_selection"

    # 2. Tone hint extracted by Input Parsing Agent from user prompt
    ctx = state.get("parsed_context", {})
    prompt_tone = ctx.get("tone_hint")
    if prompt_tone and prompt_tone in VALID_TONES:
        return prompt_tone, "prompt_hint"

    # 3. User's default tone from their saved profile
    profile = state.get("user_profile", {})
    profile_tone = profile.get("default_tone")
    if profile_tone and profile_tone in VALID_TONES:
        return profile_tone, "profile_default"

    # 4. Final fallback
    return SYSTEM_DEFAULT_TONE, "system_default"
```

---

## 7. Tone Sample Resolution

Few-shot tone samples are stored in `data/tone_samples/` as plain text files named `{tone}_{intent}.txt` (e.g., `formal_apology.txt`, `casual_follow_up.txt`).

```python
TONE_SAMPLES_DIR = Path("data/tone_samples")

def resolve_tone_sample(tone: str, intent: str) -> Optional[str]:
    """Returns the path to a tone sample file if it exists, else None."""
    filename = f"{tone}_{intent}.txt"
    path = TONE_SAMPLES_DIR / filename
    if path.exists():
        return str(path)

    # Try tone-only fallback (e.g., formal.txt)
    fallback = TONE_SAMPLES_DIR / f"{tone}.txt"
    if fallback.exists():
        return str(fallback)

    return None
```

---

## 8. Implementation — Agent Node Function

```python
def tone_stylist_agent(state: dict) -> dict:
    # Step 1: Resolve tone
    tone, resolution_source = resolve_tone(state)

    # Step 2: Get base directives for resolved tone
    directives = list(TONE_DIRECTIVES.get(tone, TONE_DIRECTIVES[SYSTEM_DEFAULT_TONE]))

    # Step 3: Apply tone + intent modifiers
    intent = state.get("intent", "other")
    modifier_key = (tone, intent)
    if modifier_key in TONE_INTENT_MODIFIERS:
        directives.extend(TONE_INTENT_MODIFIERS[modifier_key])

    # Step 4: Resolve tone sample reference
    tone_sample_ref = resolve_tone_sample(tone, intent)

    return {
        "tone": tone,
        "tone_directives": directives,
        "tone_sample_ref": tone_sample_ref,
        "tone_resolution_log": f"Resolved via: {resolution_source}"
    }
```

---

## 9. Register in LangGraph

```python
from src.agents.tone_stylist_agent import tone_stylist_agent

graph.add_node("tone_stylist", tone_stylist_agent)
graph.add_edge("tone_stylist", "draft_writer")
```

---

## 10. Tone Sample Files to Create

Create these starter samples in `data/tone_samples/`:

| Filename | Content |
|---|---|
| `formal.txt` | A generic formal business email example |
| `casual.txt` | A generic casual professional email example |
| `assertive.txt` | A generic assertive email example |
| `formal_apology.txt` | A formal apology email example |
| `casual_follow_up.txt` | A casual follow-up email example |
| `assertive_outreach.txt` | An assertive cold outreach email example |
| `empathetic_apology.txt` | An empathetic apology email example |

Each file should contain a single complete example email that serves as a style reference for the Draft Writer Agent's few-shot section.

---

## 11. Example Input → Output

**State inputs:**
```json
{
  "ui_tone_selection": null,
  "parsed_context": { "tone_hint": "formal" },
  "intent": "apology",
  "user_profile": { "default_tone": "casual" }
}
```

**Output added to state:**
```json
{
  "tone": "formal",
  "tone_directives": [
    "Use precise, professional vocabulary. Avoid slang and colloquialisms.",
    "Do not use contractions...",
    "...(base formal directives)...",
    "Begin by directly acknowledging the issue without deflecting responsibility.",
    "Do not use passive voice to obscure accountability.",
    "Offer a concrete remedy or next step."
  ],
  "tone_sample_ref": "data/tone_samples/formal_apology.txt",
  "tone_resolution_log": "Resolved via: prompt_hint"
}
```

---

## 12. Conversation Memory

### Memory Type Overview

| Memory Type | Scope | Mechanism | This Agent's Role |
|---|---|---|---|
| **Short-term (within-session)** | Single pipeline run | LangGraph shared state | Writes `tone`, `tone_directives`, `tone_sample_ref` — consumed by Draft Writer |
| **Long-term (cross-session)** | Across sessions | `user_profiles.json` | Reads `default_tone` — the user's persistent tone preference built over multiple sessions |
| **Conversational (multi-turn)** | Same UI session | LangGraph `thread_id` checkpointing | Stores resolved tone so refinement turns can change it surgically (e.g., "make it more casual") |

---

### 12.1 Short-Term Memory — Tone Resolution Within a Session

Once this agent resolves `tone` and writes it to state, that decision persists for the entire current pipeline run — through Draft Writer, Personalization, and Review. No other agent re-runs tone resolution.

On a **retry loop** (Draft Writer → Review → fail → Draft Writer), the tone is **not re-resolved** — it stays fixed in state. Only `retry_issues` and `retry_count` change. This means if the Draft Writer fails because of a tone violation, the blame lies with the Draft Writer's prompt usage, not the Tone Stylist's resolution.

```python
# Tone is resolved once and stays in state for the full run:
state["tone"] = "formal"
state["tone_directives"] = ["Use precise vocabulary...", ...]
state["tone_sample_ref"] = "data/tone_samples/formal_apology.txt"
state["tone_resolution_log"] = "Resolved via: prompt_hint"
```

---

### 12.2 Long-Term Memory — Default Tone from User Profile

The user's `default_tone` in `user_profiles.json` is the accumulated result of:
1. The profile being set up initially (manual or onboarding)
2. The Router Agent updating tone preferences over time based on user edits

This agent reads it as the **lowest-priority fallback**. Over many sessions, the `default_tone` converges to what the user actually prefers, making the tone resolution increasingly accurate without any explicit user input.

```python
# Profile entry that this agent reads:
{
    "default_tone": "formal",   # built up over sessions
    ...
}

# Resolution priority:
# 1. ui_tone_selection (this session's dropdown)
# 2. parsed_context.tone_hint (this request's stated preference)
# 3. user_profile.default_tone  ← cross-session memory
# 4. system default ("formal")
```

---

### 12.3 Multi-Turn Conversation Memory — Tone Change Refinements

Tone is the most likely thing a user will refine after seeing a draft. *"Make it less formal"*, *"try a casual version"*, *"I want it more assertive"* — these are common refinement commands.

With LangGraph's `MemorySaver` and `thread_id`, the prior tone is stored in the checkpointed state. The Tone Stylist Agent re-runs on refinement turns and applies the new tone:

```python
TONE_CHANGE_SIGNALS = {
    "casual": ["casual", "informal", "friendly", "relaxed", "conversational"],
    "formal": ["formal", "professional", "official", "proper"],
    "assertive": ["assertive", "direct", "confident", "strong"],
    "empathetic": ["empathetic", "warm", "understanding", "compassionate"],
    "urgent": ["urgent", "time-sensitive", "quick", "asap"],
    "diplomatic": ["diplomatic", "balanced", "neutral", "tactful"]
}

def detect_tone_change_in_refinement(raw_input: str) -> str | None:
    """Returns the new tone label if the refinement requests a tone change, else None."""
    lowered = raw_input.lower()
    for tone, signals in TONE_CHANGE_SIGNALS.items():
        if any(signal in lowered for signal in signals):
            return tone
    return None
```

```python
def tone_stylist_agent(state: dict) -> dict:
    raw_input = state.get("raw_input", "")
    prior_tone = state.get("tone")  # from checkpointed state

    # Detect if this is a refinement that explicitly changes tone
    tone_override = detect_tone_change_in_refinement(raw_input) if prior_tone else None

    if tone_override:
        # User explicitly asked for a different tone — override everything
        directives = list(TONE_DIRECTIVES.get(tone_override, TONE_DIRECTIVES["formal"]))
        intent = state.get("intent", "other")
        modifier_key = (tone_override, intent)
        if modifier_key in TONE_INTENT_MODIFIERS:
            directives.extend(TONE_INTENT_MODIFIERS[modifier_key])
        return {
            "tone": tone_override,
            "tone_directives": directives,
            "tone_sample_ref": resolve_tone_sample(tone_override, intent),
            "tone_resolution_log": f"Resolved via: refinement_override (was: {prior_tone})"
        }

    # Normal resolution (first run or non-tone refinement)
    tone, resolution_source = resolve_tone(state)
    # ... rest of implementation ...
```

---

### 12.4 Storing Tone Preferences Back to Long-Term Memory

The Tone Stylist Agent does not directly write to `user_profiles.json`. However, when a user consistently chooses or overrides to a specific tone, the Router Agent's style diff logic will eventually extract this pattern and update `default_tone` in the profile. The Tone Stylist indirectly benefits from this because its fallback resolution reads `default_tone`.

The cycle looks like:
```
User keeps choosing "assertive" → Router Agent notes the pattern
→ updates default_tone = "assertive" in profile
→ next session: Tone Stylist resolves "assertive" automatically via profile_default
```

---

### 12.5 Memory Scope Boundaries

| What This Agent Remembers | How Long |
|---|---|
| `tone` + `tone_directives` for current run | Until pipeline ends (in-state) |
| `tone` from prior turn in same session | Until session ends (MemorySaver) |
| User's `default_tone` preference | Permanently (user_profiles.json) |
| Tone+intent modifier directives | In-process constant (TONE_INTENT_MODIFIERS dict) |

---

## 13. Testing Checklist

- [ ] UI selection present → wins over prompt hint and profile
- [ ] No UI selection, prompt hint present → prompt hint wins
- [ ] No UI or prompt hint → profile default used
- [ ] No profile default → system default (`formal`) used
- [ ] Invalid tone string in any source → skipped, next priority tried
- [ ] `formal` + `apology` → base directives + modifier directives combined
- [ ] `assertive` + `outreach` → correct modifiers appended
- [ ] Tone sample file exists → path returned correctly
- [ ] Tone sample file does not exist → returns `None` without error

---

## 13. File Checklist

- [ ] `src/agents/tone_stylist_agent.py` — agent function + all constants + helpers
- [ ] `data/tone_samples/*.txt` — at least 6 tone sample files created
- [ ] Registered in `src/workflow/langgraph_flow.py`
- [ ] Unit tests in `tests/test_tone_stylist_agent.py`
