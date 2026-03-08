# AI Email Assistant

A multi-agent AI system that converts natural language prompts into polished, professionally structured emails. Built with LangGraph, LangChain, and OpenAI, featuring a Streamlit UI, conversation memory, tone control, and automatic quality review.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Agent Pipeline](#agent-pipeline)
5. [Project Structure](#project-structure)
6. [Setup & Installation](#setup--installation)
7. [Running the App](#running-the-app)
8. [Running Tests](#running-tests)
9. [Configuration](#configuration)
10. [Supported Tones & Intents](#supported-tones--intents)
11. [Conversation Memory](#conversation-memory)
12. [Example Prompts](#example-prompts)
13. [Model Fallback & MCP](#model-fallback--mcp)

---

## Overview

The AI Email Assistant reduces email drafting time from 15–20 minutes to under 2 minutes. You describe the email you need in plain English, select a tone, and the 7-agent pipeline handles intent detection, tone enforcement, drafting, personalization, and quality review — all automatically.

Key design principles:
- **Modular agents**: Each agent does exactly one thing and passes results through a shared state
- **Conversation memory**: Follow-up requests carry over recipient, context, and tone from prior turns
- **Quality gate**: Every draft is reviewed for grammar, tone alignment, and coherence before delivery
- **Resilient routing**: Automatic retry with tightened instructions, and model fallback via MCP if quality thresholds are not met

---

## Features

| Feature | Details |
|---|---|
| Natural language input | Describe emails in plain English — no templates to fill |
| Intent detection | Classifies email type (apology, follow-up, outreach, request, etc.) automatically |
| Tone control | Formal / Casual / Assertive — selectable in sidebar, enforced throughout the draft |
| Conversation memory | Recipient name, context, and tone carry across turns in the same session |
| Refinement commands | Short commands like "make it shorter" or "add a P.S." surgically edit the existing draft |
| Quality review | LLM-as-judge scores grammar, tone alignment, and coherence before the draft is shown |
| Auto-retry | Failed reviews trigger a retry with specific correction instructions injected into the prompt |
| Model fallback | If retries are exhausted, the system switches to the next model in the fallback chain |
| Personalization | Injects user name, company, sign-off preference, and known contacts from the profile store |
| Voice matching | Optionally rewrites the draft to match the user's writing style from prior session history |
| Export options | Copy to clipboard, open in mail client (`mailto:`), or download as `.txt` |
| Session history | Every draft generated in the session is listed at the bottom of the page |

---

## Architecture

```
User Prompt (Streamlit UI)
        │
        ▼
┌─────────────────────┐
│ [1] Input Parser    │  Extracts: recipient, subject, tone hint, constraints, context notes
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ [2] Intent Detection│  Classifies email type with confidence score
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ [3] Tone Stylist    │  Resolves tone, builds directive list, selects tone sample file
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ [4] Draft Writer    │  Generates structured email JSON from intent + tone + context
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ [5] Personalization │  Slot-fills name/company, optionally applies voice matching
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ [6] Review Validator│  Scores grammar, tone alignment, coherence — PASS or FAIL
└────────┬────────────┘
         │
    ┌────┴────┐
   PASS      FAIL ──► retry (back to Draft Writer with correction instructions)
    │                  └──► after max retries: switch model via MCP fallback
    ▼
┌─────────────────────┐
│ [7] Router & Memory │  Assembles final draft, updates user profile, logs to history
└────────┬────────────┘
         │
         ▼
   Final Draft (Streamlit UI)
```

All agents communicate through a shared `EmailAssistantState` TypedDict managed by LangGraph. No agent calls another directly — they only read from and write to the shared state, making each agent independently testable and replaceable.

---

## Agent Pipeline

### Agent 1 — Input Parser (`input_parser_agent.py`)

Converts raw natural language into a structured `parsed_context` dict. Uses an LLM with a Pydantic schema enforced via `JsonOutputParser` to extract:

- `recipient_type` — client, colleague, manager, partner, vendor, other
- `recipient_name` — extracted name if mentioned
- `subject_hint` — 3–8 word topic summary
- `tone_hint` — formal, casual, or assertive if explicitly stated
- `constraints` — concise, apologetic, persuasive, etc.
- `urgency` — True if ASAP / today / urgent is mentioned
- `context_notes` — prior decisions, rejections, relationship history, goals

For refinement inputs ("make it shorter", "more assertive"), it applies the change to the existing parsed context without re-parsing from scratch.

For follow-up turns in the same conversation, it injects the prior session's `recipient_type`, `recipient_name`, `subject_hint`, and `tone_hint` as soft context — allowing the LLM to carry over known details without overriding anything the user explicitly changes.

**Failure behaviour**: Hard fail on empty input. Soft fallback (uses raw input as subject hint) on LLM errors.

---

### Agent 2 — Intent Detection (`intent_detection_agent.py`)

Classifies the email's primary purpose using a constrained classification prompt. Returns the intent label and a confidence score.

Supported intents: `outreach`, `follow_up`, `apology`, `informational`, `internal_update`, `request`, `thank_you`, `other`

Confidence threshold: **0.65** — below this, the intent falls back to `other`.

For refinement inputs in a multi-turn session, the prior intent is preserved without re-classification.

---

### Agent 3 — Tone Stylist (`tone_stylist_agent.py`)

Fully deterministic — no LLM call. Resolves tone using a strict priority order:

```
UI sidebar selection > prompt hint > user profile default > system default (formal)
```

Outputs a `tone_directives` list (e.g., "Use formal vocabulary and avoid contractions") injected directly into the Draft Writer's prompt. Also selects a `tone_sample_ref` — a few-shot example file from `data/tone_samples/` — for the specific tone × intent combination.

Cross-product modifiers are applied for special combinations: `formal + apology`, `assertive + outreach`, `casual + follow_up`, etc.

---

### Agent 4 — Draft Writer (`draft_writer_agent.py`)

The core generative agent. Assembles a composite prompt from:
- Intent label and structural requirements
- Tone directives from the Tone Stylist
- Parsed context (recipient, subject, constraints, context notes)
- Few-shot tone sample (if available)
- Retry correction instructions (on second+ attempt)

Returns a structured `EmailDraftDict`:
```python
{
    "subject_line": str,
    "salutation": str,
    "body_paragraphs": list[str],  # 2–4 paragraphs
    "closing": str,
    "sign_off": str,
}
```

Temperature scales up with retry count (base 0.4, +0.1 per retry) to encourage variation on retries.

For surgical edits ("make it shorter", "add a P.S."), a separate lightweight chain edits only the specific part of the existing draft without regenerating the whole email.

---

### Agent 5 — Personalization (`personalization_agent.py`)

Two-phase personalization:

**Phase 1 — Slot filling** (always runs):
- Replaces `[Name]` with the recipient's actual name
- Replaces `[Company]` with the user's company from their profile
- Applies the user's preferred sign-off phrase

**Phase 2 — Voice matching** (runs only when `writing_style_notes` has ≥ 20 characters):
- Sends the draft and style notes to an LLM with instructions to rewrite to match the user's voice without changing the content
- Falls back to Phase 1 result if the LLM call fails

Also retrieves prior drafts to the same recipient type for contextual references ("As discussed in our last communication...").

---

### Agent 6 — Review Validator (`review_agent.py`)

LLM-as-judge that evaluates the draft on three dimensions:

| Dimension | Default Threshold | Floor (after retries) |
|---|---|---|
| Grammar & fluency | 0.75 | 0.75 (fixed) |
| Tone alignment | 0.70 | 0.60 |
| Coherence | 0.75 | 0.65 |

Thresholds relax by 0.05 per retry to avoid infinite rejection loops.

**Fast-fail path**: If ≥ 3 structural issues are detected (missing subject, salutation, body, closing, or sign-off), the LLM call is skipped entirely and a FAIL is returned immediately.

**Fail-open on LLM crash**: If the review chain itself errors, a PASS with 0.8 scores is returned so the user always receives a draft.

Issues from prior review rounds are deduplicated on retry. Repeated issues are escalated with a `[Repeat issue — escalated]` prefix.

---

### Agent 7 — Router & Memory (`router_agent.py`)

Dual-purpose control agent:

**Routing**:
- `PASS` → assembles the final draft string and proceeds to output
- `FAIL, retry_count < fallback_on_retry` → increments retry counter, forwards issues back to Draft Writer
- `FAIL, retry_count >= fallback_on_retry` → switches `active_model` to the next in the MCP fallback chain
- All fallbacks exhausted → returns best available draft with a `pipeline_warning`

**Memory**:
- Appends a summary entry to `prior_drafts` in `user_profiles.json` (capped at 20 entries)
- If the user edited the draft in the UI, diffs the edited vs. original text and extracts style preferences into `writing_style_notes`
- Saves the updated profile to disk (non-blocking — profile save failures do not crash the pipeline)

---

## Project Structure

```
ai_email_assistant/
├── src/
│   ├── agents/
│   │   ├── input_parser_agent.py       # Agent 1: natural language → structured context
│   │   ├── intent_detection_agent.py   # Agent 2: intent classification
│   │   ├── tone_stylist_agent.py       # Agent 3: tone resolution & directives
│   │   ├── draft_writer_agent.py       # Agent 4: email generation
│   │   ├── personalization_agent.py    # Agent 5: slot-filling & voice matching
│   │   ├── review_agent.py             # Agent 6: quality gate
│   │   ├── router_agent.py             # Agent 7: routing + memory writes
│   │   └── intent_templates.py         # Structure templates per intent type
│   ├── models/
│   │   └── state.py                    # EmailAssistantState TypedDict + all nested types
│   ├── workflow/
│   │   └── langgraph_flow.py           # LangGraph graph definition + run_pipeline()
│   ├── ui/
│   │   └── streamlit_app.py            # Streamlit front-end
│   ├── memory/
│   │   └── user_profiles.json          # Persistent user profile store
│   └── integrations/
│       └── cohere_client.py            # LiteLLM wrapper for Cohere fallback
├── data/
│   └── tone_samples/                   # Few-shot tone × intent example files
│       ├── formal.txt
│       ├── formal_apology.txt
│       ├── casual.txt
│       ├── casual_follow_up.txt
│       ├── assertive.txt
│       ├── assertive_outreach.txt
│       └── ...
├── tests/
│   ├── test_input_parser_agent.py      # 17 tests
│   ├── test_intent_detection_agent.py  # Tests for all 8 intents + edge cases
│   ├── test_tone_stylist_agent.py      # Tests for priority chain, modifiers, refinement
│   ├── test_draft_writer_agent.py      # Tests for format helpers, surgical edit, retries
│   ├── test_personalization_agent.py   # Tests for slot-filling, voice matching, fallback
│   ├── test_review_agent.py            # Tests for thresholds, fast-fail, fail-open
│   └── test_router_agent.py            # Tests for all 4 routing paths + memory
├── config/
│   └── mcp.yaml                        # Model chain and retry configuration
├── .env                                # API keys (not committed)
├── pyproject.toml                      # Project metadata, dependencies, tool config
├── streamlit.md                        # Detailed Streamlit implementation plan
└── research.md                         # Full system design research document
```

---

## Setup & Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- An OpenAI API key

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd ai_email_assistant

# 2. Install dependencies (uv creates the venv automatically)
uv sync

# 3. Create the .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

The `.env` file must be at the project root (`ai_email_assistant/.env`). It is loaded automatically by all agents and the Streamlit app via `python-dotenv`.

---

## Running the App

```bash
uv run streamlit run src/ui/streamlit_app.py
```

Opens at `http://localhost:8501`.

### How to use

1. **Enter your profile** in the sidebar (Name, Company, Role) — pre-populated from `src/memory/user_profiles.json`
2. **Select a tone** — Formal, Casual, or Assertive
3. **Type a prompt** in the main text area and click **Generate**
4. **Edit the draft** directly in the text area if needed
5. **Use the action bar** to copy, open in your mail client, or download as `.txt`
6. **Refine with a follow-up** — type a short instruction like `"make it more concise"` or `"add a P.S. about the discount"` and click Generate again — the pipeline will carry over all prior context

### Advanced Options (sidebar)

| Option | Description |
|---|---|
| Show review scores | Displays grammar, tone alignment, and coherence metrics below the draft |
| Show agent trace | Expands a JSON view of the full pipeline state for debugging |

---

## Running Tests

```bash
# Run full test suite
uv run pytest

# Run a specific agent's tests
uv run pytest tests/test_input_parser_agent.py -v

# Run with output for debugging
uv run pytest -s
```

**137 tests across 7 test files.** All tests use `unittest.mock.patch` to mock LLM chain calls — no API key required to run tests.

### Type checking

```bash
uv run pyright src/
```

The project targets **zero pyright errors** in `basic` mode across all source files.

---

## Configuration

### `config/mcp.yaml`

Controls model selection and retry behaviour:

```yaml
primary_model: gpt-4o
fallback_chain:
  - claude-3-5-sonnet-20241022
  - cohere/command-r-plus
max_retries: 3
fallback_on_retry: 2
```

- `primary_model` — model used for all agents on the first attempt
- `fallback_chain` — ordered list of fallback models; the Router Agent switches to the next one when `retry_count >= fallback_on_retry`
- `max_retries` — total retry cap before the pipeline surfaces the best available draft with a warning
- `fallback_on_retry` — retry count at which the Router Agent switches models instead of retrying with the same model

### `src/memory/user_profiles.json`

Stores persistent user data. The `"default"` profile is used when no `user_id` is specified:

```json
{
  "default": {
    "user_id": "default",
    "name": "Vinit",
    "company": "Acme Corp",
    "role": "Software Engineer",
    "default_tone": "formal",
    "sign_off_preference": "Best regards",
    "writing_style_notes": "",
    "prior_drafts": [],
    "known_contacts": {}
  }
}
```

This file is updated automatically after each successful generation (via the Router & Memory Agent).

---

## Supported Tones & Intents

### Tones

| Tone | Characteristics |
|---|---|
| **Formal** | Precise vocabulary, no contractions, passive constructions where appropriate, professional salutation and sign-off |
| **Casual** | Contractions encouraged, conversational phrasing, warm opener, shorter sentences |
| **Assertive** | Active voice, direct opening, minimal hedging, explicit call-to-action |

Cross-product modifiers are applied for specific tone × intent combinations:
- `formal + apology` → adds softening language directives
- `assertive + outreach` → adds hook-writing instructions
- `casual + follow_up` → adds instruction to reference prior communication

### Intents

| Intent | Description |
|---|---|
| `outreach` | Cold contact, introduction, partnership or sales proposal |
| `follow_up` | Checking in after a previous email, meeting, or conversation |
| `apology` | Acknowledging a mistake, delay, or failure |
| `informational` | Sharing an update, announcement, or status report |
| `internal_update` | Team or organisation-internal communication |
| `request` | Asking for something specific — approval, document, meeting |
| `thank_you` | Expressing gratitude or appreciation |
| `other` | Does not clearly fit any of the above (uses a general template) |

Each intent has a defined structure in `intent_templates.py` specifying required sections and whether a call-to-action is mandatory.

---

## Conversation Memory

The app maintains conversation continuity within a session using two mechanisms:

### 1. LangGraph Checkpointing (`MemorySaver`)

Every pipeline invocation in the same browser session shares a `thread_id`. LangGraph's `MemorySaver` checkpoints the full state after each run. On the next invocation, the prior checkpoint is loaded and merged with the new input — so `parsed_context`, `intent`, `tone`, and prior drafts are all available to agents without being re-derived.

### 2. Input Parser Context Carry-Over

When a new (non-refinement) request arrives and a prior `parsed_context` exists in the checkpoint, the Input Parser injects the prior session's recipient type, recipient name, subject hint, and tone into the parsing prompt as a soft context hint. The LLM uses this to carry over known details while correctly overriding anything the user explicitly changes in the new prompt.

### 3. Refinement Detection

Short inputs containing signals like `"make it"`, `"shorter"`, `"longer"`, `"add"`, `"remove"`, `"more formal"` (and fewer than 100 characters) are detected as refinements. Instead of re-parsing, the Input Parser applies the refinement instruction to the existing `parsed_context` using a dedicated LLM chain. The Draft Writer then surgically edits only the affected part of the existing draft.

### 4. Profile-Based Long-Term Memory

After each successful generation, the Router & Memory Agent:
- Appends a draft summary to `prior_drafts` in `user_profiles.json`
- Extracts style preferences from any edits the user makes and writes them to `writing_style_notes`
- These persist across sessions and are used by the Personalization Agent for voice matching

---

## Example Prompts

### New email
```
Write a formal apology to my client Sarah about the delayed shipment. Keep it concise.
```

### Follow-up (context carried automatically)
```
Write a follow up to check if she received my apology
```

### Refinement
```
make it shorter and more assertive
```

### Tone change
```
more casual please
```

### Urgent request
```
URGENT: Email my vendor about the missing invoice, need a reply today
```

### Complex context
```
He said no. Write a follow up persuading him by mentioning the productivity benefits.
```

### Surgical edit
```
add a P.S. mentioning a 10% discount for early renewal
```

---

## Model Fallback & MCP

The Model Control Plane (MCP) is configured in `config/mcp.yaml`. The Router & Memory Agent reads this config to manage model switching:

```
Attempt 1–2:   gpt-4o-mini          (primary model, retries with correction instructions)
Attempt 3+:    claude-3-5-sonnet    (first fallback)
If exhausted:  cohere/command-r-plus (second fallback)
Still failing: Best available draft surfaced with a pipeline_warning banner in the UI
```

LiteLLM (`src/integrations/cohere_client.py`) provides the unified interface for non-OpenAI models, enabling transparent provider switching without changing agent code.
