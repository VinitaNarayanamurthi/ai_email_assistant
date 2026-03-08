# Research Report: AI-Powered Email Assistant
## Capstone Project — Applied Agentic AI for Software Engineers

---

## 1. Overview

The AI-Powered Email Assistant is a capstone project for the "Applied Agentic AI for SWEs" course. Its core purpose is to demonstrate how agentic AI systems can solve a real productivity problem: the time cost and inconsistency of manually composing professional emails.

**Core value proposition:** Reduce email drafting time from 15–20 minutes down to under 2 minutes, while maintaining contextual relevance, appropriate tone, and on-brand quality.

---

## 2. Business Use Case

### Problem
Professionals spend excessive time composing emails. This creates:
- Lost productivity at scale
- Inconsistent tone and quality across the organization
- Cognitive load that distracts from higher-value tasks

### Business Opportunity
| Benefit | Description |
|---|---|
| Productivity Boost | 15–20 min → <2 min per email |
| Context-Aware Writing | Tailors tone and intent to recipient and situation |
| Multi-Purpose | Handles outreach, follow-ups, internal updates, apologies, and more |
| Scalability | Enables consistent, on-brand communication at team/org level |

---

## 3. Technical Architecture

The system is built as a **multi-agent pipeline** that takes user intent and context and converts it into a personalized, polished email draft.

### Core Components

| Component | Primary Tech | Alternatives | Purpose |
|---|---|---|---|
| Multi-Agent System | LangGraph / CrewAI | AutoGen, Semantic Kernel | Modular task-specific agent orchestration |
| Language Models | GPT-4, Claude | Gemini, Cohere | Language understanding and generation |
| Email Template Engine | LangChain + Function Calling | PromptLayer | Controls tone, length, and formatting |
| User Profile Store | JSON / MongoDB | Postgres, Weaviate | Stores preferences and prior context |
| Web Interface | Streamlit | Gradio, Flask | UI for composing, editing, and exporting emails |
| Memory Layer | LangGraph Memory | Redis, Pinecone | Remembers user tone and previous drafts |
| Control Plane (MCP) | LangSmith / LiteLLM | Helicone, Custom Router | Optimizes model selection and routing |

### Key Design Decisions
- **LangGraph** is the primary orchestration framework, chosen for its stateful graph-based workflow management and built-in memory support.
- **Function Calling** is used to enforce structured, consistent output formatting from LLMs.
- **MCP (Model Control Plane)** enables dynamic routing between models, supporting fallback and optimization strategies.
- **JSON / MongoDB** for user profiles keeps the persistence layer simple and flexible, with clear upgrade paths to vector stores (Weaviate) if semantic search is needed.

---

## 4. Agent Architecture — In-Depth

The system uses **7 specialized agents**, each handling a distinct, non-overlapping stage of the email generation pipeline. Agents are wired together using **LangGraph**, which treats the pipeline as a directed state graph where each node is an agent and edges define the flow of control and data.

---

### 4.1 How the Workflow Operates (LangGraph State Machine)

LangGraph represents the pipeline as a **stateful directed graph**. The central concept is a shared **state object** — a Python dictionary (or typed dataclass) that every agent reads from and writes to. No agent communicates with another directly; they all operate on this shared state. This design has important consequences:

- **Isolation**: Each agent only knows what's in the state. It cannot call another agent or access global variables.
- **Traceability**: Every state transition is logged. You can inspect the state at any point in the pipeline.
- **Resumability**: Because state is explicit, LangGraph can checkpoint mid-pipeline and resume after a failure.
- **Conditional branching**: Edges in the graph can be conditional — the Routing & Memory Agent, for example, can direct flow back to an earlier agent (retry) or forward to the end (success).

The shared state object likely contains fields such as:

```python
{
  "raw_input": str,           # Original user prompt
  "parsed_context": dict,     # Output of Input Parsing Agent
  "intent": str,              # Output of Intent Detection Agent
  "tone": str,                # Output of Tone Stylist Agent
  "draft": str,               # Output of Draft Writer Agent
  "personalized_draft": str,  # Output of Personalization Agent
  "review_result": dict,      # Output of Review & Validator Agent
  "final_draft": str,         # Final output after all stages
  "user_profile": dict,       # Loaded from user_profiles.json
  "retry_count": int,         # Managed by Routing & Memory Agent
  "error": str | None         # Error signal for fallback routing
}
```

### 4.2 Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│  (prompt + tone selector + optional metadata from UI)       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   [1] Input Parsing     │  ← Normalizes & validates
              │        Agent            │    raw input into structured
              └────────────┬────────────┘    context object
                           │
                           ▼
              ┌─────────────────────────┐
              │   [2] Intent Detection  │  ← Classifies email type
              │        Agent            │    (outreach, follow-up, etc.)
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   [3] Tone Stylist      │  ← Selects & encodes
              │        Agent            │    tone parameters
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   [4] Draft Writer      │  ← Generates email body
              │        Agent            │    using intent + tone
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   [5] Personalization   │  ← Injects user profile,
              │        Agent            │    history, context
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  [6] Review & Validator │  ← Quality gate: grammar,
              │        Agent            │    tone, coherence check
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │   Passes?               │
              │  YES ──► continue       │
              │  NO  ──► retry (back    │
              │          to step 4)     │
              └────────────┬────────────┘
                           │ (on pass)
                           ▼
              ┌─────────────────────────┐
              │  [7] Routing & Memory   │  ← Logs draft, updates
              │        Agent            │    user profile, handles
              └────────────┬────────────┘    fallback/retry routing
                           │
                           ▼
                  ┌─────────────────┐
                  │  FINAL DRAFT    │  → Displayed in Streamlit UI
                  └─────────────────┘
```

---

### 4.3 Agent 1 — Input Parsing Agent

**File:** `src/agents/input_parser_agent.py`

#### What It Does
This is the entry point of the pipeline. Its job is to take raw, unstructured user input — a freeform prompt like *"write an email to my client about the delayed delivery, keep it professional"* — and convert it into a clean, structured context object that downstream agents can reliably consume.

#### Why It Exists
Without normalization, every downstream agent would need to handle ambiguous natural language input independently. That creates duplication, inconsistency, and fragility. By centralizing parsing here, the rest of the pipeline operates on clean, predictable data.

#### What It Extracts
From the user prompt, it parses and validates:
- **Recipient info**: Who is the email going to? (name, role, relationship — client, colleague, manager)
- **Intent signal**: A preliminary intent hint (even before the Intent Detection Agent does formal classification)
- **Tone preference**: Has the user explicitly stated a tone? ("keep it professional", "be friendly")
- **Constraints**: Length constraints, urgency signals ("ASAP"), sensitive topics to avoid
- **Context metadata**: Subject matter, referenced events (e.g., "the delayed delivery"), any deadlines

#### How It Works Technically
- Sends the raw input to an LLM with a **structured extraction prompt** — instructing the model to return a JSON object with specific fields (using function calling to enforce the schema).
- Validates the returned JSON against expected fields. Missing required fields (e.g., no recipient) trigger an error signal.
- If validation fails, it either prompts the user for more information (via the UI) or falls back to safe defaults.

#### Output
A structured `parsed_context` dict added to the shared state:
```python
{
  "recipient": "client",
  "recipient_name": None,  # not provided
  "subject_hint": "delayed delivery",
  "tone_hint": "professional",
  "constraints": ["apologetic", "concise"],
  "urgency": False
}
```

#### Failure Modes
- Completely empty or gibberish input → hard validation failure, request re-input
- Ambiguous recipient → defaults to generic "recipient", downstream personalization fills in from profile
- Missing tone hint → passes None, Tone Stylist Agent uses user's profile default

---

### 4.4 Agent 2 — Intent Detection Agent

**File:** `src/agents/intent_detection_agent.py`

#### What It Does
Classifies the email's purpose into one of a fixed set of intent categories. This classification drives the structural template and rhetorical strategy the Draft Writer Agent will use.

#### Why It Exists
Different email types have different structures, conventions, and success criteria. An outreach email starts with a hook and ends with a call-to-action. A follow-up email references prior communication. An apology email leads with acknowledgment, not justification. Without explicit intent classification, the Draft Writer would have to guess the structure from context — a much harder and less reliable task.

#### Intent Categories
Based on the project description, at minimum the following intents are supported:
- **Outreach** — cold contact, introduction, partnership proposal
- **Follow-up** — checking in after a previous email or meeting
- **Apology** — acknowledging a mistake or delay
- **Informational** — sharing an update, announcement, or status
- **Internal update** — team/org communications
- *(Extensible to: request, negotiation, thank-you, onboarding, etc.)*

#### How It Works Technically
Uses **classification prompts** — a prompt engineering pattern where the LLM is not asked to generate free text, but to choose from a predefined list:

```
Given the following parsed context, classify the email intent.
Choose exactly one from: [outreach, follow-up, apology, informational, internal-update, other]
Return only the label and a confidence score.

Context: {parsed_context}
```

This is more reliable than open-ended generation because:
- Output space is constrained — the model can't hallucinate a new intent category
- Confidence scores allow the Router Agent to decide if re-classification is needed
- Downstream agents receive a deterministic token ("apology") rather than a sentence to re-parse

#### Output
Updates the shared state with:
```python
{
  "intent": "apology",
  "intent_confidence": 0.94
}
```

#### Failure Modes
- Low confidence score (e.g., < 0.6) → Router Agent may flag for retry or route to "other" with a generic template
- Intent is "other" → Draft Writer uses a general-purpose template

---

### 4.5 Agent 3 — Tone Stylist Agent

**File:** `src/agents/tone_stylist_agent.py`

#### What It Does
Determines the precise tone parameters for the email and encodes them into a form the Draft Writer Agent can use directly. Tone is treated as a first-class, composable parameter — not an afterthought.

#### Why It Exists
Tone is one of the most nuanced and impactful dimensions of email writing. The same information ("the delivery is delayed") lands very differently when written formally vs. empathetically vs. assertively. Rather than hoping a single monolithic prompt will "figure out" the right tone, this agent makes tone an explicit, controllable variable in the pipeline.

#### Supported Tones (Minimum 3 per spec)
| Tone | Characteristics |
|---|---|
| **Formal** | Precise vocabulary, passive constructions, no contractions, full sentences |
| **Casual** | Contractions, conversational phrasing, shorter sentences, friendly openers |
| **Assertive** | Direct, active voice, clear ask, minimal hedging language |
| *(Extensible: empathetic, urgent, diplomatic, persuasive, etc.)* |

#### How It Works Technically
Uses **tokenized prompts** — a technique where tone is encoded as a set of modifier tokens/instructions that get prepended or injected into the Draft Writer's prompt:

```python
tone_tokens = {
  "formal": [
    "Use formal vocabulary and avoid contractions.",
    "Prefer passive voice where appropriate.",
    "Begin with a professional salutation.",
    "Close with 'Sincerely' or 'Best regards'."
  ],
  "casual": [
    "Use a friendly, conversational tone.",
    "Contractions are encouraged.",
    "Keep sentences short and punchy.",
    "Begin with a warm opener like 'Hope you're doing well'."
  ],
  "assertive": [
    "Use direct, active voice.",
    "State the main point in the first sentence.",
    "Avoid hedging words like 'perhaps' or 'might'.",
    "End with a clear, specific call to action."
  ]
}
```

The agent also reconciles conflicts between:
- The tone the user explicitly selected in the UI (hard preference)
- The tone hint extracted by the Input Parsing Agent (soft signal)
- The user's default tone from their profile (fallback)

Priority order: **UI selection > prompt hint > profile default**

#### Tone + Intent Interaction
Some intent/tone combinations get special handling:
- `apology + formal` → adds softening language directives
- `outreach + assertive` → adds hook-writing instructions
- `follow-up + casual` → adds reference to prior contact instructions

This cross-product logic may live in a lookup table within this agent or in `data/tone_samples/`.

#### Output
Updates the shared state with:
```python
{
  "tone": "formal",
  "tone_directives": [
    "Use formal vocabulary and avoid contractions.",
    "Prefer passive voice where appropriate.",
    ...
  ],
  "tone_sample_ref": "data/tone_samples/formal_apology.txt"
}
```

#### Failure Modes
- Unrecognized tone string → defaults to "formal" (safest for professional contexts)
- Conflicting tone signals → resolves via priority order, logs the decision

---

### 4.6 Agent 4 — Draft Writer Agent

**File:** `src/agents/draft_writer_agent.py`

#### What It Does
This is the core generative agent — it produces the actual email text. It receives the fully specified context (intent, tone directives, parsed constraints) and generates a structured, coherent email draft.

#### Why It Exists
Generation is separated from every other concern. By the time this agent runs, all the hard decisions (what kind of email? what tone? what constraints?) have already been made. This agent just needs to write well.

#### What It Receives (Inputs from State)
- `parsed_context` — recipient, subject, constraints
- `intent` — drives structural template selection
- `tone_directives` — the exact tone instructions from the Tone Stylist
- `tone_sample_ref` — (optional) a few-shot example of a well-written email in this tone/intent combo

#### How It Works Technically
Constructs a **composite prompt** by assembling all upstream signals:

```
System: You are a professional email writer. Follow these tone rules strictly:
{tone_directives}

Task: Write a {intent} email with the following context:
- Recipient: {recipient}
- Subject: {subject_hint}
- Constraints: {constraints}

Structure the email with:
1. Subject line
2. Opening / salutation
3. Body (2-3 paragraphs)
4. Closing / call-to-action
5. Sign-off

Reference example (tone/style guide):
{tone_sample}
```

Uses **function calling** to enforce output structure — the LLM is required to return a JSON with fields:
```python
{
  "subject_line": str,
  "salutation": str,
  "body_paragraphs": list[str],
  "closing": str,
  "sign_off": str
}
```

This structured output ensures the Personalization Agent and Review Agent can operate on specific parts of the email (e.g., inject name into salutation, check closing for tone alignment) rather than parsing raw text.

#### Template Engine Role
The `Email Template Engine` (LangChain + Function Calling) lives primarily in this agent. LangChain's prompt templates are used to compose the final prompt string from parts, and function calling is used to get structured output back.

#### Output
Adds to shared state:
```python
{
  "draft": {
    "subject_line": "Follow-Up Regarding Project Timeline",
    "salutation": "Dear Mr. Smith,",
    "body_paragraphs": ["...", "...", "..."],
    "closing": "Please let me know if you have any questions.",
    "sign_off": "Sincerely,\n[Name]"
  }
}
```

#### Failure Modes
- LLM returns malformed JSON → function calling retry (up to N times), then Router Agent handles fallback
- Output is too short/long → Review Agent catches this and may trigger a retry with length constraints added

---

### 4.7 Agent 5 — Personalization Agent

**File:** `src/agents/personalization_agent.py`

#### What It Does
Takes the generic draft from the Draft Writer and injects user-specific and recipient-specific context to make it feel personal and contextually aware. This is the **memory consumer** of the pipeline.

#### Why It Exists
A draft written without personalization is generic. "Dear Client" vs. "Dear Sarah" is a trivial example, but deeper personalization includes: referencing a previous conversation, using the sender's known writing quirks, incorporating company-specific terminology, or adapting to the relationship history with the recipient.

#### Data Sources
Reads from:
1. **`memory/user_profiles.json`** — the persistent user profile store
2. **Current session state** — anything the Input Parsing Agent extracted about the recipient

The user profile schema likely includes:
```json
{
  "user_id": "vinit_123",
  "name": "Vinit",
  "company": "Acme Corp",
  "role": "Software Engineer",
  "default_tone": "formal",
  "sign_off_preference": "Best regards",
  "prior_drafts": [
    {
      "date": "2026-03-01",
      "intent": "follow-up",
      "recipient": "client_xyz",
      "draft_summary": "..."
    }
  ],
  "writing_style_notes": "Prefers concise emails, rarely uses exclamation marks"
}
```

#### What It Injects
| Location in Email | What Gets Injected |
|---|---|
| Salutation | Recipient's actual name if known |
| Body | References to prior interactions ("As discussed in our last call...") |
| Body | Company/role-specific context ("Given our Q1 targets...") |
| Sign-off | User's preferred sign-off phrase |
| Tone calibration | User's writing style notes (small adjustments to match their voice) |

#### How It Works Technically
Two-phase approach:
1. **Slot-filling**: Direct substitutions — replace `[Name]` with actual name, `[Company]` with company name.
2. **LLM-assisted rewriting**: For more subtle personalization (matching voice, adding references to prior context), sends the draft and profile notes to an LLM with instructions to "rewrite to match this person's style without changing the content."

The second phase is optional and only runs if the user profile has sufficient writing history (prior drafts) to calibrate against.

#### Output
Adds to shared state:
```python
{
  "personalized_draft": {
    "subject_line": "Follow-Up Regarding the March Delivery Timeline",
    "salutation": "Dear Sarah,",
    "body_paragraphs": ["As we discussed last week...", "...", "..."],
    "closing": "Looking forward to your response.",
    "sign_off": "Best regards,\nVinit"
  }
}
```

#### Failure Modes
- No user profile found → skips LLM-assisted rewriting, only does slot-filling with whatever is in state
- Prior drafts not found → skips historical reference injection

---

### 4.8 Agent 6 — Review & Validator Agent

**File:** `src/agents/review_agent.py`

#### What It Does
Acts as the **quality gate** of the pipeline. Before the draft reaches the user, this agent audits it across multiple dimensions and either approves it or flags it for retry.

#### Why It Exists
LLMs are not perfectly reliable. Even with well-crafted prompts, a draft can be: grammatically incorrect, tonally inconsistent with what was requested, contextually incoherent, or structurally incomplete. This agent catches those failures before they reach the user, maintaining output quality without requiring human review on every draft.

#### What It Checks

**1. Grammar & Fluency**
- Detects broken sentences, subject-verb disagreements, tense inconsistencies
- Can use a grammar-checking LLM prompt or a dedicated grammar tool (e.g., LanguageTool API)

**2. Tone Alignment**
- Compares the actual tone of the generated draft against the requested tone
- Example: If tone was "formal" but the draft contains "Hey there!" — that's a tone mismatch
- Uses an LLM-as-judge pattern: "Does the following email match a formal tone? Rate 1-10 and explain."

**3. Contextual Coherence**
- Does the email body actually address the stated subject?
- Are all required components present (subject line, body, closing)?
- Is the draft free of hallucinated facts (names, dates, events not in the input context)?

**4. Length & Structure**
- Is the email within reasonable length bounds?
- Does it follow the expected structure for its intent type?

#### How It Works Technically
Uses a **multi-criteria evaluation prompt** — a single LLM call that evaluates all dimensions at once and returns a structured verdict:

```python
{
  "grammar_score": 0.95,       # 0-1
  "tone_alignment_score": 0.88, # 0-1
  "coherence_score": 0.91,     # 0-1
  "structure_complete": True,
  "issues": ["Closing paragraph is too short"],
  "verdict": "PASS"             # or "FAIL"
}
```

**Pass threshold**: All scores above ~0.8 and no critical structural issues → PASS
**Fail**: Any score below threshold or critical issue → FAIL with specific issue list

#### What Happens on Failure
The `verdict: "FAIL"` result is written to the shared state along with the `issues` list. The Router Agent (next in the graph) reads this and routes flow back to the Draft Writer Agent, passing the issues as additional constraints:

```
"Retry draft. Issues found: closing paragraph too short, add a clear call-to-action."
```

This retry loop is bounded by `retry_count` in the state (likely max 2-3 retries) to prevent infinite loops.

#### Output
Adds to shared state:
```python
{
  "review_result": {
    "grammar_score": 0.95,
    "tone_alignment_score": 0.88,
    "coherence_score": 0.91,
    "structure_complete": True,
    "issues": [],
    "verdict": "PASS"
  }
}
```

#### Failure Modes
- Persistent failures after max retries → Router Agent escalates to a fallback model (e.g., switches from GPT-4 to Claude via MCP) and tries once more
- Review Agent itself errors → fails open (draft passes through) to avoid blocking the user

---

### 4.9 Agent 7 — Routing & Memory Agent

**File:** `src/agents/router_agent.py`

#### What It Does
This is the **control plane agent** — it does two distinct jobs:
1. **Routing**: Decides what happens next based on the review result (pass → output, fail → retry, critical fail → fallback model)
2. **Memory**: Persists the final draft and updates the user profile for future sessions

It is both the **memory producer** and the **pipeline orchestrator** that closes the feedback loop.

#### Why It Exists
Routing logic and memory management should not live inside individual agents — that would create tight coupling. A dedicated agent that owns these cross-cutting concerns keeps the other agents pure and focused.

#### Routing Logic

```
if review_result.verdict == "PASS":
    → finalize draft, proceed to output
elif retry_count < MAX_RETRIES:
    → increment retry_count
    → pass issues back to Draft Writer Agent
    → re-run from Draft Writer
elif retry_count >= MAX_RETRIES:
    → trigger MCP fallback: switch LLM model
    → re-run from Draft Writer with new model
    → if still failing: surface draft to user with warning flag
```

The **MCP fallback** is significant: this agent reads `config/mcp.yaml` to determine the fallback model chain (e.g., GPT-4 → Claude → Cohere) and instructs the LLM client wrappers in `integrations/` to switch providers. This is the practical application of the Model Control Plane.

#### Memory Operations
On successful completion, this agent:

1. **Logs the draft** — appends a summary of the current draft to `prior_drafts` in the user profile
2. **Updates style notes** — if the user edited the draft in the UI, the edits are diff'd against the original to extract style preferences ("user shortened the closing", "user changed passive to active voice")
3. **Persists profile** — writes the updated profile back to `memory/user_profiles.json`

This creates the **continuous personalization loop**: every email the user generates makes future emails more accurate to their voice and preferences.

#### How It Works Technically
Unlike other agents, this one is mostly **deterministic logic** (routing decisions, JSON read/write) with a small LLM component for the style diff analysis. The LangGraph conditional edge mechanism is used here:

```python
def routing_decision(state):
    if state["review_result"]["verdict"] == "PASS":
        return "output"
    elif state["retry_count"] < MAX_RETRIES:
        return "draft_writer"   # loop back
    else:
        return "fallback"       # switch model

graph.add_conditional_edges("router_agent", routing_decision, {
    "output": END,
    "draft_writer": "draft_writer_agent",
    "fallback": "fallback_handler"
})
```

#### Output
- Final `final_draft` string in shared state (assembled from structured draft object into a clean email text)
- Updated `user_profiles.json` on disk
- Routing signal consumed internally by LangGraph

#### Failure Modes
- JSON write failure → logs error, continues (memory update is non-blocking — don't fail the user's draft over a persistence error)
- All retries and fallbacks exhausted → surfaces draft with a "Review suggested" warning in the UI

---

### 4.10 How Agents Interact: The Data Flow Summary

```
Agent                   Reads from State          Writes to State
─────────────────────────────────────────────────────────────────────
Input Parsing           raw_input                 parsed_context
Intent Detection        parsed_context            intent, intent_confidence
Tone Stylist            parsed_context, intent    tone, tone_directives
Draft Writer            intent, tone_directives,  draft (structured)
                        parsed_context
Personalization         draft, user_profile       personalized_draft
Review & Validator      personalized_draft,       review_result
                        tone, intent
Routing & Memory        review_result,            final_draft, retry_count
                        retry_count               → updates user_profiles.json
```

### 4.11 Retry & Fallback Flow

```
Draft Writer → Personalization → Review
                                   │
                            ┌──────┴──────┐
                           PASS          FAIL
                            │              │
                            ▼         retry_count < MAX?
                       Router Agent         │
                            │           YES → back to Draft Writer
                            ▼              (with issue list added to prompt)
                       Final Draft      NO  → MCP Fallback
                                               (switch model)
                                               → retry once more
                                               → if still fail: output with warning
```

---

## 5. Project Deliverables & Build Schedule

### Week 1: Core Agent Workflow
- Implement **Input Parsing Agent** for context normalization
- Build **Intent Detection Agent** using classification prompts
- Set up **Draft Writer Agent** with tone-aware templates
- Integrate **Tone Stylist** for 3 tone modes (formal, casual, assertive)
- Store personalization data (company, name, style) in local JSON

### Week 2: Finalization + UI + Memory
- Add **Review Agent** for tone and grammar checks
- Implement **Router Agent** with fallback and retry logic via LangGraph
- Build **Streamlit UI** with:
  - Context + tone selector
  - Email preview and editor
  - Export to email/PDF (optional)
- Log user draft edits to personalize subsequent suggestions
- Optional: Dockerize and deploy via Streamlit Cloud or locally

---

## 6. Learning Goals

### Agentic AI Skills
- **LangGraph / CrewAI**: Build cooperative multi-agent workflows
- **Prompt Engineering**: Create tone-flexible, context-rich prompts
- **Function Calling**: Use structured outputs for consistent draft formatting
- **Personalization + Memory**: Persist preferences and reuse tone context across sessions
- **MCP Routing**: Dynamically route across models for fallback or cost optimization

### System Design Skills
- Scalable agent modules following the intent → tone → draft → review pipeline
- Reliable fallback/validation layers for output safety
- Customizable UI components for human-in-the-loop feedback
- Modular design for easy tone or template extension

---

## 7. Codebase Structure

```
email_assistant/
├── src/
│   ├── agents/
│   │   ├── input_parser_agent.py
│   │   ├── intent_detection_agent.py
│   │   ├── tone_stylist_agent.py
│   │   ├── draft_writer_agent.py
│   │   ├── personalization_agent.py
│   │   ├── review_agent.py
│   │   └── router_agent.py
│   ├── workflow/
│   │   └── langgraph_flow.py
│   ├── ui/
│   │   └── streamlit_app.py
│   ├── memory/
│   │   └── user_profiles.json
│   └── integrations/
│       ├── openai_client.py
│       └── cohere_client.py
├── data/
│   └── tone_samples/
├── config/
│   └── mcp.yaml
├── Dockerfile
└── README.md
```

### Notable Structural Observations
- **`workflow/langgraph_flow.py`** is the central orchestration file — this is where the agent DAG/graph is defined and all agents are wired together.
- **`memory/user_profiles.json`** is the persistence layer — simple JSON for now, with a clear upgrade path to MongoDB or vector stores.
- **`integrations/`** holds LLM client wrappers, enabling easy model swapping. Both OpenAI and Cohere clients are included, reflecting the MCP multi-model strategy.
- **`config/mcp.yaml`** centralizes routing and model configuration, separating infrastructure concerns from agent logic.
- **`data/tone_samples/`** likely stores few-shot examples for tone calibration — a critical resource for the Tone Stylist Agent.
- A **Dockerfile** is included, indicating the project is designed to be containerized from the start.

---

## 8. Submission Requirements

### Working Prototype Must Have
- Accepts a user prompt, tone selection, and optional metadata
- Returns a complete, editable email draft
- UI featuring:
  - Tone & intent dropdowns
  - Real-time preview
  - Editable textbox with copy/export option

### Demo Video Must Show
- User prompt → email generation pipeline end-to-end
- Tone change and validation demonstration
- Optional: fallback recovery or MCP routing visualization

---

## 9. Evaluation Criteria

| Category | Weight | Details |
|---|---|---|
| Functionality | 30% | Drafts are accurate, relevant, and tone-aligned |
| Agentic Architecture | 25% | Distinct modular agents, LangGraph routing, fallback handling |
| User Experience | 20% | Intuitive UI, live preview, editable content |
| Routing & MCP | 10% | Demonstrates fallback, logs usage, or shows model switching |
| Innovation | 10% | Custom tones, template libraries, personalization memory |
| Documentation | 10% | README, agent flows, prompt logic, deployment docs |

**Total: 100%**

The weighting reveals the project's priorities: **core functionality (30%)** and **agentic architecture quality (25%)** together account for more than half the grade, signaling that a well-engineered multi-agent pipeline matters more than UI polish or novelty.

---

## 10. Key Architectural Insights & Patterns

### 1. Separation of Concerns via Specialization
Each agent does exactly one thing. The input parser doesn't write drafts; the draft writer doesn't check grammar. This mirrors solid software engineering principles and makes the pipeline debuggable and testable at each stage.

### 2. Bidirectional Memory Loop
The Personalization Agent *reads* memory and the Routing & Memory Agent *writes* memory. This creates a closed feedback loop: each email generation cycle improves the next one by capturing user edits and preferences.

### 3. Tone as a First-Class Parameter
Rather than embedding tone in a single monolithic prompt, tone is extracted as a discrete parameter handled by a dedicated agent. This enables systematic tone control (3+ modes) and easier extension to new tone profiles without touching other agents.

### 4. Multi-Model Resilience via MCP
The `mcp.yaml` config and LiteLLM/LangSmith control plane mean the system is not locked to a single LLM provider. If GPT-4 is unavailable or cost-prohibitive, the system can route to Claude or Cohere transparently.

### 5. Human-in-the-Loop by Design
The Streamlit UI includes an editable textbox — users can correct the generated draft. These edits are logged and fed back into the memory layer, making the system a **collaborative** tool rather than a fully autonomous one. This is a mature design choice that acknowledges LLM limitations.

### 6. Validation Layer
The Review & Validator Agent acts as a safety net. It checks grammar, tone alignment, and contextual coherence *before* the output reaches the user. This prevents the system from surfacing obviously poor drafts and improves perceived quality.

---

## 11. Summary

The AI-Powered Email Assistant is a well-scoped capstone project that demonstrates the full agentic AI stack: multi-agent orchestration (LangGraph), LLM integration (GPT-4, Claude, Cohere), persistent memory (JSON/MongoDB), structured output (function calling), model routing (MCP/LiteLLM), and a human-in-the-loop UI (Streamlit). The 7-agent pipeline is deliberately modular, making each component independently testable and extensible. The two-week build schedule is aggressive but achievable by front-loading the core agents and deferring UI and memory polish to week 2.
