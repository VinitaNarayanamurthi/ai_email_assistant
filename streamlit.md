# Streamlit UI — Implementation Plan

**File:** `src/ui/streamlit_app.py`

This document describes in detail what will be built in the Streamlit front-end for the AI Email Assistant. The UI wraps the 7-agent LangGraph pipeline exposed via `src/workflow/langgraph_flow.run_pipeline()` and provides the human-in-the-loop layer required by the project specification.

---

## 1. Overview

The Streamlit app is a single-page application that covers the complete email generation workflow:

1. User enters a natural-language prompt and configures options in the sidebar
2. The pipeline runs and streams status updates
3. A finished draft appears in an editable preview panel
4. The user can refine, copy, or export the draft
5. Any edits made by the user are fed back into the memory layer for future personalization

---

## 2. Page Layout

```
┌────────────────────────────────────────────────────────────────┐
│  SIDEBAR                   │  MAIN PANEL                       │
│  ─────────                 │  ─────────                        │
│  User Profile              │  [1] Prompt Input                 │
│    • Name                  │  [2] Pipeline Status / Progress   │
│    • Company               │  [3] Email Preview (editable)     │
│    • Role                  │  [4] Quality Scores               │
│    • Default Tone          │  [5] Action Bar                   │
│                            │      (Copy / Export / Refine)     │
│  Tone Selector             │  [6] Conversation History         │
│    • Formal                │                                   │
│    • Casual                │                                   │
│    • Assertive             │                                   │
│                            │                                   │
│  Advanced Options          │                                   │
│    • Active Model          │                                   │
│    • Max Retries           │                                   │
│    • Thread ID             │                                   │
└────────────────────────────┴───────────────────────────────────┘
```

---

## 3. Session State

All transient UI state is stored in `st.session_state` so it persists across Streamlit reruns triggered by widget interactions.

```python
st.session_state = {
    "thread_id": str,           # UUID for LangGraph conversation thread
    "compiled_graph": object,   # Compiled LangGraph — built once, reused
    "pipeline_result": dict,    # Last full pipeline output dict
    "draft_edited": str,        # User's edited version of the draft text
    "generation_history": list, # List of past (prompt, final_draft) tuples
    "user_profile": dict,       # Loaded from user_profiles.json at startup
}
```

`compiled_graph` is built once on first run using `build_graph()` and stored in session state to avoid rebuilding the LangGraph on every rerender.

`thread_id` is a UUID generated once per browser session. It is passed to `run_pipeline()` so LangGraph's `MemorySaver` checkpointer maintains conversation continuity — the user can send follow-up refinements ("make it shorter") and the pipeline knows the prior context.

---

## 4. Sidebar

### 4.1 User Profile Section

```python
st.sidebar.header("Your Profile")
user_name    = st.sidebar.text_input("Name", value=profile.get("name", ""))
user_company = st.sidebar.text_input("Company", value=profile.get("company", ""))
user_role    = st.sidebar.text_input("Role", value=profile.get("role", ""))
```

Values pre-populate from `src/memory/user_profiles.json` using the `"default"` user. Changes here update `st.session_state["user_profile"]` in memory for the current session (they are not persisted until a draft is generated, at which point the Router/Memory Agent writes them back to disk).

### 4.2 Tone Selector

```python
st.sidebar.header("Tone")
tone = st.sidebar.radio(
    "Select tone",
    options=["formal", "casual", "assertive"],
    index=0,
)
```

This maps directly to `ui_tone_selection` in the pipeline's initial state. The Tone Stylist Agent reads this field with highest priority (above any tone extracted from the prompt). A `None` value lets the agent fall back to the prompt hint or the user's profile default.

### 4.3 Advanced Options (Collapsed by Default)

```python
with st.sidebar.expander("Advanced Options"):
    active_model = st.selectbox(
        "Model override",
        options=["(auto)", "gpt-4o", "claude-3-5-sonnet-20241022", "cohere/command-r-plus"],
    )
    show_scores = st.checkbox("Show review scores", value=True)
    show_agent_trace = st.checkbox("Show agent trace", value=False)
```

- **Model override**: When set to anything other than `(auto)`, the selected model is injected into `initial_state["active_model"]`, bypassing the MCP config's `primary_model` setting. Useful for testing fallback behaviour.
- **Show review scores**: Toggles the quality score panel in the main panel.
- **Show agent trace**: Toggles a collapsed expander showing the raw `pipeline_result` dict for debugging.

---

## 5. Main Panel

### 5.1 Prompt Input Area

```python
st.header("AI Email Assistant")

raw_input = st.text_area(
    "Describe the email you need",
    placeholder=(
        "e.g. Write a formal apology to my client Sarah about the delayed shipment. "
        "Keep it concise and professional."
    ),
    height=120,
)

col_generate, col_clear = st.columns([1, 5])
with col_generate:
    generate_clicked = st.button("Generate", type="primary")
with col_clear:
    clear_clicked = st.button("Clear")
```

The text area accepts free-form natural language. The placeholder gives the user an example of a well-formed prompt. The input is passed directly to `run_pipeline()` as `raw_input`.

**Refinement detection**: If a draft already exists in session state and the new input is short (e.g., "make it more casual", "add a P.S."), the Input Parsing Agent's `is_refinement_input()` function detects this and the pipeline applies the refinement to the existing parsed context rather than starting fresh. The same thread ID ensures the checkpointer provides the prior state to the graph.

### 5.2 Pipeline Status / Progress

While the pipeline is running, a status placeholder is updated at each stage using `st.status()`:

```python
with st.status("Generating email...", expanded=True) as status:
    st.write("Parsing your input...")
    # ... pipeline runs ...
    st.write("Detecting intent...")
    st.write("Applying tone directives...")
    st.write("Writing draft...")
    st.write("Personalizing...")
    st.write("Reviewing quality...")
    status.update(label="Done!", state="complete")
```

Because `run_pipeline()` is synchronous (LangGraph runs to completion before returning), the status updates are approximated by updating the label before and after the call. If streaming is added in the future (LangGraph supports async streaming via `.astream()`), each agent completion event can be used to update the status panel in real time.

**Error display**: If `pipeline_result["pipeline_status"]` is not `"success"` or `"warning"`, or if `parse_error` is set, the error is shown as `st.error(...)` with the specific message from the agent that failed.

**Warning banner**: If `pipeline_result["pipeline_warning"]` is not `None` (meaning the pipeline exhausted all retries and fell back to the best available draft), a yellow warning callout is shown:

```python
if pipeline_result.get("pipeline_warning"):
    st.warning(
        f"Quality note: {pipeline_result['pipeline_warning']}  "
        "The draft below is the best available output after all retries."
    )
```

### 5.3 Email Preview (Editable)

The final draft is displayed in an editable text area so the user can make corrections:

```python
st.subheader("Your Draft")

draft_text = pipeline_result.get("final_draft", "")

edited_draft = st.text_area(
    "Edit your draft below",
    value=draft_text,
    height=400,
    key="draft_editor",
)

st.session_state["draft_edited"] = edited_draft
```

The text area is pre-populated with the assembled plain-text email from `router_memory_agent`'s `assemble_final_draft()`. The email structure looks like:

```
Subject: Apology for the Delayed Shipment

Dear Sarah,

We sincerely apologize for the delay in your recent shipment. ...

We are taking immediate steps to resolve this...

Please do not hesitate to contact us if you have further questions.

Best regards,
Vinit
```

Changes the user makes in this text area are captured in `st.session_state["draft_edited"]`. When the next generation runs, the Router/Memory Agent receives `user_edited_draft` in the state, diffs it against the `final_draft`, and uses `extract_style_from_edits()` to learn style preferences for future emails.

### 5.4 Quality Scores Panel

Shown when the `show_scores` checkbox is enabled. Reads from `pipeline_result["review_result"]`:

```python
if show_scores and pipeline_result.get("review_result"):
    review = pipeline_result["review_result"]

    st.subheader("Quality Review")

    col1, col2, col3 = st.columns(3)
    col1.metric("Grammar",    f"{review['grammar_score']:.0%}")
    col2.metric("Tone Match", f"{review['tone_alignment_score']:.0%}")
    col3.metric("Coherence",  f"{review['coherence_score']:.0%}")

    if review.get("issues"):
        with st.expander("Review notes"):
            for issue in review["issues"]:
                st.write(f"- {issue}")
```

`st.metric()` renders each score as a large number with a label — visually clear and compact. A score below the pass threshold (e.g., tone alignment < 0.70) is visually highlighted by passing a `delta` argument with a negative value so Streamlit colours it red.

### 5.5 Action Bar

Four action buttons appear below the draft editor:

```python
col_copy, col_mailto, col_export, col_refine = st.columns(4)
```

**Copy to Clipboard**
```python
with col_copy:
    st.button("Copy to clipboard", on_click=copy_to_clipboard, args=[edited_draft])
```
Uses `st.components.v1.html()` with a tiny JavaScript snippet to write the draft text to the clipboard. This avoids requiring any third-party clipboard library.

**Open in Mail Client**
```python
with col_mailto:
    subject_line = pipeline_result.get("personalized_draft", {}).get("subject_line", "")
    mailto_link = f"mailto:?subject={urllib.parse.quote(subject_line)}&body={urllib.parse.quote(edited_draft)}"
    st.link_button("Open in mail client", mailto_link)
```
Generates a `mailto:` URI from the subject and body so the user can open the draft directly in their default mail client (Outlook, Apple Mail, Gmail via the OS handler).

**Export as Text File**
```python
with col_export:
    st.download_button(
        label="Export .txt",
        data=edited_draft,
        file_name="email_draft.txt",
        mime="text/plain",
    )
```
`st.download_button()` triggers a browser download with no additional dependencies.

**Refine Prompt**
```python
with col_refine:
    if st.button("Submit refinement"):
        # Reuses the same thread_id so the pipeline sees prior context
        # The raw_input text area is focused — user types a refinement instruction
        st.info("Type your refinement in the prompt box above and click Generate.")
```
This guides the user to type a short refinement instruction (e.g., "make it more concise") in the main prompt box and hit Generate again. Because the same `thread_id` is used, the pipeline's Input Parsing Agent detects it as a refinement and applies it to the existing parsed context without rebuilding from scratch.

### 5.6 Pipeline Debug Trace (Collapsed)

When `show_agent_trace` is enabled in Advanced Options:

```python
if show_agent_trace:
    with st.expander("Agent trace (raw state)"):
        st.json({
            "intent": pipeline_result.get("intent"),
            "intent_confidence": pipeline_result.get("intent_confidence"),
            "tone": pipeline_result.get("tone"),
            "tone_directives": pipeline_result.get("tone_directives"),
            "parse_error": pipeline_result.get("parse_error"),
            "draft_error": pipeline_result.get("draft_error"),
            "retry_count": pipeline_result.get("retry_count"),
            "active_model": pipeline_result.get("active_model"),
            "pipeline_status": pipeline_result.get("pipeline_status"),
            "pipeline_warning": pipeline_result.get("pipeline_warning"),
            "review_result": pipeline_result.get("review_result"),
        })
```

This is developer-facing and surfaces the full intermediate state, making it easy to verify that each agent produced the expected output. It is collapsed by default so it does not clutter the user experience.

### 5.7 Generation History

A record of the current session's generation history is shown at the bottom of the page:

```python
st.subheader("This Session")

for i, entry in enumerate(reversed(st.session_state["generation_history"])):
    with st.expander(f"Draft {len(history) - i}: {entry['prompt'][:60]}..."):
        st.text(entry["draft"])
        st.caption(f"Intent: {entry['intent']}  |  Tone: {entry['tone']}  |  Model: {entry['model']}")
```

Each collapsed expander shows a previous prompt and the draft it produced, along with the detected intent, resolved tone, and the model that was used. This lets the user compare outputs from different prompts or tone selections.

---

## 6. Data Flow: UI to Pipeline and Back

```
User types prompt
        │
        ▼
st.session_state["user_profile"]    ←── user_profiles.json (loaded at startup)
st.sidebar tone selection           ←── ui_tone_selection
        │
        ▼
run_pipeline(
    raw_input=raw_input,
    user_id="default",
    ui_tone_selection=tone,
    thread_id=st.session_state["thread_id"],
    compiled_graph=st.session_state["compiled_graph"],
)
        │
        ▼
pipeline_result = {
    "final_draft": str,
    "personalized_draft": EmailDraftDict,
    "review_result": ReviewResultDict,
    "intent": str,
    "tone": str,
    "active_model": str,
    "retry_count": int,
    "pipeline_status": str,
    "pipeline_warning": str | None,
    "parse_error": str | None,
}
        │
        ▼
Display in st.text_area (editable)
        │
User edits draft
        │
        ▼
st.session_state["draft_edited"] = edited_text
        │
(on next generation)
        ▼
initial_state["user_edited_draft"] = st.session_state["draft_edited"]
        │
Router/Memory Agent diffs edited vs. original,
updates writing_style_notes in user_profiles.json
```

---

## 7. State Persistence Between Generations

The Streamlit app uses two levels of persistence:

| Level | Mechanism | Scope |
|---|---|---|
| In-session UI state | `st.session_state` | Current browser session only |
| LangGraph conversation memory | `MemorySaver` + `thread_id` | Current browser session only |
| User profile and style notes | `src/memory/user_profiles.json` | Permanent (across sessions) |
| Generation history (UI) | `st.session_state["generation_history"]` | Current browser session only |

The JSON profile file is the only durable store. All other state resets when the page is refreshed or the Streamlit server restarts. For production use, the `SqliteSaver` checkpointer (already supported in `build_graph(use_sqlite=True)`) can be enabled to persist LangGraph conversation state across restarts.

---

## 8. Error Handling in the UI

| Scenario | UI Behaviour |
|---|---|
| Empty prompt submitted | `st.warning("Please enter a prompt.")` — generation does not run |
| `parse_error` set (e.g., input too vague) | `st.error(parse_error)` — draft area is hidden |
| `draft_error` set (LLM failed after retries) | `st.error(draft_error)` — draft area is hidden |
| `pipeline_warning` set (retries exhausted, best-effort draft) | `st.warning(pipeline_warning)` — draft is still shown |
| Exception raised by `run_pipeline()` | `st.error("Unexpected error: ...")` with the exception message |

All errors are surfaced to the user with plain-English messages. Stack traces are only shown in the debug trace panel (collapsed) or in the terminal running the Streamlit server.

---

## 9. Running the App

```bash
# Install Streamlit (if not already installed)
uv add streamlit

# Run the app
uv run streamlit run src/ui/streamlit_app.py
```

The app opens at `http://localhost:8501` by default.

An `.env` file (not committed to version control) must contain:
```
OPENAI_API_KEY=sk-...
```
The app reads this via `os.environ` or `python-dotenv`. Without a valid key, the pipeline will fail when the Draft Writer Agent attempts to call the OpenAI API.

---

## 10. File Structure for the UI Module

```
src/
└── ui/
    └── streamlit_app.py     # Single-file Streamlit application
```

The app imports from:
- `src.workflow.langgraph_flow` — `build_graph`, `run_pipeline`
- `src.memory.user_profiles` — profile loading utilities (or reads JSON directly)
- Standard library: `uuid`, `urllib.parse`, `json`, `os`

No additional UI framework dependencies beyond `streamlit` itself are required.
