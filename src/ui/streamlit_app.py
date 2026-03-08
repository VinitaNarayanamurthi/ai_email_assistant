import json
import os
import urllib.parse
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as st_components

from src.workflow.langgraph_flow import build_graph, run_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFILES_PATH = Path("src/memory/user_profiles.json")
TONE_OPTIONS = ["formal", "casual", "assertive"]
MODEL_OPTIONS = ["(auto)", "gpt-4o", "claude-3-5-sonnet-20241022", "cohere/command-r-plus"]
PASS_THRESHOLDS = {
    "grammar_score": 0.75,
    "tone_alignment_score": 0.70,
    "coherence_score": 0.75,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_profile_from_disk() -> dict:
    if not PROFILES_PATH.exists():
        return {"name": "", "company": "", "role": "", "default_tone": "formal"}
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles: dict = json.load(f)
    return profiles.get("default") or {}


def score_delta(score: float, key: str) -> float:
    threshold = PASS_THRESHOLDS.get(key, 0.75)
    return round(score - threshold, 2)


def clipboard_js(text: str) -> str:
    escaped = text.replace("`", "\\`").replace("$", "\\$")
    return f"""
    <script>
    async function copyText() {{
        try {{
            await navigator.clipboard.writeText(`{escaped}`);
        }} catch (e) {{
            const ta = document.createElement('textarea');
            ta.value = `{escaped}`;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        }}
    }}
    copyText();
    </script>
    """


# ---------------------------------------------------------------------------
# Session state initialisation — runs once per browser session
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    if "compiled_graph" not in st.session_state:
        st.session_state["compiled_graph"] = build_graph()
    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None
    if "draft_edited" not in st.session_state:
        st.session_state["draft_edited"] = ""
    if "generation_history" not in st.session_state:
        st.session_state["generation_history"] = []
    if "user_profile" not in st.session_state:
        st.session_state["user_profile"] = load_profile_from_disk()
    if "copy_trigger" not in st.session_state:
        st.session_state["copy_trigger"] = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[str | None, str, bool, bool]:
    """Render the sidebar and return (tone_selection, model_override, show_scores, show_trace)."""
    st.sidebar.title("AI Email Assistant")
    st.sidebar.divider()

    # --- User profile ---
    st.sidebar.header("Your Profile")
    profile = st.session_state["user_profile"]

    name = st.sidebar.text_input("Name", value=profile.get("name", ""))
    company = st.sidebar.text_input("Company", value=profile.get("company", ""))
    role = st.sidebar.text_input("Role", value=profile.get("role", ""))

    if name != profile.get("name") or company != profile.get("company") or role != profile.get("role"):
        st.session_state["user_profile"] = {**profile, "name": name, "company": company, "role": role}

    st.sidebar.divider()

    # --- Tone selector ---
    st.sidebar.header("Tone")
    profile_default_tone = profile.get("default_tone", "formal")
    default_index = TONE_OPTIONS.index(profile_default_tone) if profile_default_tone in TONE_OPTIONS else 0
    selected_tone = st.sidebar.radio("Select tone", options=TONE_OPTIONS, index=default_index)

    st.sidebar.divider()

    # --- Advanced options ---
    with st.sidebar.expander("Advanced Options"):
        model_choice = st.selectbox("Model override", options=MODEL_OPTIONS, index=0)
        show_scores = st.checkbox("Show review scores", value=True)
        show_trace = st.checkbox("Show agent trace", value=False)

    model_override = None if model_choice == "(auto)" else model_choice

    return selected_tone, model_override, show_scores, show_trace  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Quality scores panel
# ---------------------------------------------------------------------------


def render_quality_scores(review: dict) -> None:
    st.subheader("Quality Review")

    grammar = float(review.get("grammar_score", 0))
    tone_score = float(review.get("tone_alignment_score", 0))
    coherence = float(review.get("coherence_score", 0))

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Grammar",
        f"{grammar:.0%}",
        delta=f"{score_delta(grammar, 'grammar_score'):+.0%}",
        delta_color="normal",
    )
    col2.metric(
        "Tone Match",
        f"{tone_score:.0%}",
        delta=f"{score_delta(tone_score, 'tone_alignment_score'):+.0%}",
        delta_color="normal",
    )
    col3.metric(
        "Coherence",
        f"{coherence:.0%}",
        delta=f"{score_delta(coherence, 'coherence_score'):+.0%}",
        delta_color="normal",
    )

    issues = review.get("issues") or []
    if issues:
        with st.expander("Review notes"):
            for issue in issues:
                st.write(f"- {issue}")


# ---------------------------------------------------------------------------
# Action bar
# ---------------------------------------------------------------------------


def render_action_bar(draft_text: str, subject_line: str) -> None:
    col_copy, col_mailto, col_export, _ = st.columns([1, 1, 1, 3])

    with col_copy:
        if st.button("Copy", use_container_width=True):
            st.session_state["copy_trigger"] = True
            st.toast("Copied to clipboard!")

    with col_mailto:
        mailto_link = (
            "mailto:?subject="
            + urllib.parse.quote(subject_line)
            + "&body="
            + urllib.parse.quote(draft_text)
        )
        st.link_button("Open in Mail", mailto_link, use_container_width=True)

    with col_export:
        st.download_button(
            label="Export .txt",
            data=draft_text,
            file_name="email_draft.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # Inject clipboard JS after the button press
    if st.session_state.get("copy_trigger"):
        st_components.html(clipboard_js(draft_text), height=0)
        st.session_state["copy_trigger"] = False


# ---------------------------------------------------------------------------
# Debug trace panel
# ---------------------------------------------------------------------------


def render_debug_trace(result: dict) -> None:
    with st.expander("Agent trace (raw state)"):
        st.json(
            {
                "intent": result.get("intent"),
                "intent_confidence": result.get("intent_confidence"),
                "intent_fallback": result.get("intent_fallback"),
                "tone": result.get("tone"),
                "tone_directives": result.get("tone_directives"),
                "tone_resolution_log": result.get("tone_resolution_log"),
                "parse_error": result.get("parse_error"),
                "draft_error": result.get("draft_error"),
                "retry_count": result.get("retry_count"),
                "active_model": result.get("active_model"),
                "pipeline_status": result.get("pipeline_status"),
                "pipeline_warning": result.get("pipeline_warning"),
                "personalization_log": result.get("personalization_log"),
                "review_result": result.get("review_result"),
            }
        )


# ---------------------------------------------------------------------------
# Generation history
# ---------------------------------------------------------------------------


def render_history() -> None:
    history: list = st.session_state["generation_history"]
    if not history:
        return

    st.divider()
    st.subheader("This Session")

    for i, entry in enumerate(reversed(history)):
        label = f"Draft {len(history) - i}: {entry['prompt'][:60]}{'...' if len(entry['prompt']) > 60 else ''}"
        with st.expander(label):
            st.text(entry["draft"])
            meta_parts = []
            if entry.get("intent"):
                meta_parts.append(f"Intent: {entry['intent']}")
            if entry.get("tone"):
                meta_parts.append(f"Tone: {entry['tone']}")
            if entry.get("model"):
                meta_parts.append(f"Model: {entry['model']}")
            if meta_parts:
                st.caption("  |  ".join(meta_parts))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="AI Email Assistant",
        page_icon="envelope",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    selected_tone, model_override, show_scores, show_trace = render_sidebar()

    # -----------------------------------------------------------------------
    # Prompt input
    # -----------------------------------------------------------------------
    st.header("AI Email Assistant")
    st.caption("Describe the email you need. Use a full sentence for a new draft, or a short instruction to refine the current draft.")

    raw_input = st.text_area(
        "Your prompt",
        placeholder=(
            "e.g. Write a formal apology to my client Sarah about the delayed shipment. "
            "Keep it concise and professional."
        ),
        height=120,
        label_visibility="collapsed",
    )

    col_generate, col_clear, col_spacer = st.columns([1, 1, 6])
    with col_generate:
        generate_clicked = st.button("Generate", type="primary", use_container_width=True)
    with col_clear:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state["pipeline_result"] = None
        st.session_state["draft_edited"] = ""
        st.rerun()

    # -----------------------------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------------------------
    if generate_clicked:
        if not raw_input.strip():
            st.warning("Please enter a prompt before generating.")
        else:
            prior_result = st.session_state.get("pipeline_result")
            prior_edited = st.session_state.get("draft_edited", "")

            with st.status("Generating email...", expanded=True) as status_box:
                st.write("Parsing your input...")
                st.write("Detecting intent and tone...")
                st.write("Writing draft...")
                st.write("Personalising and reviewing...")

                try:
                    # Build initial state additions
                    extra: dict = {}
                    if model_override:
                        extra["active_model"] = model_override
                    if prior_edited and prior_result:
                        extra["user_edited_draft"] = prior_edited

                    # Merge sidebar profile values into user_profile
                    profile = st.session_state["user_profile"]

                    result = run_pipeline(
                        raw_input=raw_input,
                        user_id=str(profile.get("user_id", "default")),
                        ui_tone_selection=selected_tone,
                        thread_id=st.session_state["thread_id"],
                        compiled_graph=st.session_state["compiled_graph"],
                    )

                    # Inject any model override into the result for display
                    if model_override and not result.get("active_model"):
                        result = {**result, "active_model": model_override}

                    st.session_state["pipeline_result"] = result
                    final_draft = str(result.get("final_draft") or "")
                    st.session_state["draft_edited"] = final_draft

                    # Append to history
                    history_entry = {
                        "prompt": raw_input,
                        "draft": final_draft,
                        "intent": result.get("intent", ""),
                        "tone": result.get("tone", ""),
                        "model": result.get("active_model", ""),
                    }
                    st.session_state["generation_history"].append(history_entry)

                    status_box.update(label="Done!", state="complete", expanded=False)

                except Exception as exc:
                    status_box.update(label="Error", state="error", expanded=True)
                    st.error(f"Unexpected error: {exc}")
                    st.stop()

    # -----------------------------------------------------------------------
    # Results panel
    # -----------------------------------------------------------------------
    result: dict | None = st.session_state.get("pipeline_result")

    if result is None:
        st.info("Enter a prompt above and click Generate to create your email.")
        return

    # Parse / draft hard errors
    parse_error = result.get("parse_error")
    draft_error = result.get("draft_error")
    pipeline_warning = result.get("pipeline_warning")

    if draft_error:
        st.error(f"Draft generation failed: {draft_error}")
        if show_trace:
            render_debug_trace(result)
        return

    if parse_error and not result.get("final_draft"):
        st.error(f"Could not parse your input: {parse_error}")
        st.info("Try being more specific — include who the email is for and what it is about.")
        if show_trace:
            render_debug_trace(result)
        return

    # Soft warnings
    if parse_error:
        st.warning(f"Parsing note: {parse_error}")

    if pipeline_warning:
        st.warning(
            f"Quality note: {pipeline_warning}  "
            "The draft below is the best available output after all retries."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Email preview (editable)
    # -----------------------------------------------------------------------
    st.subheader("Your Draft")

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    intent_val = str(result.get("intent") or "")
    tone_val = str(result.get("tone") or "")
    model_val = str(result.get("active_model") or "gpt-4o")
    retries_val = int(result.get("retry_count") or 0)

    if intent_val:
        col_meta1.caption(f"Intent: **{intent_val}**")
    if tone_val:
        col_meta2.caption(f"Tone: **{tone_val}**")
    col_meta3.caption(f"Model: **{model_val}**  |  Retries: **{retries_val}**")

    edited_draft = st.text_area(
        "Edit your draft",
        value=st.session_state["draft_edited"],
        height=400,
        label_visibility="collapsed",
        key="draft_editor",
    )
    st.session_state["draft_edited"] = edited_draft

    # -----------------------------------------------------------------------
    # Action bar
    # -----------------------------------------------------------------------
    personalized_draft = result.get("personalized_draft") or {}
    subject_line = str(personalized_draft.get("subject_line") or "")  # type: ignore[union-attr]
    render_action_bar(edited_draft or "", subject_line)

    # Refine hint
    st.caption(
        "To refine: type a short instruction above (e.g. *make it more concise*) and click Generate. "
        "Your edits here are saved and used to personalise future drafts."
    )

    # -----------------------------------------------------------------------
    # Quality scores
    # -----------------------------------------------------------------------
    if show_scores:
        review = result.get("review_result")
        if review and isinstance(review, dict):
            st.divider()
            render_quality_scores(review)

    # -----------------------------------------------------------------------
    # Debug trace
    # -----------------------------------------------------------------------
    if show_trace:
        st.divider()
        render_debug_trace(result)

    # -----------------------------------------------------------------------
    # Generation history
    # -----------------------------------------------------------------------
    render_history()


if __name__ == "__main__":
    main()
