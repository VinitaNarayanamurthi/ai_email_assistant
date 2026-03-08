import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.models.state import EmailAssistantState, UserProfileDict

CONFIG_PATH = Path("config/mcp.yaml")
PROFILES_PATH = Path("src/memory/user_profiles.json")

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


def _load_mcp_config() -> dict[str, object]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            loaded = yaml.safe_load(f)
            return loaded if isinstance(loaded, dict) else {}
    return {}


_mcp_config: Optional[dict[str, object]] = None


def get_mcp_config() -> dict[str, object]:
    global _mcp_config
    if _mcp_config is None:
        _mcp_config = _load_mcp_config()
    return _mcp_config


def get_primary_model() -> str:
    return str(get_mcp_config().get("primary_model") or "gpt-4o")


def get_fallback_chain() -> list[str]:
    chain = get_mcp_config().get("fallback_chain") or []
    return [str(m) for m in chain]  # type: ignore[union-attr]


def get_max_retries() -> int:
    val = get_mcp_config().get("max_retries") or 3
    return int(val) if isinstance(val, (int, float)) else 3


def get_fallback_on_retry() -> int:
    val = get_mcp_config().get("fallback_on_retry") or 2
    return int(val) if isinstance(val, (int, float)) else 2


def determine_next_action(state: EmailAssistantState) -> str:
    verdict = str((state.get("review_result") or {}).get("verdict") or "FAIL")
    retry_count = int(state.get("retry_count") or 0)
    active_model = str(state.get("active_model") or get_primary_model())
    fallback_chain = get_fallback_chain()
    fallback_on_retry = get_fallback_on_retry()

    if verdict == "PASS":
        return "success"

    if retry_count < fallback_on_retry:
        return "retry"

    current_index = (
        fallback_chain.index(active_model) if active_model in fallback_chain else -1
    )
    next_index = current_index + 1
    if next_index < len(fallback_chain):
        return "fallback"

    return "warning"


def assemble_final_draft(draft: dict[str, object], sender_name: str = "") -> str:
    lines: list[str] = []

    subject = str(draft.get("subject_line") or "")
    if subject:
        lines.append(f"Subject: {subject}")
        lines.append("")

    salutation = str(draft.get("salutation") or "")
    if salutation:
        lines.append(salutation)
        lines.append("")

    body = draft.get("body_paragraphs") or []
    for para in body:  # type: ignore[union-attr]
        para_str = str(para).strip()
        if para_str:
            lines.append(para_str)
            lines.append("")

    closing = str(draft.get("closing") or "")
    if closing:
        lines.append(closing)
        lines.append("")

    sign_off = str(draft.get("sign_off") or "")
    if sign_off:
        lines.append(sign_off)
        if sender_name:
            lines.append(sender_name)

    return "\n".join(lines).strip()


def log_draft_to_profile(
    profile: dict[str, object], state: EmailAssistantState
) -> dict[str, object]:
    profile = deepcopy(profile)
    ctx = state.get("parsed_context") or {}
    draft = state.get("personalized_draft") or {}

    assembled = assemble_final_draft(dict(draft))  # type: ignore[arg-type]
    summary = assembled[:200] + "..." if len(assembled) > 200 else assembled

    entry = {
        "date": str(date.today()),
        "intent": str(state.get("intent") or "other"),
        "recipient_type": str(ctx.get("recipient_type") or "unknown"),
        "recipient_name": ctx.get("recipient_name"),
        "subject_hint": str(ctx.get("subject_hint") or ""),
        "draft_summary": summary,
    }

    prior_drafts = list(profile.get("prior_drafts") or [])  # type: ignore[arg-type]
    prior_drafts.append(entry)
    profile["prior_drafts"] = prior_drafts[-20:]
    return profile


def _build_style_diff_chain() -> object:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", STYLE_DIFF_SYSTEM_PROMPT),
            (
                "human",
                "Original:\n{original}\n\nEdited:\n{edited}\n\nReturn a JSON list of style observations.",
            ),
        ]
    )
    return prompt | llm | JsonOutputParser()


def extract_style_from_edits(original: str, edited: str) -> list[str]:
    if not original or not edited or original.strip() == edited.strip():
        return []

    chain = _build_style_diff_chain()

    try:
        observations = chain.invoke({"original": original, "edited": edited})  # type: ignore[attr-defined]
        return observations if isinstance(observations, list) else []
    except Exception:
        return []


def update_style_notes(
    profile: dict[str, object], new_observations: list[str]
) -> dict[str, object]:
    profile = deepcopy(profile)
    existing = str(profile.get("writing_style_notes") or "")
    if new_observations:
        combined = existing + " " + " ".join(new_observations)
        profile["writing_style_notes"] = combined.strip()
    return profile


def record_recurring_issues(
    profile: dict[str, object], review_history: list[list[str]]
) -> dict[str, object]:
    all_issues = [issue for cycle in review_history for issue in cycle]
    tone_issues = [
        i for i in all_issues if "tone" in i.lower() or "contraction" in i.lower()
    ]

    if len(tone_issues) >= 2:
        note = (
            "Note: LLM tends to use informal language in formal drafts for this user. "
            "Add explicit anti-contraction reminder."
        )
        existing = str(profile.get("writing_style_notes") or "")
        if note not in existing:
            profile = deepcopy(profile)
            profile["writing_style_notes"] = (existing + " " + note).strip()
    return profile


def save_user_profile(user_id: str, profile: dict[str, object]) -> None:
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if PROFILES_PATH.exists():
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            all_profiles: dict[str, object] = json.load(f)
    else:
        all_profiles = {}

    all_profiles[user_id] = profile

    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, indent=2, ensure_ascii=False)


def router_memory_agent(state: EmailAssistantState) -> dict[str, object]:
    action = determine_next_action(state)
    profile: dict[str, object] = dict(state.get("user_profile") or {})  # type: ignore[arg-type]
    user_id = str(profile.get("user_id") or "default")

    if action == "success":
        draft_raw = state.get("personalized_draft") or state.get("draft") or {}
        sender_name = str(profile.get("name") or "")
        final_text = assemble_final_draft(dict(draft_raw), sender_name)  # type: ignore[arg-type]
        updated_profile = deepcopy(profile)

        updated_profile = log_draft_to_profile(updated_profile, state)

        user_edited = state.get("user_edited_draft")
        if user_edited:
            style_obs = extract_style_from_edits(final_text, user_edited)
            updated_profile = update_style_notes(updated_profile, style_obs)

        ctx = state.get("parsed_context") or {}
        recipient_type = ctx.get("recipient_type")
        recipient_name = ctx.get("recipient_name")
        if recipient_type and recipient_name:
            raw_contacts = updated_profile.get("known_contacts")
            known_contacts: dict[str, str] = (
                {str(k): str(v) for k, v in raw_contacts.items()}  # type: ignore[union-attr]
                if isinstance(raw_contacts, dict)
                else {}
            )
            known_contacts[str(recipient_type)] = str(recipient_name)
            updated_profile["known_contacts"] = known_contacts  # type: ignore[assignment]

        tone = str(state.get("tone") or "")
        intent = str(state.get("intent") or "")
        review = state.get("review_result") or {}
        tone_score = float(review.get("tone_alignment_score") or 0.0)  # type: ignore[arg-type]
        if tone and intent and tone_score >= 0.90:
            sample_path = Path(f"data/tone_samples/{tone}_{intent}_user.txt")
            try:
                sample_path.write_text(final_text, encoding="utf-8")
            except Exception:
                pass

        prior_review_issues: list[str] = list(state.get("prior_review_issues") or [])  # type: ignore[arg-type]
        retry_count = int(state.get("retry_count") or 0)
        if retry_count >= 1 and prior_review_issues:
            updated_profile = record_recurring_issues(updated_profile, [prior_review_issues])

        try:
            save_user_profile(user_id, updated_profile)
        except Exception:
            pass

        return {
            "final_draft": final_text,
            "user_profile": updated_profile,
            "pipeline_status": "success",
            "pipeline_warning": None,
        }

    if action == "retry":
        return {
            "retry_count": int(state.get("retry_count") or 0) + 1,
            "retry_issues": list((state.get("review_result") or {}).get("issues") or []),  # type: ignore[arg-type]
            "pipeline_status": "retrying",
        }

    if action == "fallback":
        current_model = str(state.get("active_model") or get_primary_model())
        fallback_chain = get_fallback_chain()
        current_index = (
            fallback_chain.index(current_model) if current_model in fallback_chain else -1
        )
        next_model = fallback_chain[current_index + 1]

        return {
            "active_model": next_model,
            "retry_count": int(state.get("retry_count") or 0) + 1,
            "retry_issues": list((state.get("review_result") or {}).get("issues") or []),  # type: ignore[arg-type]
            "pipeline_status": "fallback",
        }

    draft_raw = state.get("personalized_draft") or state.get("draft") or {}
    sender_name = str(profile.get("name") or "")
    final_text = assemble_final_draft(dict(draft_raw), sender_name)  # type: ignore[arg-type]

    return {
        "final_draft": final_text,
        "pipeline_status": "warning",
        "pipeline_warning": (
            "This draft could not be fully validated after multiple attempts. "
            "Please review it carefully before sending."
        ),
    }
