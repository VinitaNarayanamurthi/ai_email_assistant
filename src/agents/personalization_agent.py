import json
from copy import deepcopy
from typing import Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.models.state import EmailAssistantState, EmailDraftDict

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


def apply_slot_filling(
    draft: EmailDraftDict,
    ctx: dict[str, object],
    profile: dict[str, object],
) -> tuple[EmailDraftDict, list[str]]:
    result: EmailDraftDict = deepcopy(draft)
    log: list[str] = []

    recipient_name = ctx.get("recipient_name")
    if not recipient_name:
        known_contacts = profile.get("known_contacts") or {}
        recipient_type = str(ctx.get("recipient_type") or "")
        recipient_name = known_contacts.get(recipient_type) if isinstance(known_contacts, dict) else None  # type: ignore[union-attr]

    if recipient_name and recipient_name != "[Name]":
        name_str = str(recipient_name)
        salutation = result.get("salutation") or ""
        result["salutation"] = salutation.replace("[Name]", name_str)
        log.append(f"Injected recipient name: {name_str}")

    sign_off_pref = profile.get("sign_off_preference")
    if sign_off_pref:
        result["sign_off"] = str(sign_off_pref) + ","
        log.append(f"Applied preferred sign-off: {sign_off_pref}")

    company = profile.get("company")
    if company:
        company_str = str(company)
        body = result.get("body_paragraphs") or []
        result["body_paragraphs"] = [
            p.replace("[Company]", company_str).replace("[Your Company]", company_str)
            for p in body
        ]
        if company_str in str(result.get("body_paragraphs")):
            log.append(f"Injected company name: {company_str}")

    sender_name = profile.get("name")
    if sender_name:
        current_sign_off = result.get("sign_off") or ""
        result["sign_off"] = current_sign_off.rstrip(",") + ","
        log.append(f"Sign-off set, sender name available: {sender_name}")

    return result, log


def find_prior_context(
    profile: dict[str, object],
    intent: str,
    recipient_type: str,
) -> Optional[str]:
    prior_drafts = profile.get("prior_drafts") or []
    if not prior_drafts:
        return None

    for draft in reversed(list(prior_drafts)):  # type: ignore[call-overload]
        if (
            isinstance(draft, dict)
            and draft.get("recipient_type") == recipient_type
            and draft.get("intent") == intent
        ):
            return str(draft.get("draft_summary") or "")

    for draft in reversed(list(prior_drafts)):  # type: ignore[call-overload]
        if isinstance(draft, dict) and draft.get("recipient_type") == recipient_type:
            return str(draft.get("draft_summary") or "")

    return None


def _build_voice_match_chain() -> object:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VOICE_MATCH_SYSTEM_PROMPT),
            ("human", VOICE_MATCH_USER_PROMPT),
        ]
    )
    return prompt | llm | JsonOutputParser()


def apply_voice_matching(
    draft: EmailDraftDict,
    profile: dict[str, object],
    prior_context: Optional[str],
) -> tuple[EmailDraftDict, list[str]]:
    style_notes = str(profile.get("writing_style_notes") or "")
    log: list[str] = []

    if not style_notes or len(style_notes) < 20:
        return draft, log

    try:
        chain = _build_voice_match_chain()
        refined: EmailDraftDict = chain.invoke(  # type: ignore[assignment]
            {
                "style_notes": style_notes,
                "prior_context": prior_context or "No prior interactions on record.",
                "draft_json": json.dumps(draft, indent=2),
            }
        )
        log.append("Applied LLM voice matching based on writing style notes.")
        return refined, log
    except Exception as e:
        log.append(f"Voice matching skipped (error): {str(e)}")
        return draft, log


def personalization_agent(state: EmailAssistantState) -> dict[str, object]:
    draft = state.get("draft")
    if not draft:
        return {
            "personalized_draft": draft,
            "personalization_log": ["No draft to personalize."],
        }

    ctx = state.get("parsed_context") or {}
    profile = state.get("user_profile") or {}
    intent = str(state.get("intent") or "other")
    log: list[str] = []

    updated_draft, slot_log = apply_slot_filling(
        draft,
        dict(ctx),  # type: ignore[arg-type]
        dict(profile),  # type: ignore[arg-type]
    )
    log.extend(slot_log)

    prior_context = find_prior_context(
        dict(profile),  # type: ignore[arg-type]
        intent=intent,
        recipient_type=str(ctx.get("recipient_type") or ""),
    )
    if prior_context:
        log.append(
            f"Found prior interaction context for {ctx.get('recipient_type')} + {intent}."
        )

    updated_draft, voice_log = apply_voice_matching(
        updated_draft,
        dict(profile),  # type: ignore[arg-type]
        prior_context,
    )
    log.extend(voice_log)

    return {
        "personalized_draft": updated_draft,
        "personalization_log": log,
    }
